import streamlit as st
import cv2
import numpy as np
import ezdxf
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import requests
import math
import io

# Load the YOLO model
model = YOLO("best (6).pt")  # Replace with your custom-trained YOLO model
model_door_win = YOLO("best door and window.pt")
model_wall = YOLO("latest walls.pt")

# Conversion factors to millimeters based on DXF units

def get_max_dimension_from_dxf(uploaded_file):
    """
    Extract the maximum dimension from a DXF file uploaded as a file-like object and convert it to millimeters.

    Args:
        uploaded_file (file-like object): Uploaded DXF file.

    Returns:
        float: The maximum dimension in millimeters.
    """
    # Conversion factors for DXF units to millimeters
    CONVERSION_FACTORS = {
        0: 1.0,  # Unitless (default to 1.0)
        1: 25.4,  # Inches to mm
        2: 10.0,  # Feet to mm
        3: 304.8,  # Yards to mm
        4: 1000.0,  # Miles to mm
        5: 1.0,  # Millimeters
        6: 10.0,  # Centimeters to mm
        7: 1000.0,  # Meters to mm
        8: 1e6,  # Kilometers to mm
    }

    # Ensure file is not None
    if uploaded_file is None:
        raise ValueError("Error: No file was uploaded.")

    try:

        # Read the DXF file from the temporary file
        doc = ezdxf.readfile(uploaded_file)

    except IOError as io_error:
        raise ValueError(f"Error: Unable to open the uploaded file. {io_error}")
    except ezdxf.DXFStructureError as dxf_error:
        raise ValueError(f"Error: Invalid DXF file structure. {dxf_error}")

    # Access the modelspace
    msp = doc.modelspace()

    # Get the units from the DXF file
    insunits = doc.header.get('$INSUNITS', 0)  # Default to 0 (no units)
    conversion_factor = CONVERSION_FACTORS.get(insunits, 1.0)  # Default to 1.0 if units are unknown

    # Initialize variables to track the maximum dimension
    max_dimension = 0

    # Iterate through all DIMENSION entities
    for dimension in msp.query("DIMENSION"):
        # Explicitly defined dimension text
        dim_text = dimension.dxf.text
        dim_value = None

        if not dim_text:
            # If no explicit text, calculate the dimension value
            try:
                dim_value = dimension.get_measurement()
            except ezdxf.DXFStructureError as e:
                print(f"Warning: {e}")
        else:
            try:
                # Convert explicit dimension text to a float (if possible)
                dim_value = float(dim_text)
            except ValueError:
                print(f"Warning: Unable to convert dimension text '{dim_text}' to a float.")

        if dim_value is not None:
            # Update the maximum dimension if the current one is larger
            if dim_value > max_dimension:
                max_dimension = dim_value

    # Convert the maximum dimension to millimeters
    max_dimension_mm = max_dimension * conversion_factor

    return max_dimension_mm



def calculate_dimension_bounding_box_length(results, dimension_class_id):
    """
    Calculate the length of each dimension bounding box and return the longest one.

    Args:
        results: Inference results from the YOLO model.
        dimension_class_id (int): Class ID for dimensions.

    Returns:
        dict: The longest dimension bounding box and its length.
    """
    longest_bbox = None
    max_length = 0

    # Iterate over detected bounding boxes
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
        conf = box.conf[0].cpu().numpy()  # Confidence score
        class_id = int(box.cls[0].cpu().numpy())  # Class ID

        # Process only the dimension class
        if class_id != dimension_class_id:
            continue  # Skip non-dimension classes

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = xyxy

        # Calculate the width and height of the bounding box
        width = x_max - x_min
        height = y_max - y_min

        # For dimensions, the length is the larger of width or height
        length = max(width, height)

        # Update the longest bounding box if the current one is larger
        if length > max_length:
            max_length = length
            longest_bbox = {
                "class_id": class_id,
                "confidence": conf,
                "bounding_box": [x_min, y_min, x_max, y_max],
                "length": length
            }

    return longest_bbox

def find_intersection_points(box1, box2):
    """
    Find the intersection points of two bounding boxes.

    Args:
        box1 (tuple): First bounding box [x_min, y_min, x_max, y_max].
        box2 (tuple): Second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        list: Intersection points if any, otherwise an empty list.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        return [
            (inter_x_min, inter_y_min),
            (inter_x_max, inter_y_min),
            (inter_x_min, inter_y_max),
            (inter_x_max, inter_y_max),
        ]
    return []
def calculate_wall_length(box):
        """
        Calculate the real-world length of a wall given its bounding box.

        Args:
            box (list): Bounding box in the format [x_min, y_min, x_max, y_max].

        Returns:
            float: Wall length.
        """
        x_min, y_min, x_max, y_max = box
        return math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

def detect_wall_intersections_and_obstructions(results, wall_class_id, obstruction_class_ids):
    """
    Detect intersections between walls and extract obstructions (e.g., doors, windows) using YOLO predictions.

    Args:
        results: Inference results from the YOLO model.
        wall_class_id (int): Class ID for walls.
        obstruction_class_ids (list): List of class IDs for obstructions.

    Returns:
        tuple:
            wall_intersections (list of intersection points),
            wall_lengths (list of wall lengths),
            wall_coordinates (list of bounding box coordinates),
            obstructions (list of bounding boxes for obstructions).
    """
    # Initialize wall detections and obstruction detections
    wall_detections = []
    obstructions = []

    # Iterate over detected boxes
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Coordinates (x_min, y_min, x_max, y_max)
        conf = box.conf[0].cpu().numpy()  # Confidence score
        class_id = int(box.cls[0].cpu().numpy())  # Class ID

        # Filter detections for walls
        if class_id == wall_class_id:
            x_min, y_min, x_max, y_max = xyxy
            wall_detections.append((x_min, y_min, x_max, y_max))

        # Filter detections for obstructions
        if class_id in obstruction_class_ids:
            x_min, y_min, x_max, y_max = xyxy
            obstructions.append((x_min, y_min, x_max, y_max))

    # Initialize outputs
    wall_intersections = []
    wall_lengths = []
    wall_coordinates = []

    num_boxes = len(wall_detections)

    for i in range(num_boxes):
        box1 = wall_detections[i]
        wall_lengths.append(calculate_wall_length(box1))
        wall_coordinates.append((box1[0], box1[1], box1[2], box1[3]))

        for j in range(i + 1, num_boxes):
            box2 = wall_detections[j]
            intersection = find_intersection_points(box1, box2)
            if intersection:
                wall_intersections.extend(intersection)

    # Return the required outputs
    return wall_intersections, wall_lengths, wall_coordinates, obstructions

def place_columns_with_conditions(intersections, wall_lengths, wall_coordinates, scale_factor, obstructions, threshold=7000, min_interval=3000, max_interval=7000):
    """
    Place columns at intersections and along walls based on the original wall length after scaling,
    ensuring the distance between columns is within specified bounds and avoiding obstructions.

    Args:
        intersections (list): List of intersection points (x, y).
        wall_lengths (list): List of bounding box lengths of walls.
        wall_coordinates (list): List of bounding box coordinates for walls [(x_min, y_min, x_max, y_max)].
        scale_factor (float): Factor to convert bounding box length to actual length.
        obstructions (list): List of bounding boxes for doors/windows [(x_min, y_min, x_max, y_max)].
        threshold (float): Minimum wall length (in meters) to start placing additional columns.
        min_interval (float): Minimum distance (in meters) between additional columns.
        max_interval (float): Maximum distance (in meters) between additional columns.

    Returns:
        list: List of column coordinates (x, y) to place on the image.
    """
    columns = []

    # Add columns at wall intersections
    for intersection in intersections:
        columns.append(intersection)

    # Process each wall
    for i, (bbox_length, bbox_coords) in enumerate(zip(wall_lengths, wall_coordinates)):
        # Convert bounding box length to real-world length
        real_length = bbox_length * scale_factor

        # Add columns at wall start and end
        x_min, y_min, x_max, y_max = bbox_coords
        columns.append((x_min, y_min))  # Start of the wall
        columns.append((x_max, y_max))  # End of the wall

        # Place additional columns if wall length exceeds the threshold
        if real_length > threshold:
            # Calculate interval based on real length, ensuring it falls between min_interval and max_interval
            num_intervals = max(1, math.ceil(real_length / max_interval))  # At least one column
            interval = real_length / num_intervals
            if interval < min_interval:
                interval = min_interval  # Adjust to meet the minimum interval requirement

            # Determine the direction vector for the wall
            dx = (x_max - x_min) / bbox_length
            dy = (y_max - y_min) / bbox_length

            # Place columns at equal intervals along the wall
            for n in range(1, num_intervals):
                # Scale the interval to the bounding box
                interval_scaled = n * interval / scale_factor
                column_x = x_min + dx * interval_scaled
                column_y = y_min + dy * interval_scaled

                # Check if the column is inside an obstruction
                column_inside_obstruction = False
                for obs in obstructions:
                    obs_x_min, obs_y_min, obs_x_max, obs_y_max = obs
                    if obs_x_min <= column_x <= obs_x_max and obs_y_min <= column_y <= obs_y_max:
                        column_inside_obstruction = True

                        # Calculate distances to start and end of the obstruction
                        dist_to_start = ((column_x - obs_x_min)**2 + (column_y - obs_y_min)**2)**0.5
                        dist_to_end = ((column_x - obs_x_max)**2 + (column_y - obs_y_max)**2)**0.5

                        # Place column before or after the obstruction
                        if dist_to_start < dist_to_end:
                            # Before the obstruction
                            column_x = obs_x_min - abs(dx) * (min_interval / scale_factor)
                            column_y = obs_y_min - abs(dy) * (min_interval / scale_factor)
                            print(f"Column adjusted to before obstruction: ({column_x}, {column_y})")
                        else:
                            # After the obstruction
                            column_x = obs_x_max + abs(dx) * (min_interval / scale_factor)
                            column_y = obs_y_max + abs(dy) * (min_interval / scale_factor)
                            print(f"Column adjusted to after obstruction: ({column_x}, {column_y})")

                        break  # Exit obstruction loop once handled

                # Add the column if valid
                columns.append((column_x, column_y))

    return columns
def plot_columns_on_annotated_frame(annotated_frame, columns):
    """
    Plot column coordinates onto the annotated frame and return the updated frame.

    Args:
        annotated_frame (numpy.ndarray): Annotated frame generated from `results[0].plot()`.
        columns (list): List of column coordinates (x, y) to plot.

    Returns:
        numpy.ndarray: The updated annotated frame with columns plotted.
    """
    # Ensure the frame is valid
    if annotated_frame is None:
        raise ValueError("Error: Annotated frame is not valid.")

    # Plot each column as a circle
    for column in columns:
        x, y = map(int, column)  # Convert coordinates to integers
        cv2.circle(annotated_frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red circle

    # Return the updated frame
    return annotated_frame

def convert_dxf_to_image(dxf_uploaded_file):
    """Convert a DXF file to an image using convertapi.com."""
    api_key = "secret_ErbOTw4sYs7t3O5Y"  # Replace with your ConvertAPI key

    # Upload the DXF file to convertapi.com
    payload = {'StoreFile': 'true'}
    response = requests.post(
    f"https://v2.convertapi.com/convert/dxf/to/png?Secret={api_key}",data=payload,
    files={"file": (dxf_uploaded_file.name, dxf_uploaded_file)}
    )
    # st.write(response.status_code)
    if response.status_code == 200:
        # Get the converted file URL
        result = response.json()
        # st.write(result)
        converted_file_url = result['Files'][0]['Url']

        # Download the converted PNG file
        converted_file_response = requests.get(converted_file_url)
        if converted_file_response.status_code == 200:
            # Load the downloaded PNG as a PIL image
            return Image.open(io.BytesIO(converted_file_response.content))
        else:
            raise Exception("Failed to download converted file from ConvertAPI.")
    else:
        raise Exception(f"ConvertAPI request failed with status code {response.status_code}: {response.text}")
def convert_pdf_to_image(pdf_uploaded_file):
    """
    Convert a PDF file to an image using convertapi.com.

    Args:
        pdf_uploaded_file: A file-like object representing the uploaded PDF file.

    Returns:
        PIL.Image object of the first page of the converted PDF.
    """
    api_key = "secret_ErbOTw4sYs7t3O5Y"  # Replace with your ConvertAPI key

    # Upload the PDF file to convertapi.com
    payload = {'StoreFile': 'true'}
    response = requests.post(
        f"https://v2.convertapi.com/convert/pdf/to/png?Secret={api_key}",
        data=payload,
        files={"file": (pdf_uploaded_file.name, pdf_uploaded_file)}
    )

    if response.status_code == 200:
        # Parse the result
        result = response.json()

        # Get the URL of the first converted PNG file
        converted_file_url = result['Files'][0]['Url']

        # Download the converted PNG file
        converted_file_response = requests.get(converted_file_url)
        if converted_file_response.status_code == 200:
            # Load the downloaded PNG as a PIL image
            return Image.open(io.BytesIO(converted_file_response.content))
        else:
            raise Exception("Failed to download converted file from ConvertAPI.")
    else:
        raise Exception(f"ConvertAPI request failed with status code {response.status_code}: {response.text}")
def segment_image(image_path):
    """
    Segments an image into four equal parts and returns them as Image objects.

    Args:
        image_path (str): Path to the input image.

    Returns:
        List of Image objects for the segmented images.
    """
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    #finding longer dimension
    longer_dim = width
    if width < height:
        longer_dim = height
    if ((height > 2000) and (width > 1000)) or ((width > 2000) and (height > 1000)):
        # Calculate the dimensions for each segment
        mid_width = width // 2
        mid_height = height // 2

        # Define bounding boxes for the four segments
        boxes = [
            (0, 0, mid_width, mid_height),  # Top-left
            (mid_width, 0, width, mid_height),  # Top-right
            (0, mid_height, mid_width, height),  # Bottom-left
            (mid_width, mid_height, width, height),  # Bottom-right
        ]
    elif longer_dim > 1000:
        if width >= height:
            # Split vertically if the width is the longer dimension
            mid_point = width // 2
            boxes = [
                (0, 0, mid_point, height),  # Left half
                (mid_point, 0, width, height),  # Right half
            ]
        else:
            # Split horizontally if the height is the longer dimension
            mid_point = height // 2
            boxes = [
                (0, 0, width, mid_point),  # Top half
                (0, mid_point, width, height),  # Bottom half
            ]
    else:
        boxes = [
            (0, 0, width, height)
        ]

    # Crop each segment and store them
    segmented_images = [image.crop(box) for box in boxes]

    return segmented_images

# Streamlit UI
st.title("Image & CAD File Upload with YOLO Annotation")
st.write("Upload an image or DWG/DXF file, and the YOLO model will process it.")

# File uploader
uploaded_file = st.file_uploader("Choose a file (Image, PDF or DWG/DXF)...", type=["jpg", "png", "jpeg","pdf","dwg", "dxf"])

if uploaded_file is not None:
    # Check file type
    file_type = uploaded_file.name.split('.')[-1].lower()
     # Ensure the save directory exists
    save_dir = "uploaded_files"
    os.makedirs(save_dir, exist_ok=True)

    # Define save path
    save_path = os.path.join(save_dir, uploaded_file.name)

    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_type in ["jpg", "png", "jpeg"]:
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    elif file_type in ["pdf", "dxf"]:
        # Convert DWG/DXF to image using ConvertAPI
        # Extract the base name without extension
        uploaded_file_name = uploaded_file.name
        base_name = os.path.splitext(uploaded_file_name)[0]

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Define save path with a PNG extension
        save_path = os.path.join(save_dir, f"{base_name}.png")
        try: 
            image = convert_pdf_to_image(uploaded_file)
            image.save(save_path, format="PNG")
            st.image(image, caption="Converted PDF to Image", use_container_width=True)

            # Convert PIL Image to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error converting DXF/DWG to image: {e}")
            img_bgr = None
    elif file_type in ["dwg", "dxf"]:
        # Convert DWG/DXF to image using ConvertAPI
        try: 
            image = convert_dxf_to_image(uploaded_file)
            max_dimension = get_max_dimension_from_dxf(save_path)
            st.image(image, caption="Converted DXF/DWG to Image", use_container_width=True)

            # Convert PIL Image to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error converting DXF/DWG to image: {e}")
            img_bgr = None
    else:
        st.error("Unsupported file type!")
        img_bgr = None

    # Run YOLO model if an image is available
    if img_bgr is not None:
        segmented_images = segment_image(save_path)
        # results = model(img_bgr)
        # dimension_class_id = 0  # Replace with your dimension class ID
        # longest_dimension_bbox = calculate_dimension_bounding_box_length(results, dimension_class_id)
        # wall_class_id = 2  # Replace with your wall class ID
        # obstruction_class_ids = [1, 3]  # Replace with your obstruction class IDs
        # wall_intersections, wall_lengths, wall_coordinates, obstructions = detect_wall_intersections_and_obstructions(
        # results, wall_class_id, obstruction_class_ids)
        # longest_length = longest_dimension_bbox['length']
        # if max_dimension > 0:
        #     scale_factor = longest_length/max_dimension
        # else:
        #     scale_factor = 50
            #scale_factor = longest_length/41795
        # Now you can pass these results to the column placement function
        # columns = place_columns_with_conditions(wall_intersections,wall_lengths,wall_coordinates, scale_factor, obstructions)
        # Annotate the image
        st.write("### Segmented Images")
        combined_wall_results = []
        combined_dw_results = []
        for i, img in enumerate(segmented_images, 1):
            results_wall = model_wall(img)
            annotated_frame = results_wall[0].plot()
            combined_wall_results.append(results_wall)
            results_dw = model_door_win(img)
            results_annotate = model_door_win(annotated_frame)
            combined_dw_results.append(results_dw)
            annotated_frame = results_annotate[0].plot()
            st.image(annotated_frame, caption=f"Annotated Image {i}", use_container_width=True)
        # annotated_frame = results[0].plot()  # YOLO annotates the frame
        # annotated_frame_2 = plot_columns_on_annotated_frame(img_bgr, columns)
        # Display the annotated image
        # st.image(annotated_frame, caption="Annotated Image", use_container_width=True)
        # st.image(annotated_frame_2, caption="Image with columns", use_container_width=True)
