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
model_door_win = YOLO("doors and windows original.pt")
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
import math

def calculate_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two coordinates in 2D space.

    Args:
        coord1 (tuple): The first coordinate as (x1, y1).
        coord2 (tuple): The second coordinate as (x2, y2).

    Returns:
        float: The distance between the two coordinates.
    """
    x1, y1 = coord1
    x2, y2 = coord2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def find_intersection_center(box1, box2):
    """
    Find the center point of the intersection of two bounding boxes only if they are aligned
    in different directions (one in x-direction, the other in y-direction).

    Args:
        box1 (tuple): First bounding box [x_min, y_min, x_max, y_max].
        box2 (tuple): Second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        tuple: Center point of the intersection if conditions are met, otherwise None.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the coordinates of the intersection box
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Check if there is a valid intersection
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        # Determine alignment of each box
        width1, height1 = x_max1 - x_min1, y_max1 - y_min1
        width2, height2 = x_max2 - x_min2, y_max2 - y_min2

        # Check if one box is x-aligned and the other is y-aligned
        if (width1 > height1 and height2 > width2) or (height1 > width1 and width2 > height2):
            # Calculate the center of the intersection box
            center_x = (inter_x_min + inter_x_max) / 2
            center_y = (inter_y_min + inter_y_max) / 2
            return (center_x, center_y)

    # No intersection or boxes are in the same direction
    return None

def find_intersection_curve(box1, box2):
    """
    Find the center point of the intersection of two bounding boxes only if they are aligned
    in different directions (one in x-direction, the other in y-direction).

    Args:
        box1 (tuple): First bounding box [x_min, y_min, x_max, y_max].
        box2 (tuple): Second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        tuple: Center point of the intersection if conditions are met, otherwise None.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the coordinates of the intersection box
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Check if there is a valid intersection
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        # Determine alignment of each box
        width1, height1 = x_max1 - x_min1, y_max1 - y_min1
        width2, height2 = x_max2 - x_min2, y_max2 - y_min2

        # Calculate the center of the intersection box
        center_x = (inter_x_min + inter_x_max) / 2
        center_y = (inter_y_min + inter_y_max) / 2
        return (center_x, center_y)

    # No intersection or boxes are in the same direction
    return None

def find_intersection_slope(box1, box2,sloped_walls_pos,sloped_walls_neg):
    """
    Find the center point of the intersection of two bounding boxes only if they are aligned
    in different directions (one in x-direction, the other in y-direction).

    Args:
        box1 (tuple): First bounding box [x_min, y_min, x_max, y_max].
        box2 (tuple): Second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        tuple: Center point of the intersection if conditions are met, otherwise None.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the coordinates of the intersection box
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Check if there is a valid intersection
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        # Determine alignment of each box
        width1, height1 = x_max1 - x_min1, y_max1 - y_min1
        width2, height2 = x_max2 - x_min2, y_max2 - y_min2

        # Calculate the center of the intersection box
        center_x = (inter_x_min + inter_x_max) / 2
        center_y = (inter_y_min + inter_y_max) / 2
        intersection_center = (center_x, center_y)
        if box1 in sloped_walls_neg and box2 in sloped_walls_pos:
            top_left = (x_min1, y_max1)
            bottom_right = (x_max1, y_min1)
            d1 = calculate_distance(intersection_center, top_left)
            d2 = calculate_distance(intersection_center, bottom_right)
            if d1 < d2:
                return top_left
            else:
                return bottom_right
        elif box1 in sloped_walls_pos and box2 in sloped_walls_neg:
            top_right = (x_max1, y_max1)
            bottom_left = (x_min1, y_min1)
            d1 = calculate_distance(intersection_center, top_right)
            d2 = calculate_distance(intersection_center, bottom_left)
            if d1 < d2:
                return top_right
            else:
                return bottom_left
        else:
            return None
    return None
def find_non_intersecting_end(box1, box2):
    """
    Find the coordinates of the non-intersecting end of the shorter bounding box.

    Args:
        box1 (tuple): First bounding box [x_min, y_min, x_max, y_max].
        box2 (tuple): Second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        tuple: Coordinates of the non-intersecting end of the shorter bounding box, or None if no intersection.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the coordinates of the intersection box
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Check if there is a valid intersection
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        # Calculate dimensions of the boxes
        width1, height1 = x_max1 - x_min1, y_max1 - y_min1
        width2, height2 = x_max2 - x_min2, y_max2 - y_min2

        # Determine the shorter box
        if width1 * height1 < width2 * height2:  # Box 1 is shorter
            shorter_box = box1
            other_box = box2
        else:  # Box 2 is shorter
            shorter_box = box2
            other_box = box1

        # Determine the non-intersecting end of the shorter box
        if shorter_box == box1:
            if x_min1 < inter_x_min:  # Left side of box1 is non-intersecting
                return (x_min1, (y_min1 + y_max1) / 2)
            elif x_max1 > inter_x_max:  # Right side of box1 is non-intersecting
                return (x_max1, (y_min1 + y_max1) / 2)
            elif y_min1 < inter_y_min:  # Top side of box1 is non-intersecting
                return ((x_min1 + x_max1) / 2, y_min1)
            elif y_max1 > inter_y_max:  # Bottom side of box1 is non-intersecting
                return ((x_min1 + x_max1) / 2, y_max1)
        else:
            if x_min2 < inter_x_min:  # Left side of box2 is non-intersecting
                return (x_min2, (y_min2 + y_max2) / 2)
            elif x_max2 > inter_x_max:  # Right side of box2 is non-intersecting
                return (x_max2, (y_min2 + y_max2) / 2)
            elif y_min2 < inter_y_min:  # Top side of box2 is non-intersecting
                return ((x_min2 + x_max2) / 2, y_min2)
            elif y_max2 > inter_y_max:  # Bottom side of box2 is non-intersecting
                return ((x_min2 + x_max2) / 2, y_max2)

    # No intersection
    return None


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
def identify_wall_types(results, image):
    corner_curves = []
    sloped_walls_pos = []
    sloped_walls_neg = []
    straight_walls = []
    image = np.array(image)
    for result in results:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_wall = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped_wall, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                wall_type = "None"
                if area > 50:
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        (x, y), (major_axis, minor_axis), angle = ellipse
                        if major_axis == 0 or minor_axis == 0:
                            continue
                        axis_ratio = minor_axis / major_axis if major_axis > minor_axis else major_axis / minor_axis
                        if 0.7 < axis_ratio < 1.0:
                            wall_type = "Curved Wall"
                            corner_curves.append((x1, y1, x2, y2))
                    if wall_type == "None":
                        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        slope = vy / vx if vx != 0 else float('inf')
                        if abs(slope) > 0.1 and abs(slope) < 10:
                            wall_type = "Sloped Wall"
                            if slope > 0:
                                sloped_walls_pos.append((x1, y1, x2, y2))
                            else:
                                sloped_walls_neg.append((x1, y1, x2, y2))
                        else:
                            wall_type = "Straight Wall"
                            # straight_walls.append((x1, y1, x2, y2))
    return corner_curves, sloped_walls_pos, sloped_walls_neg, straight_walls

def detect_wall_intersections(results,image):
    """
    Detect intersections between walls and extract obstructions (e.g., doors, windows) using YOLO predictions.

    Args:
        results: Inference results from the YOLO model.

    Returns:
        list: List of center points of intersections between walls.
    """
    # Initialize wall detections and obstruction detections
    wall_class_id = 0
    wall_detections = []
    corner_curves, sloped_walls_pos, sloped_walls_neg, straight_walls = identify_wall_types(results, image)
    # Iterate over detected boxes
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Coordinates (x_min, y_min, x_max, y_max)
        conf = box.conf[0].cpu().numpy()  # Confidence score
        class_id = int(box.cls[0].cpu().numpy())  # Class ID

        # Filter detections for walls
        if class_id == wall_class_id:
            x_min, y_min, x_max, y_max = map(int, xyxy) 
            wall_detections.append((x_min, y_min, x_max, y_max))

    # Initialize output
    wall_intersections = []

    num_boxes = len(wall_detections)

    for i in range(num_boxes):
        box1 = wall_detections[i]
        for j in range(i + 1, num_boxes):
            box2 = wall_detections[j]
            # if (box1 in corner_curves) or (box2 in corner_curves):
            #     intersection_center = find_intersection_curve(box1, box2)
            if ((box1 in sloped_walls_pos) or (box1 in sloped_walls_neg)) and ((box2 in sloped_walls_pos) or (box2 in sloped_walls_neg)):
                intersection_center = find_intersection_slope(box1,box2,sloped_walls_pos,sloped_walls_neg)
            elif (box1 in sloped_walls_pos or box1 in sloped_walls_neg) or (box2 in sloped_walls_pos or box2 in sloped_walls_neg):
                intersection_center = find_intersection_curve(box1,box2)
            elif (box1 in corner_curves) or (box2 in corner_curves):
                intersection_center = find_intersection_curve(box1, box2)
            else:
                intersection_center = find_intersection_center(box1, box2)
            if intersection_center:
                wall_intersections.append(intersection_center)

    # Return the list of intersection centers
    return wall_intersections

def detect_window_intersections(results):
    """
    Detect intersections between walls and extract obstructions (e.g., doors, windows) using YOLO predictions.

    Args:
        results: Inference results from the YOLO model.

    Returns:
        list: List of center points of intersections between walls.
    """
    # Initialize wall detections and obstruction detections
    window_class_id = 1
    window_detections = []

    # Iterate over detected boxes
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Coordinates (x_min, y_min, x_max, y_max)
        conf = box.conf[0].cpu().numpy()  # Confidence score
        class_id = int(box.cls[0].cpu().numpy())  # Class ID

        # Filter detections for walls
        if class_id == window_class_id:
            x_min, y_min, x_max, y_max = xyxy
            window_detections.append((x_min, y_min, x_max, y_max))

    # Initialize output
    window_intersections = []

    num_boxes = len(window_detections)

    for i in range(num_boxes):
        box1 = window_detections[i]

        for j in range(i + 1, num_boxes):
            box2 = window_detections[j]
            intersection_center = find_intersection_center(box1, box2)
            if intersection_center:
                non_intersecting_end = find_non_intersecting_end(box1, box2)
                window_intersections.append(non_intersecting_end)

    # Return the list of intersection centers
    return window_intersections


def place_columns_with_conditions(intersections, win_intersections):
    """
    Place columns at intersections and along walls based on the original wall length after scaling,
    ensuring the distance between columns is within specified bounds and avoiding obstructions.

    Args:
        intersections (list): List of intersection center points (x, y).

    Returns:
        list: List of column coordinates (x, y) to place on the image.
    """
    columns = []

    # Add columns at wall intersections
    for intersection in intersections:
        columns.append(intersection)
    for intersection in win_intersections:
        columns.append(intersection)

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
        cv2.circle(annotated_frame, (x, y), radius=15, color=(255, 0, 0), thickness=-1) 
    # Return the updated frame
    return annotated_frame

def plot_window_intersections(annotated_frame, intersections):
    """
    Plot intersections coordinates onto the annotated frame and return the updated frame.

    Args:
        annotated_frame (numpy.ndarray): Annotated frame generated from `results[0].plot()`.
        intersections (list): List of intersections coordinates (x, y) to plot.

    Returns:
        numpy.ndarray: The updated annotated frame with intersections plotted.
    """
    # Ensure the frame is valid
    if annotated_frame is None:
        raise ValueError("Error: Annotated frame is not valid.")

    # Plot each intersection as a circle
    for intersection in intersections:
        x, y = map(int, intersection)  # Convert coordinates to integers
        cv2.circle(annotated_frame, (x, y), radius=15, color=(0, 255, 0), thickness=-1) 
    # Return the updated frame
    return annotated_frame


def convert_dxf_to_image(dxf_uploaded_file):
    """Convert a DXF file to an image using convertapi.com."""
    api_key = "secret_178fvDmZ6YFbSaKl"  # Replace with your ConvertAPI key

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
    api_key = "secret_178fvDmZ6YFbSaKl"  # Replace with your ConvertAPI key

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
def merge_images(images, original_width, original_height):
    """
    Merge segmented images into a single image and return the offsets of each image in the merged image.
    """
    merged_image = Image.new("RGB", (original_width, original_height))
    offsets = []  # Store the offsets for each image in the merged image

    current_x, current_y = 0, 0
    for img in images:
        merged_image.paste(img, (current_x, current_y))
        offsets.append((current_x, current_y))  # Track the offset of each image
        current_x += img.width  # Update current_x for next image (adjust logic as per layout)

        # If the next image doesn't fit horizontally, move to the next row
        if current_x >= original_width:
            current_x = 0
            current_y += img.height

    return merged_image, offsets
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
# Streamlit app title
st.title("YOLOv8 Weight Selector")

# Dropdown for selecting wall type
wall_type = st.selectbox(
    "Select Wall Type:",
    ("Hashed Walls", "Plain Walls")
)

weights = "hashed_wall.pt" if wall_type == "Hashed Walls" else "wall with corner curve.pt"
model_wall = YOLO(weights)

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
        original_image = Image.open(save_path)
        original_width, original_height = original_image.size
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
        class_labels = {0: "Door", 1: "Window"}  # Replace class IDs with labels
        class_labels_wall = {0: "Wall"}
        class_colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # Assign unique colors for each class
        class_colors_wall = {0: (255,0,0)}
        annotated_images = []
        original_img, offsets = merge_images(segmented_images, original_width, original_height)
        all_column_positions = []
        for i, img in enumerate(segmented_images, 1):
            # Annotate image for wall results
            results_wall = model_wall(img)
            combined_wall_results.append(results_wall)
            intersections = detect_wall_intersections(results_wall, img)

            # Create a copy of the original image for wall annotation
            annotated_frame_wall = np.array(img)

            for box, conf, cls in zip(results_wall[0].boxes.xyxy, results_wall[0].boxes.conf, results_wall[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box)  # Extract box coordinates
                label = f"{class_labels_wall[int(cls)]} {conf:.2f}"  # Label for wall class
                color = class_colors_wall[int(cls)]  # Color for wall class

                # Draw rectangle for the box
                cv2.rectangle(annotated_frame_wall, (x1, y1), (x2, y2), color, 2)

            # Annotate image for door/window results on the same frame
            results_dw = model_door_win(img)
            combined_dw_results.append(results_dw)
            win_intersections = detect_window_intersections(results_dw)
            columns = place_columns_with_conditions(intersections, win_intersections)
            for col in columns:
                col_x, col_y = col  # Column position in segmented image
                offset_x, offset_y = offsets[i - 1]  # Get the offset for this segmented image
                absolute_x = col_x + offset_x
                absolute_y = col_y + offset_y
                all_column_positions.append((absolute_x, absolute_y)) 
            for box, conf, cls in zip(results_dw[0].boxes.xyxy, results_dw[0].boxes.conf, results_dw[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box)  # Extract box coordinates
                label = f"{class_labels[int(cls)]} {conf:.2f}"  # Label for door/window class
                color = class_colors[int(cls)]  # Color for door/window class

                # Draw rectangle for the box
                cv2.rectangle(annotated_frame_wall, (x1, y1), (x2, y2), color, 2)

                # Draw the label
                # (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # cv2.rectangle(annotated_frame_wall, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)  # Label background
                # cv2.putText(annotated_frame_wall, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Display combined annotation for wall and door/window
            st.image(annotated_frame_wall, caption=f"Annotated Image {i}", use_container_width=True)
            annotated_frame_2 = plot_columns_on_annotated_frame(annotated_frame_wall, columns)
            # st.image(annotated_frame_2, caption="Image with wall intersections", use_container_width=True)
            annotated_frame_3 = plot_window_intersections(annotated_frame_wall, win_intersections)
            annotated_image_pil = Image.fromarray(annotated_frame_wall)
            annotated_images.append(annotated_image_pil)
            # st.image(annotated_frame_3, caption="Image with window intersections", use_container_width=True)
        merged_image,off = merge_images(annotated_images, original_width, original_height)
        st.image(merged_image, caption="Combined Images", use_container_width=True)
        original_image_np = np.array(original_image)
        original_img_col = plot_columns_on_annotated_frame(original_image_np, all_column_positions)
        st.image(original_img_col, caption="Combined Images", use_container_width=True)
        # annotated_frame = results[0].plot()  # YOLO annotates the frame
        # annotated_frame_2 = plot_columns_on_annotated_frame(img_bgr, columns)
        # Display the annotated image
        # st.image(annotated_frame, caption="Annotated Image", use_container_width=True)
