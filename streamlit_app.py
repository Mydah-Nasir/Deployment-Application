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
import pytesseract
import re
import statistics

# For Windows, set the Tesseract executable path manually
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Load the YOLO model
model = YOLO("best (6).pt")  # Replace with your custom-trained YOLO model
model_door_win = YOLO("doors and windows original.pt")
model_wall = YOLO("latest walls.pt")

# Conversion factors to millimeters based on DXF units

def get_conversion_unit_from_dxf(uploaded_file):
    """
    Extract the maximum dimension from a DXF file uploaded as a file-like object and convert it to millimeters.

    Args:
        uploaded_file (file-like object): Uploaded DXF file.

    Returns:
        float: The maximum dimension in millimeters.
    """
    # Conversion factors for DXF units to millimeters
    CONVERSION_FACTORS = {
    0: 1.0,          # Unitless
    1: 25.4,         # Inches (units.IN) to mm
    2: 304.8,        # Feet (units.FT) to mm
    3: 1609344.0,    # Miles (units.MI) to mm
    4: 1.0,          # Millimeters (units.MM)
    5: 10.0,         # Centimeters (units.CM) to mm
    6: 1000.0,       # Meters (units.M) to mm
    7: 1e6,          # Kilometers (units.KM) to mm
    8: 0.0254,       # Microinches to mm
    9: 0.0254,       # Mils to mm
    10: 914.4,       # Yards (units.YD) to mm
    11: 1e-7,        # Angstroms to mm
    12: 1e-6,        # Nanometers to mm
    13: 0.001,       # Microns to mm
    14: 100.0,       # Decimeters (units.DM) to mm
    15: 10000.0,     # Decameters to mm
    16: 100000.0,    # Hectometers to mm
    17: 1e9,         # Gigameters to mm
    18: 1.496e14,    # Astronomical units to mm
    19: 9.461e18,    # Light years to mm
    20: 3.086e19,    # Parsecs to mm
    21: 304.8006,    # US Survey Feet to mm
    22: 25.40005,    # US Survey Inch to mm
    23: 914.4018288, # US Survey Yard to mm
    24: 1609344.0,   # US Survey Mile to mm
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
    print(insunits)
    conversion_factor = CONVERSION_FACTORS.get(insunits, 1.0)  # Default to 1.0 if units are unknown
    return conversion_factor



def calculate_scale_factor_dxf(results, conversion_factor, image_path):
    """
    Calculate the length of each dimension bounding box and return the longest one.

    Args:
        results: Inference results from the YOLO model.
        dimension_class_id (int): Class ID for dimensions.

    Returns:
        dict: The longest dimension bounding box and its length.
    """
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    image = cv2.imread(image_path)

    # Filter bounding boxes with class_id 0 (Dimensions)
    dimension_boxes = [box for box, class_id in zip(detections, class_ids) if class_id == 0]

    # Resize scale factor (400% increase as per your requirement)
    resize_scale = 5

    # Extract text and integers from each bounding box
    extracted_texts = []
    extracted_integers = []
    scales = []  # Array to store pixel-to-meter scales

    for box in dimension_boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4])
        cropped = image[y_min:y_max, x_min:x_max]  # Crop the bounding box
        pixel_width = x_max - x_min
        pixel_height = y_max - y_min
        length = max(pixel_width, pixel_height)

        # Resize the cropped image
        h, w = cropped.shape[:2]
        resized_cropped = cv2.resize(cropped, (w * resize_scale, h * resize_scale), interpolation=cv2.INTER_LINEAR)

        # Convert resized image to grayscale (optional, for better OCR accuracy)
        resized_cropped_gray = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2GRAY)

        # Extract text with Tesseract
        text = pytesseract.image_to_string(resized_cropped_gray, config='--psm 6')  # Use Tesseract OCR
        extracted_texts.append(text.strip())  # Append cleaned text

        # Extract integers from the text using regex
        integers = re.findall(r'\d+', text)  # Find all sequences of digits
        integers = list(map(int, integers))  # Convert to integers
        extracted_integers.append(integers)
        print(integers)
        # Calculate pixel-to-meter scale if there is exactly one integer
        if len(integers) == 1:
            dimension = integers[0]*conversion_factor
            if dimension > 0:
                scale = length / dimension  # Pixels per meter
                scales.append(scale)
    if len(scales) > 0:
        median_scale = statistics.median(scales)
    else:
        median_scale = 0.05
    return median_scale
def find_scale_factor(results, image_path, unit):
    conversion_factors = {
        "mm": 1,          # 1 mm = 1 mm
        "cm": 10,         # 1 cm = 10 mm
        "m": 1000,        # 1 m = 1000 mm
        "feet": 304.8,    # 1 foot = 304.8 mm
        "inches": 25.4    # 1 inch = 25.4 mm
    }
    conversion_factor =  conversion_factors.get(unit.lower(), None)
    # Get bounding boxes
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    image = cv2.imread(image_path)

    # Filter bounding boxes with class_id 0 (Dimensions)
    dimension_boxes = [box for box, class_id in zip(detections, class_ids) if class_id == 0]

    # Resize scale factor (400% increase as per your requirement)
    resize_scale = 5

    # Extract text and integers from each bounding box
    extracted_texts = []
    extracted_integers = []
    scales = []  # Array to store pixel-to-meter scales

    for box in dimension_boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4])
        cropped = image[y_min:y_max, x_min:x_max]  # Crop the bounding box
        pixel_width = x_max - x_min
        pixel_height = y_max - y_min
        length = max(pixel_width, pixel_height)

        # Resize the cropped image
        h, w = cropped.shape[:2]
        resized_cropped = cv2.resize(cropped, (w * resize_scale, h * resize_scale), interpolation=cv2.INTER_LINEAR)

        # Convert resized image to grayscale (optional, for better OCR accuracy)
        resized_cropped_gray = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2GRAY)

        # Extract text with Tesseract
        text = pytesseract.image_to_string(resized_cropped_gray, config='--psm 6')  # Use Tesseract OCR
        extracted_texts.append(text.strip())  # Append cleaned text

        # Extract integers from the text using regex
        integers = re.findall(r'\d+', text)  # Find all sequences of digits
        integers = list(map(int, integers))  # Convert to integers
        extracted_integers.append(integers)
        print(integers)
        # Calculate pixel-to-meter scale if there is exactly one integer
        if len(integers) == 1:
            dimension = integers[0]*conversion_factor
            if dimension > 0:
                scale = length / dimension  # Pixels per meter
                scales.append(scale)
    if len(scales) > 0:
        median_scale = statistics.median(scales)
    else:
        median_scale = 0.05
    return median_scale
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
    if inter_x_min <= inter_x_max and inter_y_min <= inter_y_max:
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
    if inter_x_min <= inter_x_max and inter_y_min <= inter_y_max:
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
    if inter_x_min <= inter_x_max and inter_y_min <= inter_y_max:
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
            intersection_center = find_intersection_center(box1,box2)
            return intersection_center
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

def get_intersection_type(box1, box2):
    """
    Determine the type of intersection (corner or T-section) between two bounding boxes.

    Args:
        box1 (tuple): Bounding box 1 (x_min, y_min, x_max, y_max).
        box2 (tuple): Bounding box 2 (x_min, y_min, x_max, y_max).

    Returns:
        str: "corner", "T-section", or None if there is no intersection.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the intersection coordinates
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    # Check if there's a valid intersection
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        # Intersection dimensions
        inter_width = inter_x_max - inter_x_min
        inter_height = inter_y_max - inter_y_min

        # Check for corner intersection
        if (inter_width == x_max1 - x_min1 or inter_width == x_max2 - x_min2) and \
           (inter_height == y_max1 - y_min1 or inter_height == y_max2 - y_min2):
            return "corner"
        # Check for T-section intersection
        if (inter_width < x_max1 - x_min1 and inter_height == y_max1 - y_min1) or \
           (inter_height < y_max1 - y_min1 and inter_width == x_max1 - x_min1):
            return "T-section"
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
                if area > 300:
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
def calculate_door_window_bounding_boxes_for_segment(results_dw, offset):
    """
    Calculates the bounding boxes for doors and windows in the original image 
    for a specific segment based on detection results and its offset.

    Args:
        results_dw (object): Detection results from `model_door_win` for a single segmented image.
                             Contains bounding boxes, confidence scores, and classes.
        offset (tuple): Offset (offset_x, offset_y) for the segmented image.

    Returns:
        tuple: Two lists of adjusted bounding boxes for windows and doors in the original image coordinates.
    """
    windows_bounding_boxes = []
    doors_bounding_boxes = []
    offset_x, offset_y = offset  # Unpack the offset tuple

    # Extract bounding boxes from results_dw
    for box, conf, cls in zip(results_dw[0].boxes.xyxy, results_dw[0].boxes.conf, results_dw[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)  # Extract box coordinates as integers

        # Adjust bounding box coordinates using the offset
        absolute_x1 = x1 + offset_x
        absolute_y1 = y1 + offset_y
        absolute_x2 = x2 + offset_x
        absolute_y2 = y2 + offset_y

        # Determine the class (e.g., 0 for windows, 1 for doors)
        if cls == 1:  # Assuming class 0 is for windows
            windows_bounding_boxes.append((absolute_x1, absolute_y1, absolute_x2, absolute_y2))
        elif cls == 0:  # Assuming class 1 is for doors
            doors_bounding_boxes.append((absolute_x1, absolute_y1, absolute_x2, absolute_y2))

    return windows_bounding_boxes, doors_bounding_boxes


def calculate_wall_bounding_boxes_for_segment(results_dw, offset):
    """
    Calculates the bounding boxes for doors and windows in the original image 
    for a specific segment based on detection results and its offset.

    Args:
        results_dw (object): Detection results from `model_door_win` for a single segmented image.
                             Contains bounding boxes, confidence scores, and classes.
        offset (tuple): Offset (offset_x, offset_y) for the segmented image.

    Returns:
        list: List of adjusted bounding boxes in the original image coordinates.
    """
    bounding_boxes = []
    offset_x, offset_y = offset  # Unpack the offset tuple

    # Extract bounding boxes from results_dw
    for box, conf, cls in zip(results_dw[0].boxes.xyxy, results_dw[0].boxes.conf, results_dw[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)  # Extract box coordinates as integers

        # Adjust bounding box coordinates using the offset
        absolute_x1 = x1 + offset_x
        absolute_y1 = y1 + offset_y
        absolute_x2 = x2 + offset_x
        absolute_y2 = y2 + offset_y

        # Append adjusted bounding box to the result
        bounding_boxes.append((absolute_x1, absolute_y1, absolute_x2, absolute_y2))

    return bounding_boxes


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
def calculate_average_column_distance(all_column_positions, scale_factor):
    """
    Calculate the average distance between consecutive columns.

    Args:
        all_column_positions (list): A list of column positions [(x1, y1), (x2, y2), ...].

    Returns:
        float: The average distance between consecutive columns, or None if not enough positions.
    """
    if len(all_column_positions) < 2:
        # Not enough columns to calculate distances
        return None

    distances = []
    num_distances = len(all_column_positions) - 1
    for i in range(num_distances):
        x1, y1 = all_column_positions[i]
        x2, y2 = all_column_positions[i + 1]
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5/ scale_factor # Euclidean distance
        if 3000 < distance < 7000:
            distances.append(distance)
    if len(distances) > 0:
        average_dist = sum(distances)/len(distances)
    else:
        average_dist = 5000

    return average_dist
def remove_columns_with_scale(all_column_positions, scale_factor, threshold):
    """
    Place columns at intersections and along walls based on the original wall length after scaling,
    ensuring the distance between columns is within specified bounds and avoiding obstructions.

    Args:
        all_column_positions (list): List of intersection center points (x, y).
        scale_factor (float): Pixels per millimeter.

    Returns:
        list: List of column coordinates (x, y) to place on the image.
    """
    columns = []
    distances = []

    # Sort intersections by their x-coordinate, then by y-coordinate
    all_column_positions.sort(key=lambda point: (point[0], point[1]))
    # Iterate through intersections and filter based on distance
    for i in range(len(all_column_positions)):
        keep_column = True
        for j in range(i):
            x1, y1 = all_column_positions[i]
            x2, y2 = all_column_positions[j]

            # Calculate actual distance in millimeters
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / scale_factor

            if distance < threshold:
                # If the distance is less than 2000 mm, remove the column with the higher y-coordinate
                if y1 < y2:
                    keep_column = False
                else:
                    # Remove the previously added column
                    if all_column_positions[j] in columns:
                        columns.remove(all_column_positions[j])
        if keep_column:
            columns.append(all_column_positions[i])

    return columns
def adjust_columns_near_doors_windows(columns_with_dimensions, doors_bbox, windows_bbox):
    """
    Adjust the positions of columns based on the bounding boxes of doors and windows.

    If a column lies within a bounding box, it is moved to either the start (top/left)
    or the end (bottom/right) of the bounding box.

    Args:
        columns_with_dimensions (list): List of column data [(x, y, width, length)].
        doors_bbox (list): List of bounding boxes [(x1, y1, x2, y2)] for doors.
        windows_bbox (list): List of bounding boxes [(x1, y1, x2, y2)] for windows.

    Returns:
        list: Updated list of column data [(x, y, width, length)].
    """
    adjusted_columns = []

    for col_x, col_y, width, length in columns_with_dimensions:
        adjusted = False

        # Check for adjustment near doors
        for x1, y1, x2, y2 in doors_bbox:
            if x1 <= col_x <= x2 and y1 <= col_y <= y2:
                # Adjust based on the door bounding box
                if abs(x2 - x1) > abs(y2 - y1):
                    col_y = y2 + 15  # Adjust vertically
                else:
                    col_x = x2 + 15  # Adjust horizontally
                adjusted = True
                # break

        # Check for adjustment near windows if not already adjusted
        if not adjusted:
            for x1, y1, x2, y2 in windows_bbox:
                if x1 <= col_x <= x2 and y1 <= col_y <= y2:
                    # Adjust based on the window bounding box
                    if abs(x2 - x1) > abs(y2 - y1):
                        col_x = x2 + 15  # Adjust vertically (opposite direction)
                    else:
                        col_y = y2 + 15  # Adjust horizontally (opposite direction)
                    adjusted = True
                    # break

        # Append adjusted or original column data
        adjusted_columns.append((col_x, col_y, width, length))

    return adjusted_columns


def filter_columns_by_walls(walls_bbox, columns_with_dimensions):
    """
    Filters columns to ensure they only remain if they lie within any wall's bounding box.

    Args:
        walls_bbox (list): List of bounding boxes [(x1, y1, x2, y2)] representing walls.
        columns_with_dimensions (list): List of column data [(x, y, width, length)].

    Returns:
        list: Filtered list of column data [(x, y, width, length)].
    """
    filtered_columns = []
    margin = 20

    for col_x, col_y, width, length in columns_with_dimensions:
        in_wall = False  # Flag to check if column is within any wall

        # Check if the column lies within any wall's bounding box
        for x1, y1, x2, y2 in walls_bbox:
            if (x1 - margin) <= col_x <= (x2 + margin) and (y1 - margin) <= col_y <= (y2 + margin):
                in_wall = True
                break  # No need to check further; column lies in a wall

        # If the column is within a wall, keep it
        if in_wall:
            filtered_columns.append((col_x, col_y, width, length))

    return filtered_columns


def place_columns_with_scale(all_column_positions, scale_factor):
    """
    Place columns at intersections and along walls based on the original wall length after scaling,
    ensuring the distance between columns is within specified bounds and adding intermediate columns if necessary.

    Args:
        all_column_positions (list): List of intersection center points (x, y).
        scale_factor (float): Pixels per millimeter.

    Returns:
        list: List of tuples containing column coordinates (x, y), width, and length.
    """
    columns = []
    all_column_positions.sort(key=lambda point: (point[0], point[1]))  # Sort for easier processing
    average_dist = calculate_average_column_distance(all_column_positions, scale_factor)

    margin = 5  # Allowable margin for alignment

    # Iterate through each column position
    for i, (x1, y1) in enumerate(all_column_positions):
        keep_column = True
        nearest_vertical = None
        nearest_horizontal = None
        min_vert_dist = float("inf")
        min_horiz_dist = float("inf")
        width = 300*scale_factor

        # Find the nearest vertical and horizontal neighbors
        for j, (x2, y2) in enumerate(all_column_positions):
            if i == j:
                continue

            # Check for vertical alignment with margin
            if abs(x1 - x2) <= margin:  # Allow small difference for vertical alignment
                dist = abs(y2 - y1)
                if dist < min_vert_dist:
                    min_vert_dist = dist
                    nearest_vertical = (x2, y2)

            # Check for horizontal alignment with margin
            if abs(y1 - y2) <= margin:  # Allow small difference for horizontal alignment
                dist = abs(x2 - x1)
                if dist < min_horiz_dist:
                    min_horiz_dist = dist
                    nearest_horizontal = (x2, y2)

        # If vertical distance is greater than threshold and nearest vertical found, place intermediate columns
        if nearest_vertical:
            dist_mm = min_vert_dist / scale_factor
            if dist_mm > 7000:
                num_columns = int(dist_mm / average_dist)
                actual_spacing = dist_mm / num_columns
                for k in range(1, num_columns):
                    new_y = y1 + (nearest_vertical[1] - y1) * (k / num_columns)
                    new_column = (x1, new_y)
                    if new_column not in [col[:2] for col in columns]:
                        actual_length = actual_spacing / 20
                        length = actual_length*scale_factor
                        columns.append((x1, new_y, width, length))  # Add width and length

        # If horizontal distance is greater than threshold and nearest horizontal found, place intermediate columns
        if nearest_horizontal:
            dist_mm = min_horiz_dist / scale_factor
            if dist_mm > 7000:
                num_columns = int(dist_mm / average_dist)
                actual_spacing = dist_mm / num_columns
                for k in range(1, num_columns):
                    new_x = x1 + (nearest_horizontal[0] - x1) * (k / num_columns)
                    new_column = (new_x, y1)
                    if new_column not in [col[:2] for col in columns]:
                        actual_length = actual_spacing / 20
                        length = actual_length*scale_factor
                        columns.append((new_x, y1, length, width))  # Add width and length

        # Check if the column itself should be kept
        for existing_col in columns:
            if abs(existing_col[0] - x1) <= margin and abs(existing_col[1] - y1) <= margin:
                keep_column = False
                break

        if keep_column:
            columns.append((x1, y1, width, average_dist*scale_factor / 20))  # Add width and length for the original column

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

def plot_columns_with_dimensions_on_frame(annotated_frame, columns_with_dimensions):
    """
    Plot rectangles for columns with dimensions onto the annotated frame and return the updated frame.

    Args:
        annotated_frame (numpy.ndarray): Annotated frame generated from `results[0].plot()`.
        columns_with_dimensions (list): List of columns with dimensions [(x, y, width, length)].

    Returns:
        numpy.ndarray: The updated annotated frame with rectangles plotted.
    """
    # Ensure the frame is valid
    if annotated_frame is None:
        raise ValueError("Error: Annotated frame is not valid.")

    # Plot each column as a rectangle
    for column in columns_with_dimensions:
        x, y, width, length = column
        x1 = int(x - width / 2)  # Top-left x-coordinate
        y1 = int(y - length / 2)  # Top-left y-coordinate
        x2 = int(x + width / 2)  # Bottom-right x-coordinate
        y2 = int(y + length / 2)  # Bottom-right y-coordinate

        # Draw the rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)

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

def convert_dwg_to_image(dwg_uploaded_file):
    """
    Convert a DWG file to an image using ConvertAPI.

    Args:
        dwg_uploaded_file: A file-like object representing the uploaded DWG file.

    Returns:
        PIL.Image object of the first page of the converted DWG file.
    """
    api_key = "secret_178fvDmZ6YFbSaKl"  # Replace with your ConvertAPI key

    # Upload the DWG file to convertapi.com
    payload = {'StoreFile': 'true'}
    response = requests.post(
        f"https://v2.convertapi.com/convert/dwg/to/png?Secret={api_key}",
        data=payload,
        files={"file": (dwg_uploaded_file.name, dwg_uploaded_file)}
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
st.title("Select Wall Type")

# Dropdown for selecting wall type
wall_type = st.selectbox(
    "Select Wall Type:",
    ("Hashed Walls", "Plain Walls")
)

weights = "hashed_walls_improved.pt" if wall_type == "Hashed Walls" else "best (10).pt"
model_wall = YOLO(weights)

st.title("Image/PDF/CAD File Upload")
unit = st.selectbox(
    "Select a unit for image, pdf or dwg:",
    ["mm", "cm", "m", "feet", "inches"]  # Options in the dropdown
    )
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
    max_dimension = 0
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
    
    elif file_type in ["pdf"]:
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
        finally:
                # Remove the uploaded file
                file_path = os.path.join(save_dir, uploaded_file.name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Uploaded file '{uploaded_file.name}' has been removed.")
                else:
                    print(f"Uploaded file '{uploaded_file.name}' does not exist.")
    elif file_type in ["dxf"]:
        # Convert DWG/DXF to image using ConvertAPI
        try: 
            image = convert_dxf_to_image(uploaded_file)
            conversion_factor = get_conversion_unit_from_dxf(save_path)
            image.save(save_path, format="PNG")
            st.image(image, caption="Converted DXF/DWG to Image", use_container_width=True)

            # Convert PIL Image to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error converting DXF/DWG to image: {e}")
            img_bgr = None
    elif file_type in ["dwg"]:
        # Convert DWG/DXF to image using ConvertAPI
        # Extract the base name without extension
        uploaded_file_name = uploaded_file.name
        base_name = os.path.splitext(uploaded_file_name)[0]

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Define save path with a PNG extension
        save_path = os.path.join(save_dir, f"{base_name}.png")
        try: 
            image = convert_dwg_to_image(uploaded_file)
            image.save(save_path, format="PNG")
            st.image(image, caption="Converted PDF to Image", use_container_width=True)

            # Convert PIL Image to OpenCV format
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error converting DXF/DWG to image: {e}")
            img_bgr = None
        finally:
            # Remove the uploaded file
            file_path = os.path.join(save_dir, uploaded_file.name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Uploaded file '{uploaded_file.name}' has been removed.")
            else:
                print(f"Uploaded file '{uploaded_file.name}' does not exist.")
    else:
        st.error("Unsupported file type!")
        img_bgr = None

    # Run YOLO model if an image is available
    if img_bgr is not None:
        segmented_images = segment_image(save_path)
        original_image = Image.open(save_path)
        original_width, original_height = original_image.size
        img_dim = img_bgr.copy()
        results_dim = model(img_dim)
        print(file_type)
        if file_type in ["jpg", "png", "jpeg", "pdf"]:
            scale_factor = find_scale_factor(results_dim, save_path, unit)
        elif file_type in ["dxf"]:
            scale_factor = calculate_scale_factor_dxf(results_dim, conversion_factor, save_path)
        else:
            scale_factor = 0.05
        print(scale_factor)
        if scale_factor > 0.1:
            scale_factor = 0.05
        # results = model(img_bgr)
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
        st.write("### Annotated Images")
        combined_wall_results = []
        combined_dw_results = []
        class_labels = {0: "Door", 1: "Window"}  # Replace class IDs with labels
        class_labels_wall = {0: "Wall"}
        class_colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # Assign unique colors for each class
        class_colors_wall = {0: (255,0,0)}
        annotated_images = []
        original_img, offsets = merge_images(segmented_images, original_width, original_height)
        all_column_positions = []
        all_doors = []
        all_wins = []
        all_walls = []
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
            wins_bboxs, doors_bboxs = calculate_door_window_bounding_boxes_for_segment(results_dw, offsets[i -1])
            walls_bboxs = calculate_wall_bounding_boxes_for_segment(results_wall, offsets[i -1])
            all_doors.extend(doors_bboxs)
            all_wins.extend(wins_bboxs)
            all_walls.extend(walls_bboxs)
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
            # st.image(annotated_frame_wall, caption=f"Annotated Image {i}", use_container_width=True)
            annotated_frame_2 = plot_columns_on_annotated_frame(annotated_frame_wall, columns)
            # st.image(annotated_frame_2, caption="Image with wall intersections", use_container_width=True)
            annotated_frame_3 = plot_window_intersections(annotated_frame_wall, win_intersections)
            annotated_image_pil = Image.fromarray(annotated_frame_wall)
            annotated_images.append(annotated_image_pil)
            # st.image(annotated_frame_3, caption="Image with window intersections", use_container_width=True)
        merged_image,off = merge_images(annotated_images, original_width, original_height)
        st.image(merged_image, caption="Annotated Images", use_container_width=True)
        original_image_np = np.array(original_image)
        updated_cols = remove_columns_with_scale(all_column_positions, scale_factor, 50)
        columns_with_dimension = place_columns_with_scale(updated_cols, scale_factor) 
        new_columns_adj = adjust_columns_near_doors_windows(columns_with_dimension, all_doors, all_wins)
        columns_walls = filter_columns_by_walls(all_walls, new_columns_adj)
        new_columns = []
        for column in columns_walls:
            x, y = column[0], column[1]  # Extract x, y coordinates
            new_columns.append((x, y))
        final_cols = remove_columns_with_scale(new_columns, scale_factor, 2000)
        final_cols_set = set(final_cols)
        filtered_columns = [
        column for column in columns_walls if (column[0], column[1]) in final_cols_set
        ]
        #original_img_col = plot_columns_with_dimensions_on_frame(original_image_np, filtered_columns)
        original_img_col = plot_columns_on_annotated_frame(original_image_np, final_cols)
        st.write("### Image with columns")
        st.image(original_img_col, caption="Image with columns", use_container_width=True)
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"File at {save_path} has been removed.")
        else:
            print(f"File at {save_path} does not exist.")
        # annotated_frame = results[0].plot()  # YOLO annotates the frame
        # annotated_frame_2 = plot_columns_on_annotated_frame(img_bgr, columns)
        # Display the annotated image
        # st.image(annotated_frame, caption="Annotated Image", use_container_width=True)
