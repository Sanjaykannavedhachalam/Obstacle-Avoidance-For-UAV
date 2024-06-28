import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import random
import threading

# Initialize Tkinter window
root = tk.Tk()
root.title("Obstacles Detection")

# Create a label to hold the video frames
label = tk.Label(root)
label.pack()

# Load the YOLOv8 model with instance segmentation
model = YOLO('yolov8n-seg.pt')  # Using the pre-trained YOLOv8 Nano model with segmentation

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load COCO dataset class names (80 classes)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Generate random colors for different classes
def get_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return colors

# Get colors for the COCO dataset classes
colors = get_colors(len(class_names))

# Function to apply semi-transparent shading to the segmentation masks
def apply_segmentation_shading(frame, mask, color):
    mask_img = np.stack([mask] * 3, axis=-1) * color
    mask_img = mask_img.astype(np.uint8)
    return cv2.addWeighted(frame, 1.0, mask_img, 0.5, 0)

# Global variable to store the latest processed frame
latest_frame = None
lock = threading.Lock()

# Function to update the frame in a separate thread
def update_frame():
    global latest_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        with lock:
            latest_frame = frame

# Function to process the frame and update the UI
def process_frame():
    global latest_frame

    with lock:
        if latest_frame is not None:
            frame = latest_frame.copy()
            latest_frame = None
        else:
            # Schedule the next frame update
            label.after(10, process_frame)
            return

    # Perform detection and segmentation
    results = model(frame)

    # If results is a list, get the first element (assuming it contains the predictions)
    if isinstance(results, list):
        results = results[0]

    # Get frame dimensions
    height, width, _ = frame.shape
    frame_center_x, frame_center_y = width // 2, height // 2

    # Draw lines to divide the frame into regions
    line_color = (255, 0, 0)
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, height), line_color, 2)
    cv2.line(frame, (0, frame_center_y), (width, frame_center_y), line_color, 2)

    # Initialize direction to None
    direction = None

    # Check if results contain boxes and masks
    if hasattr(results, 'boxes') and hasattr(results, 'masks') and results.boxes is not None and results.masks is not None:
        boxes = results.boxes
        masks = results.masks

        frame_area = height * width
        obstacle_center = False
        obstacle_large = False

        # Counters for regions
        top_left, top_right, bottom_left, bottom_right = 0, 0, 0, 0

        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            class_name = class_names[class_id]
            color = colors[class_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label for obstacle class name, width, and height
            box_width, box_height = x2 - x1, y2 - y1
            label_text = f"{class_name}, W: {box_width}, H: {box_height}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Apply segmentation shading
            mask = mask.data.cpu().numpy()[0]
            frame = apply_segmentation_shading(frame, mask, color)

            # Calculate the area of the bounding box
            box_area = box_width * box_height

            # Determine the region of the frame where the obstacle is located
            if x1 < frame_center_x and y1 < frame_center_y:
                top_left += 1
            if x1 >= frame_center_x and y1 < frame_center_y:
                top_right += 1
            if x1 < frame_center_x and y1 >= frame_center_y:
                bottom_left += 1
            if x1 >= frame_center_x and y1 >= frame_center_y:
                bottom_right += 1

            # Check if the obstacle is covering more than 50% of the frame
            if box_area > 0.5 * frame_area:
                obstacle_large = True

            # Check if the obstacle is at the center of the frame
            if x1 < frame_center_x < x2 and y1 < frame_center_y < y2:
                obstacle_center = True

        # Check for stop conditions
        if obstacle_large or obstacle_center:
            direction = "stop"
            cv2.putText(frame, "STOP", (frame_center_x - 50, frame_center_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # Determine the direction to move based on the region with obstacles
            if top_left:
                direction = "go bottom right"
                cv2.line(frame, (frame_center_x, frame_center_y), (width, height), (0, 255, 0), 3)
            elif top_right:
                direction = "go bottom left"
                cv2.line(frame, (frame_center_x, frame_center_y), (0, height), (0, 255, 0), 3)
            elif bottom_left:
                direction = "go top right"
                cv2.line(frame, (frame_center_x, frame_center_y), (width, 0), (0, 255, 0), 3)
            elif bottom_right:
                direction = "go top left"
                cv2.line(frame, (frame_center_x, frame_center_y), (0, 0), (0, 255, 0), 3)

        # Print direction if determined
        if direction:
            print(direction)

    # If no detections, add a "No obstacles detected" message
    else:
        cv2.putText(frame, "No obstacles detected or camera blocked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the frame to an image format suitable for Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    label.config(image=imgtk)
    label.imgtk = imgtk

    # Schedule the next frame update
    label.after(10, process_frame)

# Start the frame update loop in a separate thread
threading.Thread(target=update_frame, daemon=True).start()

# Start the frame processing loop
process_frame()

# Function to handle the window close event
def on_closing():
    cap.release()
    root.destroy()

# Bind the window close event to the handler
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the Tkinter event loop
root.mainloop()
