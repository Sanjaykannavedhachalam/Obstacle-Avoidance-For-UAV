import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import torch
import torchvision

# Initialize the tkinter window
root = tk.Tk()
root.title("Obstacles Detection")

label = tk.Label(root)
label.pack()

# Load the Faster R-CNN model pre-trained on COCO dataset
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define colors for stationary and moving objects
stationary_color = (0, 255, 0)  # Green
moving_color = (0, 0, 255)  # Red

previous_boxes = None

def calculate_intersection_area(box, quadrant):
    x1, y1, x2, y2 = box
    q_x1, q_y1, q_x2, q_y2 = quadrant

    inter_x1 = max(x1, q_x1)
    inter_y1 = max(y1, q_y1)
    inter_x2 = min(x2, q_x2)
    inter_y2 = min(y2, q_y2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    return 0

def update_frame():
    global previous_boxes

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # Convert the frame to RGB format and to a tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(frame_rgb / 255.).permute(2, 0, 1).float().unsqueeze(0)

    # Get predictions from the model
    with torch.no_grad():
        predictions = model(img)

    # Initialize cumulative obstruction values
    top_left_obstruction = 0
    top_right_obstruction = 0
    bottom_left_obstruction = 0
    bottom_right_obstruction = 0

    frame_height, frame_width, _ = frame.shape
    section_width = frame_width // 2
    section_height = frame_height // 2

    # Quadrants boundaries
    top_left_quad = [0, 0, section_width, section_height]
    top_right_quad = [section_width, 0, frame_width, section_height]
    bottom_left_quad = [0, section_height, section_width, frame_height]
    bottom_right_quad = [section_width, section_height, frame_width, frame_height]

    # Draw division lines for quadrants
    cv2.line(frame, (section_width, 0), (section_width, frame_height), (255, 255, 255), 1)
    cv2.line(frame, (0, section_height), (frame_width, section_height), (255, 255, 255), 1)

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    for i, box in enumerate(boxes):
        if scores[i] < 0.5:  # Threshold for confidence score
            continue

        x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
        obstacle_width = x2 - x1
        obstacle_height = y2 - y1
        box_area = obstacle_width * obstacle_height

        # Calculate the area of the bounding box overlapping with each quadrant
        top_left_obstruction += calculate_intersection_area(box, top_left_quad)
        top_right_obstruction += calculate_intersection_area(box, top_right_quad)
        bottom_left_obstruction += calculate_intersection_area(box, bottom_left_quad)
        bottom_right_obstruction += calculate_intersection_area(box, bottom_right_quad)

        # Determine if the object is moving
        moving_object = True
        if previous_boxes is not None and i < len(previous_boxes):
            prev_x1, prev_y1, prev_x2, prev_y2 = map(int, previous_boxes[i])
            # Calculate movement threshold based on bounding box size or distance
            if abs(x1 - prev_x1) < 10 and abs(y1 - prev_y1) < 10:
                moving_object = False

        # Draw the bounding box with color based on movement detection
        if moving_object:
            cv2.rectangle(frame, (x1, y1), (x2, y2), moving_color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), stationary_color, 2)

        # Display the coordinates, height, width, and box area
        label_text = f"({(x1 + x2) / 2},{(y1 + y2) / 2}) W:{obstacle_width}px H:{obstacle_height}px Area:{box_area}px²"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stationary_color, 2)

    previous_boxes = boxes

    # Calculate percentages for each quadrant
    total_top_left = section_width * section_height
    total_top_right = section_width * section_height
    total_bottom_left = section_width * section_height
    total_bottom_right = section_width * section_height

    top_left_percentage = min((top_left_obstruction / total_top_left) * 100, 100)
    top_right_percentage = min((top_right_obstruction / total_top_right) * 100, 100)
    bottom_left_percentage = min((bottom_left_obstruction / total_bottom_left) * 100, 100)
    bottom_right_percentage = min((bottom_right_obstruction / total_bottom_right) * 100, 100)

    # Display which quadrants have obstacles and their percentages
    if top_left_percentage > 0:
        cv2.putText(frame, f"Top Left Obstructed: {top_left_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Top Left Clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stationary_color, 2)
    if top_right_percentage > 0:
        cv2.putText(frame, f"Top Right Obstructed: {top_right_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Top Right Clear", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stationary_color, 2)
    if bottom_left_percentage > 0:
        cv2.putText(frame, f"Bottom Left Obstructed: {bottom_left_percentage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Bottom Left Clear", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stationary_color, 2)
    if bottom_right_percentage > 0:
        cv2.putText(frame, f"Bottom Right Obstructed: {bottom_right_percentage:.2f}%", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Bottom Right Clear", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stationary_color, 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
