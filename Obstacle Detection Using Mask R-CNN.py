# MASK R-CNN 

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

# Load the Mask R-CNN model pre-trained on COCO dataset
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
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

    # Draw division lines for quadrants
    cv2.line(frame, (section_width, 0), (section_width, frame_height), (255, 255, 255), 1)
    cv2.line(frame, (0, section_height), (frame_width, section_height), (255, 255, 255), 1)

    masks = predictions[0]['masks'].cpu().numpy()
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    for i, mask in enumerate(masks):
        if scores[i] < 0.5:  # Threshold for confidence score
            continue

        mask = mask[0] > 0.5  # Threshold mask values
        mask_area = np.sum(mask)  # Total number of non-zero pixels in the mask

        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, stationary_color, 3)

        # Determine the percentage of obstruction in each quadrant
        top_left_mask = mask[:section_height, :section_width]
        top_right_mask = mask[:section_height, section_width:]
        bottom_left_mask = mask[section_height:, :section_width]
        bottom_right_mask = mask[section_height:, section_width:]

        top_left_obstruction += np.sum(top_left_mask)
        top_right_obstruction += np.sum(top_right_mask)
        bottom_left_obstruction += np.sum(bottom_left_mask)
        bottom_right_obstruction += np.sum(bottom_right_mask)

        x1, y1, x2, y2 = map(int, boxes[i])  # Get bounding box coordinates
        obstacle_width = x2 - x1
        obstacle_height = y2 - y1

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

        # Display the coordinates, height, width, and mask area
        label_text = f"({(x1 + x2) / 2},{(y1 + y2) / 2}) W:{obstacle_width}px H:{obstacle_height}px Area:{mask_area}px²"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stationary_color, 2)

    previous_boxes = boxes

    # Calculate percentages for each quadrant
    total_top_left = section_width * section_height
    total_top_right = section_width * section_height
    total_bottom_left = section_width * section_height
    total_bottom_right = section_width * section_height

    top_left_percentage = (top_left_obstruction / total_top_left) * 100
    top_right_percentage = (top_right_obstruction / total_top_right) * 100
    bottom_left_percentage = (bottom_left_obstruction / total_bottom_left) * 100
    bottom_right_percentage = (bottom_right_obstruction / total_bottom_right) * 100

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
