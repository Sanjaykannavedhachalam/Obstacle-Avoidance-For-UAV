import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from torchvision import models, transforms

root = tk.Tk()
root.title("Obstacles Detection")

label = tk.Label(root)
label.pack()

# Load the pre-trained SSD model for detection
detection_model = models.detection.ssd300_vgg16(pretrained=True)
detection_model.eval()

# Load the pre-trained DeepLabV3 model for segmentation
segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
segmentation_model.eval()

# Define the transformation to apply to the frames for the models
transform = transforms.Compose([
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # Convert frame to RGB and apply transformations
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        detection_outputs = detection_model(input_tensor)
        segmentation_outputs = segmentation_model(input_tensor)['out']

    # Get model outputs
    boxes = detection_outputs[0]['boxes'].cpu().numpy()
    scores = detection_outputs[0]['scores'].cpu().numpy()

    # Process segmentation outputs
    segmentation_mask = torch.argmax(segmentation_outputs.squeeze(), dim=0).detach().cpu().numpy()
    segmentation_mask = (segmentation_mask == 15).astype(np.uint8)  # Assuming class 15 is the target class (e.g., person)

    frame_height, frame_width, _ = frame.shape
    section_width = frame_width // 2
    section_height = frame_height // 2

    # Draw the lines dividing the frame into four segments
    cv2.line(frame, (section_width, 0), (section_width, frame_height), (255, 0, 0), 2)
    cv2.line(frame, (0, section_height), (frame_width, section_height), (255, 0, 0), 2)

    # Initialize cumulative obstruction values
    top_left_obstruction = 0
    top_right_obstruction = 0
    bottom_left_obstruction = 0
    bottom_right_obstruction = 0

    total_top_left = section_width * section_height
    total_top_right = section_width * section_height
    total_bottom_left = section_width * section_height
    total_bottom_right = section_width * section_height

    for i in range(len(boxes)):
        if scores[i] < 0.5:
            continue

        box = boxes[i]
        x1, y1, x2, y2 = map(int, box)

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display the coordinates, height, and width
        obstacle_width = x2 - x1
        obstacle_height = y2 - y1
        label_text = f"({x1},{y1}) W:{obstacle_width}px H:{obstacle_height}px"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate the area of the obstacle
        obstacle_area = obstacle_width * obstacle_height

        # Determine which quadrants the obstacle overlaps and add the overlapping area to the respective quadrant obstruction
        for x in range(x1, x2):
            for y in range(y1, y2):
                if x < section_width and y < section_height:  # Top-left quadrant
                    top_left_obstruction += 1
                elif x >= section_width and y < section_height:  # Top-right quadrant
                    top_right_obstruction += 1
                elif x < section_width and y >= section_height:  # Bottom-left quadrant
                    bottom_left_obstruction += 1
                elif x >= section_width and y >= section_height:  # Bottom-right quadrant
                    bottom_right_obstruction += 1

    # Calculate percentages for each quadrant
    top_left_percentage = (top_left_obstruction / total_top_left) * 100
    top_right_percentage = (top_right_obstruction / total_top_right) * 100
    bottom_left_percentage = (bottom_left_obstruction / total_bottom_left) * 100
    bottom_right_percentage = (bottom_right_obstruction / total_bottom_right) * 100

    # Ensure percentages do not exceed 100%
    top_left_percentage = min(top_left_percentage, 100)
    top_right_percentage = min(top_right_percentage, 100)
    bottom_left_percentage = min(bottom_left_percentage, 100)
    bottom_right_percentage = min(bottom_right_percentage, 100)

    # Update obstacle status based on percentages
    top_left_obstacle = top_left_percentage > 0
    top_right_obstacle = top_right_percentage > 0
    bottom_left_obstacle = bottom_left_percentage > 0
    bottom_right_obstacle = bottom_right_percentage > 0

    # Display which quadrants have obstacles and their percentages
    if top_left_obstacle:
        cv2.putText(frame, f"Top Left Obstructed: {top_left_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Top Left Clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if top_right_obstacle:
        cv2.putText(frame, f"Top Right Obstructed: {top_right_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Top Right Clear", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if bottom_left_obstacle:
        cv2.putText(frame, f"Bottom Left Obstructed: {bottom_left_percentage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Bottom Left Clear", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if bottom_right_obstacle:
        cv2.putText(frame, f"Bottom Right Obstructed: {bottom_right_percentage:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Bottom Right Clear", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Apply segmentation mask to the frame
    segmentation_mask_colored = np.zeros_like(frame)
    segmentation_mask_colored[segmentation_mask == 1] = [0, 255, 0]
    frame = cv2.addWeighted(frame, 1, segmentation_mask_colored, 0.5, 0)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
