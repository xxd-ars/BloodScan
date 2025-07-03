import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_annotations(image, label_path):
    """Draw annotations from YOLO label file onto the image."""
    h, w, _ = image.shape

    # Read the label file
    with open(label_path, 'r') as f:
        annotations = f.readlines()

    for annotation in annotations:
        label_data = list(map(float, annotation.strip().split()))
        class_id = int(label_data[0])
        points = np.array(label_data[1:]).reshape(-1, 2)
        points[:, 0] *= w
        points[:, 1] *= h

        # Draw polygon on the image
        points = points.astype(np.int32)
        if class_id == 0:
            color_ = (0, 255, 255)
        elif class_id == 1:
            color_ = (0, 255, 0)
        else:
            color_ = (0, 0, 255)
        cv2.polylines(image, [points], isClosed=True, color=color_, thickness=2)
        
        # Draw class ID near the first point
        if len(points) > 0:
            cv2.putText(image, str(class_id), (points[0][0], points[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image

def display_image_with_annotations(image_path, label_path, outpt_path, display = False):
    """Display the original image and the annotated image side by side."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create a copy for drawing annotations
    annotated_image = image.copy()

    # Draw annotations on the copy
    if os.path.exists(label_path):
        annotated_image = draw_annotations(annotated_image, label_path)
    else:
        print(f"Label file not found: {label_path}")

    # Convert images from BGR to RGB for displaying with Matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    plt.imsave(outpt_path, annotated_image)
    if display:
        # Display the images side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Annotated Image")
        plt.imshow(annotated_image)
        
        plt.axis("off")

        plt.tight_layout()
        plt.show()

# Example usage
file_names = ['train', 'test', 'valid']
for file_name in file_names:
    files_path = f'./tests/yolo_seg/datasets/Blood-Scan-8-cropped/{file_name}/'

    for img_file in os.listdir(files_path + 'images/'):
        print(img_file)
        image_path = os.path.join(files_path + 'images/', img_file)
        label_path = os.path.join(files_path + 'labels/', img_file.replace('.jpg', '.txt'))
        
        outpt_path = os.path.join(files_path + 'images_cropped/', img_file)
        display_image_with_annotations(image_path, label_path, outpt_path)
    
# image_name = '2022-03-28_103204_1_T5_2348_bmp.rf.37ae53cd248aaed83e2ec86fc0f4d45e'
# image_path = files_path + 'images/' + image_name + '.jpg'  # Replace with your image path
# outpt_path = files_path + 'images_cropped/' + image_name + '.jpg'  # Replace with your image path
# label_path = files_path + 'labels/' + image_name + '.txt'  # Replace with your label path
# display_image_with_annotations(image_path, label_path, outpt_path)