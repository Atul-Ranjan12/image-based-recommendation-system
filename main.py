import os
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# Set up directories for images
image_dir = './Images'

# Ensure the zip file is extracted (if needed)
# with zipfile.ZipFile('flickr8k.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')


# Load the image captioning model with device argument
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Load the YOLOv8 model
model = YOLO("./yolov8s.pt")  # Load the model


# Function to run object detection
def detect_objects(image_path):
    image = Image.open(image_path)
    results = model(image, verbose=False)

    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            object_info = {
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "box": box.xyxy.tolist()[0]
            }
            detected_objects.append(object_info)

    return detected_objects


# Function to create the DataFrame with tqdm progress bar
def generate_image_sentiment_df(image_dir):
    data = []
    image_files = [img for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Use tqdm to create a progress bar
    print("Executing script...")
    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)

        # Step 1: Generate caption for the image
        caption = image_to_text(img_path)[0]['generated_text']

        # Step 2: Detect objects in the image
        objects = detect_objects(img_path)
        simplified_objects = [{"class": obj['class'], "confidence": obj['confidence']} for obj in objects]

        # Add a row to the data list
        data.append({
            'image_path': img_path,
            'generated_caption': caption,
            'objects_detected': simplified_objects,
        })

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(data)
    print("Finished execution!")
    return df

# Generate the DataFrame
df = generate_image_sentiment_df(image_dir)

# Display the DataFrame
print(df)

# Save DataFrame to a CSV file
df.to_csv('image_captioning_sentiment_results.csv', index=False)
