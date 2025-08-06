import streamlit as st
import torch
from PIL import Image
import shutil
from pathlib import Path
import uuid
import cv2
import numpy as np

# Load model only once
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom',
                          path='C:/code/satellite-image-detection/yolov5/runs/train/cyclone_detector/weights/best.pt',
                          force_reload=False)

model = load_model()
model.conf = 0.25  # set YOLO model confidence threshold

# Streamlit UI
st.title("🌪️ Cyclone Detection from Satellite Images")
st.write("Upload a satellite image to detect cyclone presence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = Path("temp_upload")
    temp_dir.mkdir(exist_ok=True)
    
    unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    image_path = temp_dir / unique_filename

    # Save uploaded file
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display input image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Cyclone"):
        st.info("Running detection...")

        # Run detection
        results = model(str(image_path))
        results.render()  # Updates results.imgs with boxes and labels

        # Extract confidence scores
        confidences = results.xyxy[0][:, 4].cpu().numpy() if results.xyxy[0].shape[0] > 0 else []

        # Filter detections with confidence > 0.70
        if any(conf > 0.70 for conf in confidences):
            result_img = Image.fromarray(results.ims[0])
            st.success("✅ Cyclone detected!")
            st.image(result_img, caption="Detection Result", use_container_width=True)
        else:
            st.warning("⚠️ No cyclone detected.")

    # Cleanup temp upload directory
    shutil.rmtree(temp_dir)
