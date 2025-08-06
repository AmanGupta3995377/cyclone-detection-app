import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import shutil
import uuid
import sys

# Set model path (relative path works for both local and cloud)
MODEL_PATH = Path("best.pt")

# Load YOLOv5 model once
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH), force_reload=False)

model = load_model()
model.conf = 0.25  # YOLO confidence threshold

# Streamlit UI
st.title("🌪️ Cyclone Detection from Satellite Images")
st.write("Upload a satellite image to detect cyclone presence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create temporary upload folder
    temp_dir = Path("temp_upload")
    temp_dir.mkdir(exist_ok=True)

    # Unique filename to avoid caching issues
    image_path = temp_dir / f"{uuid.uuid4().hex}_{uploaded_file.name}"

    # Save uploaded image
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display uploaded image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Cyclone"):
        st.info("Running detection...")

        # Run YOLO detection
        results = model(str(image_path))
        results.render()  # Apply bounding boxes to image

        # Get confidence scores
        confidences = results.xyxy[0][:, 4].cpu().numpy() if results.xyxy[0].shape[0] > 0 else []

        # Check for detection with confidence > 0.70
        if any(conf > 0.70 for conf in confidences):
            result_img = Image.fromarray(results.ims[0])
            st.success("✅ Cyclone detected!")
            st.image(result_img, caption="Detection Result", use_container_width=True)
        else:
            st.warning("⚠️ No cyclone detected.")

    # Cleanup
    shutil.rmtree(temp_dir)
