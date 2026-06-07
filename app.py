import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(
    page_title="VisionTrack - Object Detection",
    page_icon="👁️",
    layout="wide"
)

# --- Premium CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .vision-header {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(255, 65, 108, 0.3);
    }
    .vision-title { font-size: 3rem; font-weight: 600; margin: 0; }
    .vision-sub { font-size: 1.2rem; font-weight: 300; margin: 0; opacity: 0.9; }
    
    /* Panel styling */
    .control-panel {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    @media (prefers-color-scheme: light) {
        .control-panel {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="vision-header">
    <p class="vision-title">👁️ VisionTrack</p>
    <p class="vision-sub">Real-Time Object Detection & Tracking with YOLOv8</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# --- Layout ---
col_controls, col_video = st.columns([1, 2])

with col_controls:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
    
    source_type = st.radio("Select Source", ["Webcam", "Upload Video", "Upload Image"], index=0)
    
    model_name = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    enable_tracking = st.checkbox("Enable Object Tracking (BoT-SORT)", value=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    model = load_model(model_name)
    
    if source_type == "Webcam":
        run = st.checkbox("🟢 Start Webcam stream")
        stop = st.button("🛑 Stop")
        if stop:
            run = False

with col_video:
    st.markdown("### 📡 Live Feed")
    frame_placeholder = st.empty()
    status_text = st.empty()

    if source_type == "Webcam":
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open Webcam.")
            else:
                status_text.info("Stream started...")
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Cannot read frame from webcam")
                        break
                    
                    if enable_tracking:
                        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
                    else:
                        results = model(frame, conf=conf_threshold, verbose=False)
                        
                    annotated_frame = results[0].plot()
                    
                    # Convert BGR to RGB for Streamlit
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                    
                    # Small sleep to yield execution
                    time.sleep(0.01)
                    
                cap.release()
                status_text.warning("Stream stopped.")
        else:
            status_text.info("Webcam is offline. Check the 'Start Webcam stream' box to begin.")
            
    elif source_type == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            if st.button("▶️ Process Video", use_container_width=True):
                cap = cv2.VideoCapture(tfile.name)
                status_text.info("Processing video...")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if enable_tracking:
                        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
                    else:
                        results = model(frame, conf=conf_threshold, verbose=False)
                        
                    annotated_frame = results[0].plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                    time.sleep(0.01)
                    
                cap.release()
                status_text.success("Processing complete!")

    elif source_type == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            import numpy as np
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("Analyzing image..."):
                results = model(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
            status_text.success(f"Detected {len(results[0].boxes)} object(s).")
