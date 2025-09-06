import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# --- Model Loading ---
MODEL_PATH = 'cyclist_detection_model.h5'

@st.cache(allow_output_mutation=True)
def load_model():
    """Loads the trained Keras model."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_model()

# --- Prediction Function ---
def predict_cyclist(frame, model):
    """
    Preprocesses a video frame and uses the loaded model to predict
    if a cyclist is present.
    """
    if model is None:
        st.error("Model file not found. Please train the model using the notebook and place 'cyclist_detection_model.h5' in the root directory.")
        return None

    # Preprocess the frame to match the model's input requirements
    img = cv2.resize(frame, (150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    return prediction[0][0] > 0.5 # Returns True if a cyclist is detected

# --- Streamlit App ---
st.title("Bicycle Trip Counter")

if model is None:
    st.warning("Model not found. Please run the training notebook to generate the model file.")
else:
    st.success("Model loaded successfully!")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None and model is not None:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    video = cv2.VideoCapture(temp_video_path)

    cyclist_count = 0
    frame_skip = 15  # Process every 15th frame
    frame_count = 0
    detected_in_previous_frame = False

    st.write("Processing video...")
    progress_bar = st.progress(0)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            is_cyclist = predict_cyclist(frame, model)
            if is_cyclist and not detected_in_previous_frame:
                cyclist_count += 1
                detected_in_previous_frame = True
            elif not is_cyclist:
                detected_in_previous_frame = False

        frame_count += 1
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)

    video.release()
    os.remove(temp_video_path)

    st.header(f"Estimated Cyclist Count: {cyclist_count}")
    st.balloons()
