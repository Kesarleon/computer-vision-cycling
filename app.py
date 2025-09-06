import streamlit as st
import cv2
import os
import tempfile
import time
import tensorflow as tf
import requests
from src.video_processing import process_video

# --- Constantes y Configuración ---
MODEL_PATH = 'bicycle_detection_model.h5'
IMG_HEIGHT, IMG_WIDTH = 128, 128

# --- Carga del Modelo ---
@st.cache_resource
def load_detection_model():
    """Carga el modelo de detección de Keras. Cacheado para alto rendimiento."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with st.spinner("Cargando modelo de IA..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        return None

# --- Funciones de Ayuda ---
def download_video(url):
    """Descarga un video desde una URL y lo guarda en un archivo temporal."""
    try:
        with st.spinner(f"Descargando video desde {url}..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            with open(tfile.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return tfile.name
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el video: {e}")
        return None

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Análisis de Video: Conteo de Ciclistas", layout="wide", page_icon="🚴")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("🚴 Análisis de Video IA")
    st.info("Esta aplicación utiliza IA para detectar y contar ciclistas en un video.")

    model = load_detection_model()
    if model is None:
        st.error(f"Modelo no encontrado en `{MODEL_PATH}`. Ejecuta el notebook de entrenamiento.")
        st.stop()
    else:
        st.success("Modelo cargado.")

    st.header("1. Elige una Fuente de Video")
    source_option = st.radio("Selecciona una opción:", ("Subir un archivo", "Usar URL de un video"))

    uploaded_file = None
    video_url = ""
    if source_option == "Subir un archivo":
        uploaded_file = st.file_uploader("Sube tu video", type=["mp4", "avi", "mov"])
    else:
        video_url = st.text_input("Pega la URL del video aquí", "https://www.pexels.com/es-es/video/28384723/")

    st.header("2. Ajusta los Parámetros")
    detection_threshold = st.slider("Umbral de Confianza", 0.1, 0.9, 0.5, 0.05)
    line_position_percent = st.slider("Posición de la Línea de Conteo", 10, 90, 50, 5)

    st.header("3. Inicia el Análisis")
    process_button = st.button("🚀 Procesar Video")

# --- Área Principal ---
st.title("Panel de Control de Conteo de Ciclistas")

video_path = None
if process_button:
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    elif video_url:
        video_path = download_video(video_url)
    else:
        st.warning("Por favor, sube un video o proporciona una URL para comenzar.")

if video_path:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("No se puede abrir el video. La URL puede ser inválida o el formato no es compatible.")
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        line_y = int(h * (line_position_percent / 100))

        st_frame = st.empty()
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Conteo Actual", "0")
        col2.metric("Tiempo de Procesamiento", "0s")
        col3.metric("Progreso", "0%")
        progress_bar = st.progress(0)

        def progress_callback(fraction):
            col3.metric("Progreso", f"{int(fraction*100)}%")
            progress_bar.progress(fraction)

        start_time = time.time()
        processor = process_video(
            video_path=video_path, model=model, line_y=line_y,
            detection_threshold=detection_threshold, img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT, progress_callback=progress_callback
        )

        final_count = 0
        for frame, count in processor:
            col1.metric("Conteo Actual", str(count))
            elapsed_time = f"{time.time() - start_time:.2f}s"
            col2.metric("Tiempo de Procesamiento", elapsed_time)
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            final_count = count

        total_time = time.time() - start_time
        st.success(f"¡Análisis completado en {total_time:.2f} segundos!")
        col1.metric("Conteo Final", str(final_count))
        col2.metric("Tiempo Total", f"{total_time:.2f}s")
        col3.metric("Progreso", "100%")
        progress_bar.progress(1.0)
        st.balloons()

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
else:
    if not process_button:
        st.info("Configura los parámetros en la barra lateral y haz clic en 'Procesar Video'.")
