import streamlit as st
import cv2
import os
import tempfile
import time
import tensorflow as tf
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
        # Usar st.spinner para mostrar que el modelo se está cargando
        with st.spinner("Cargando modelo de IA..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        return None

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Análisis de Video: Conteo de Ciclistas", layout="wide", page_icon="🚴")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("🚴 Análisis de Video IA")
    st.info("Esta aplicación utiliza un modelo de IA para detectar y contar ciclistas en un video.")

    model = load_detection_model()

    if model is None:
        st.error(f"Modelo no encontrado en `{MODEL_PATH}`. Por favor, ejecuta el notebook de entrenamiento.")
        st.stop()
    else:
        st.success("Modelo cargado.")

    st.header("Configuración")
    uploaded_file = st.file_uploader(
        "Sube tu video",
        type=["mp4", "avi", "mov"],
        help="Sube un video para que la IA lo analice."
    )

    detection_threshold = st.slider(
        "Umbral de Confianza de Detección",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Un valor más alto significa que la IA debe estar más segura para detectar una bicicleta. Ayuda a reducir falsos positivos."
    )

    # Placeholder para la línea de conteo, se actualizará después de cargar el video
    line_position_percent = st.slider(
        "Posición de la Línea de Conteo (Vertical)",
        min_value=10, max_value=90, value=50, step=5,
        help="Define la línea virtual que un ciclista debe cruzar para ser contado. Se mide como un porcentaje desde la parte superior del video."
    )

    process_button = st.button("🚀 Iniciar Análisis")

# --- Área Principal ---
st.title("Panel de Control de Conteo de Ciclistas")

if uploaded_file is None:
    st.warning("Por favor, sube un archivo de video usando el panel de la izquierda para comenzar.")

if uploaded_file and process_button:
    # Guardar archivo subido a un archivo temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Obtener dimensiones del video para la línea de conteo
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    line_y = int(h * (line_position_percent / 100))

    # Crear placeholders para la salida
    st_frame = st.empty()
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Conteo Actual", "0")
    with col2:
        st.metric("Tiempo de Procesamiento", "0s")
    with col3:
        st.metric("Progreso", "0%")

    progress_bar = st.progress(0)

    def progress_callback(fraction):
        # Actualiza la métrica de progreso
        col3.metric("Progreso", f"{int(fraction*100)}%")
        progress_bar.progress(fraction)

    try:
        start_time = time.time()

        processor = process_video(
            video_path=video_path,
            model=model,
            line_y=line_y,
            detection_threshold=detection_threshold,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            progress_callback=progress_callback
        )

        final_count = 0
        for frame, count in processor:
            # Actualizar métricas en tiempo real
            col1.metric("Conteo Actual", str(count))
            elapsed_time = f"{time.time() - start_time:.2f}s"
            col2.metric("Tiempo de Procesamiento", elapsed_time)

            # Mostrar el fotograma procesado
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            final_count = count

        end_time = time.time()
        total_time = end_time - start_time

        # Estado final
        st.success(f"¡Análisis completado en {total_time:.2f} segundos!")
        col1.metric("Conteo Final", str(final_count))
        col2.metric("Tiempo Total", f"{total_time:.2f}s")
        col3.metric("Progreso", "100%")
        progress_bar.progress(1.0)

        st.balloons()

    except IOError as e:
        st.error(f"Error al procesar el video: {e}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
    finally:
        # Limpiar archivo temporal
        if os.path.exists(video_path):
            os.remove(video_path)
else:
    st.info("Configure los parámetros en la barra lateral y haga clic en 'Iniciar Análisis'.")
