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
# --- Añadir URL del modelo ---
# IMPORTANTE: Sube tu archivo 'bicycle_detection_model.h5' a un servicio de hosting
# (como un release de GitHub, Google Drive, etc.) y pega aquí el enlace de descarga directa.
MODEL_URL = "https://github.com/DagsAd/TFM-Dasboard-Streamlit-Computer-Vision/raw/main/bicycle_detection_model.h5"
IMG_HEIGHT, IMG_WIDTH = 128, 128

def download_model(url, path):
    """Descarga el modelo desde una URL y muestra una barra de progreso."""
    st.info(f"El modelo no se encuentra localmente. Descargando desde la nube...")
    st.warning("Esto puede tardar unos minutos la primera vez.")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lanza un error para respuestas 4xx/5xx

        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024 # 1 MB

        progress_bar = st.progress(0)
        progress_status = st.empty()

        with open(path, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded_size += len(chunk)
                progress = downloaded_size / total_size if total_size > 0 else 0
                progress_bar.progress(min(progress, 1.0))
                progress_status.text(f"Descargando... {int(downloaded_size / chunk_size)} / {int(total_size / chunk_size)} MB")

        progress_status.text("¡Descarga completada!")
        progress_bar.empty()

    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el modelo: {e}")
        st.error("Por favor, verifica la URL del modelo y tu conexión a internet.")
        # Si la descarga falla, detenemos la app para evitar más errores.
        st.stop()
    except IOError as e:
        st.error(f"No se pudo escribir el archivo del modelo en el disco: {e}")
        st.stop()


# --- Carga del Modelo ---
@st.cache_resource
def load_detection_model():
    """
    Carga el modelo de detección de Keras.
    Si no existe localmente, lo descarga desde una URL.
    """
    # Comprobar si el modelo existe. Si no, descargarlo.
    if not os.path.exists(MODEL_PATH):
        download_model(MODEL_URL, MODEL_PATH)

    # Una vez que el modelo está (o ha sido) descargado, cargarlo.
    try:
        with st.spinner("Cargando modelo de IA en memoria..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        st.error("El archivo del modelo podría estar corrupto. Intenta borrarlo para que se descargue de nuevo.")
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

    st.header("Parámetros de la Línea de Conteo")
    line_type = st.selectbox(
        "Tipo de Línea",
        ("Horizontal", "Vertical", "Inclinada"),
        help="Selecciona la orientación de la línea de conteo."
    )

    line_coords_percent = {}
    if line_type == "Horizontal":
        line_coords_percent['y1'] = st.slider(
            "Posición Vertical (%)", 1, 99, 50,
            help="Posición de la línea horizontal desde la parte superior del video."
        )
    elif line_type == "Vertical":
        line_coords_percent['x1'] = st.slider(
            "Posición Horizontal (%)", 1, 99, 50,
            help="Posición de la línea vertical desde la izquierda del video."
        )
    elif line_type == "Inclinada":
        col1, col2 = st.columns(2)
        with col1:
            line_coords_percent['x1'] = st.slider("Punto 1 - X (%)", 1, 99, 25)
            line_coords_percent['y1'] = st.slider("Punto 1 - Y (%)", 1, 99, 25)
        with col2:
            line_coords_percent['x2'] = st.slider("Punto 2 - X (%)", 1, 99, 75)
            line_coords_percent['y2'] = st.slider("Punto 2 - Y (%)", 1, 99, 75)


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

    # Obtener dimensiones del video y calcular coordenadas de la línea
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Calcular coordenadas de la línea en píxeles
    if line_type == "Horizontal":
        y = int(h * (line_coords_percent['y1'] / 100))
        line_coords = ((0, y), (w, y))
    elif line_type == "Vertical":
        x = int(w * (line_coords_percent['x1'] / 100))
        line_coords = ((x, 0), (x, h))
    elif line_type == "Inclinada":
        x1 = int(w * (line_coords_percent['x1'] / 100))
        y1 = int(h * (line_coords_percent['y1'] / 100))
        x2 = int(w * (line_coords_percent['x2'] / 100))
        y2 = int(h * (line_coords_percent['y2'] / 100))
        line_coords = ((x1, y1), (x2, y2))
    else: # Fallback
        line_coords = ((0, h // 2), (w, h // 2))


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
            line_coords=line_coords,
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
