import streamlit as st
import cv2
import os
import tempfile
import time
import requests
from src.video_processing import process_video

# --- Constantes y Configuración ---
YOLO_MODEL_DIR = "yolo_model"
YOLO_CONFIG_PATH = os.path.join(YOLO_MODEL_DIR, "yolov3.cfg")
YOLO_WEIGHTS_PATH = os.path.join(YOLO_MODEL_DIR, "yolov3.weights")
YOLO_NAMES_PATH = os.path.join(YOLO_MODEL_DIR, "coco.names")

YOLO_CONFIG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
YOLO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

def download_file_with_progress(url, path, message):
    st.info(message)
    if "yolov3.weights" in path:
        st.warning("Este archivo es grande (aprox. 240 MB) y la descarga puede tardar varios minutos.")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

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
                file_name = os.path.basename(path)
                progress_status.text(f"Descargando {file_name}: {int(downloaded_size / chunk_size)} / {int(total_size / chunk_size)} MB")

        progress_status.text(f"¡Descarga de {os.path.basename(path)} completada!")
        progress_bar.empty()

    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo '{os.path.basename(path)}': {e}")
        st.stop()
    except IOError as e:
        st.error(f"No se pudo escribir el archivo '{os.path.basename(path)}' en el disco: {e}")
        st.stop()

def setup_yolo_model():
    """Verifica si los archivos del modelo YOLO existen y, si no, los descarga."""
    if not os.path.exists(YOLO_MODEL_DIR):
        os.makedirs(YOLO_MODEL_DIR)

    if not os.path.exists(YOLO_CONFIG_PATH):
        download_file_with_progress(YOLO_CONFIG_URL, YOLO_CONFIG_PATH, "Descargando archivo de configuración de YOLO...")

    if not os.path.exists(YOLO_WEIGHTS_PATH):
        download_file_with_progress(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH, "Descargando pesos del modelo YOLOv3...")

    if not os.path.exists(YOLO_NAMES_PATH):
        download_file_with_progress(YOLO_NAMES_URL, YOLO_NAMES_PATH, "Descargando nombres de clases de YOLO...")

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Análisis de Video: Conteo de Ciclistas", layout="wide", page_icon="🚴")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("🚴 Análisis de Video IA")
    st.info("Esta aplicación utiliza un modelo de IA para detectar y contar ciclistas en un video.")

    # Configurar y verificar el modelo YOLO
    setup_yolo_model()
    st.success("Modelo de detección listo.")

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
    st.subheader("Registro de Análisis en Tiempo Real")
    log_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        start_time = time.time()

        processor = process_video(
            video_path=video_path,
            line_coords=line_coords,
            detection_threshold=detection_threshold
        )

        log_entries = []
        final_count = 0
        for frame, count, progress in processor:
            # Actualizar la barra de progreso
            progress_bar.progress(progress)

            # Crear y añadir el registro
            elapsed_time = f"{time.time() - start_time:.2f}s"
            log_text = f"**Progreso:** {int(progress*100)}% | **Conteo Actual:** {count} | **Tiempo:** {elapsed_time}"
            log_entries.insert(0, log_text) # Insertar al principio para orden descendente

            # Mostrar los registros
            log_placeholder.markdown("\n\n".join(log_entries))

            # Mostrar el fotograma procesado
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            final_count = count

        end_time = time.time()
        total_time = end_time - start_time

        # Estado final
        st.success(f"¡Análisis completado en {total_time:.2f} segundos!")
        final_log = f"**FINALIZADO** | **Conteo Final:** {final_count} | **Tiempo Total:** {total_time:.2f}s"
        log_entries.insert(0, final_log)
        log_placeholder.markdown("\n\n".join(log_entries))
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
