import streamlit as st
import cv2
import os
import tempfile
import time
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from src.video_processing import process_video
except ImportError as e:
    st.error(f"Error importing video processing module: {e}")
    st.stop()

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Análisis de Video: Conteo de Ciclistas", layout="wide", page_icon="🚴")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("🚴 Análisis de Video IA")
    st.info(
        "Esta aplicación utiliza el modelo de IA **YOLOv8** para detectar y contar "
        "ciclistas en un video. El modelo se descarga automáticamente."
    )

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
        
        # Show initial status
        with st.spinner("Inicializando modelo YOLOv8... Esto puede tomar unos minutos la primera vez."):
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
