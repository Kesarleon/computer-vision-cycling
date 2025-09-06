import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import tempfile
from collections import deque

# --- Constantes y Configuración ---
MODEL_PATH = 'bicycle_detection_model.h5'
IMG_HEIGHT, IMG_WIDTH = 128, 128
DETECTION_THRESHOLD = 0.5  # Umbral para considerar una detección como positiva

# --- Carga del Modelo ---
@st.cache(allow_output_mutation=True)
def load_detection_model():
    """Carga el modelo de detección de Keras."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# --- Lógica de Detección y Conteo ---
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        return self.objects

def detect_bicycles(frame, model):
    """Detecta bicicletas en un fotograma usando el modelo CNN."""
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > DETECTION_THRESHOLD:
        # Simula un bounding box si se detecta una bicicleta
        # En un modelo real de detección, obtendríamos las coordenadas del BBox
        h, w, _ = frame.shape
        return [(int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))]
    return []

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Contador de Viajes Ciclistas", layout="wide")

st.title("🚴‍♀️ Contador de Viajes de Ciclistas")
st.markdown("Sube un video para analizar y contar los ciclistas que cruzan la línea central.")

model = load_detection_model()

if model is None:
    st.error("No se pudo cargar el modelo `bicycle_detection_model.h5`. Asegúrate de que el archivo existe en el directorio raíz del proyecto. Puedes generarlo ejecutando el notebook `notebooks/training.ipynb`.")
else:
    st.success("✅ Modelo de detección cargado correctamente.")

uploaded_file = st.file_uploader("Elige un archivo de video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("Procesar Video para Contar Ciclistas"):
        with st.spinner("Analizando video... Esto puede tardar unos momentos."):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error al abrir el video.")
            else:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                # Línea de conteo en el centro
                line_y = h // 2

                tracker = CentroidTracker()
                tracked_paths = {}
                bicycle_count = 0
                counted_ids = set()

                st_frame = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rects = detect_bicycles(frame, model)
                    objects = tracker.update(rects)

                    for (object_id, centroid) in objects.items():
                        # Guardar el historial de posiciones
                        if object_id not in tracked_paths:
                            tracked_paths[object_id] = deque(maxlen=30)

                        tracked_paths[object_id].append(centroid)

                        # Dibujar el centroide y el ID
                        text = f"ID {object_id}"
                        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                        # Lógica de cruce de línea
                        if len(tracked_paths[object_id]) > 1:
                            prev_centroid = tracked_paths[object_id][-2]
                            # Si el objeto no ha sido contado y cruza la línea
                            if prev_centroid[1] < line_y and centroid[1] >= line_y and object_id not in counted_ids:
                                bicycle_count += 1
                                counted_ids.add(object_id)
                                cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 4) # Línea verde al cruzar
                            elif prev_centroid[1] > line_y and centroid[1] <= line_y and object_id not in counted_ids:
                                bicycle_count += 1
                                counted_ids.add(object_id)
                                cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 4) # Línea verde al cruzar

                    # Dibujar la línea de conteo
                    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

                    # Mostrar el conteo en el video
                    cv2.putText(frame, f"Conteo: {bicycle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Mostrar el fotograma procesado
                    st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                cap.release()
                st.success("¡Análisis de video completado!")
                st.metric("Total de Viajes de Ciclistas Contados", bicycle_count)

    # Limpiar archivo temporal
    if os.path.exists(video_path):
        os.remove(video_path)
