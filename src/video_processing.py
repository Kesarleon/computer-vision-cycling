import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from src.tracker import CentroidTracker

def detect_bicycles(frame, model, detection_threshold, img_width, img_height):
    """Detecta bicicletas en un fotograma usando el modelo CNN."""
    img = cv2.resize(frame, (img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > detection_threshold:
        h, w, _ = frame.shape
        return [(int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))]
    return []

# --- Funciones para Geometría y Detección de Cruce ---

def orientation(p, q, r):
    """
    Determina la orientación de un triplete ordenado (p, q, r).
    Retorna:
    0 --> p, q y r son colineales
    1 --> Sentido horario (Clockwise)
    2 --> Sentido antihorario (Counterclockwise)
    """
    # Ver https://www.geeksforgeeks.org/orientation-3-ordered-points/
    # para más detalles.
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):
    """Dado que tres puntos p, q, r son colineales, la función verifica
    si el punto q se encuentra en el segmento 'pr'."""
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def do_intersect(p1, q1, p2, q2):
    """Retorna verdadero si el segmento de línea 'p1q1' y 'p2q2' se intersectan."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Caso general: las orientaciones son diferentes
    if o1 != o2 and o3 != o4:
        return True

    # Casos especiales de colinealidad (cuando un punto está en el segmento del otro)
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

def process_video(video_path, model, line_coords, detection_threshold, img_width, img_height, progress_callback):
    """
    Procesa un video para contar ciclistas y produce fotogramas anotados.

    Args:
        video_path (str): Ruta al archivo de video.
        model: Modelo de Keras para la detección.
        line_coords (tuple): Tupla con dos puntos ((x1, y1), (x2, y2)) que definen la línea.
        detection_threshold (float): Umbral de confianza para la detección.
        img_width (int): Ancho de la imagen para el modelo.
        img_height (int): Alto de la imagen para el modelo.
        progress_callback (function): Función para actualizar la barra de progreso.

    Yields:
        tuple: Tupla con el fotograma procesado (np.array) y el conteo actual (int).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error al abrir el archivo de video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    line_p1, line_p2 = line_coords

    tracker = CentroidTracker(max_disappeared=50, max_distance=75)
    tracked_paths = {}
    bicycle_count = 0
    counted_ids = set()

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        rects = detect_bicycles(frame, model, detection_threshold, img_width, img_height)
        objects = tracker.update(rects)

        # Dibujar la línea de conteo principal
        cv2.line(frame, line_p1, line_p2, (0, 0, 255), 2)

        for (object_id, centroid) in objects.items():
            if object_id not in tracked_paths:
                tracked_paths[object_id] = deque(maxlen=30)

            tracked_paths[object_id].append(centroid)

            # Dibuja el centroide y el ID
            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            if len(tracked_paths[object_id]) > 1:
                prev_centroid = tracked_paths[object_id][-2]

                # Comprobar si el trayecto del centroide cruza la línea de conteo
                if object_id not in counted_ids and do_intersect(prev_centroid, centroid, line_p1, line_p2):
                    bicycle_count += 1
                    counted_ids.add(object_id)

                    # Resaltar la línea momentáneamente para indicar el cruce
                    cv2.line(frame, line_p1, line_p2, (0, 255, 0), 4)

        # Mostrar el conteo total en el video
        cv2.putText(frame, f"Conteo: {bicycle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        progress_callback(frame_num / total_frames)
        yield frame, bicycle_count

    cap.release()
