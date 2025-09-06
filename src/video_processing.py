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

def process_video(video_path, model, line_y, detection_threshold, img_width, img_height, progress_callback):
    """
    Procesa un video para contar ciclistas y produce fotogramas anotados.

    Args:
        video_path (str): Ruta al archivo de video.
        model: Modelo de Keras para la detección.
        line_y (int): Posición vertical de la línea de conteo.
        detection_threshold (float): Umbral de confianza para la detección.
        img_width (int): Ancho de la imagen para el modelo.
        img_height (int): Alto de la imagen para el modelo.
        progress_callback (function): Función para actualizar la barra de progreso.

    Yields:
        tuple: Una tupla conteniendo el fotograma procesado (np.array),
               el conteo actual de bicicletas (int), y el número total de
               fotogramas (int).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error al abrir el archivo de video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    tracker = CentroidTracker(max_disappeared=50)
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

        for (object_id, centroid) in objects.items():
            if object_id not in tracked_paths:
                tracked_paths[object_id] = deque(maxlen=30)

            tracked_paths[object_id].append(centroid)

            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            if len(tracked_paths[object_id]) > 1:
                prev_centroid = tracked_paths[object_id][-2]
                crossed_down = prev_centroid[1] < line_y and centroid[1] >= line_y
                crossed_up = prev_centroid[1] > line_y and centroid[1] <= line_y

                if (crossed_down or crossed_up) and object_id not in counted_ids:
                    bicycle_count += 1
                    counted_ids.add(object_id)
                    # Dibuja la línea en verde momentáneamente para indicar el cruce
                    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 4)

        # Dibujar la línea de conteo estándar
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
        # Mostrar el conteo en el video
        cv2.putText(frame, f"Conteo: {bicycle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        progress_callback(frame_num / total_frames)
        yield frame, bicycle_count

    cap.release()
