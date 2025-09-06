# Conteo de Viajes de Ciclistas con Visión por Computadora

Este proyecto implementa una solución de visión por computadora para detectar y contar ciclistas en un video. Utiliza un modelo de Red Neuronal Convolucional (CNN) entrenado con TensorFlow/Keras y una aplicación web construida con Streamlit para la interacción del usuario.

## 🎯 Objetivo

El objetivo es proporcionar una herramienta que pueda analizar un archivo de video para contar cuántos ciclistas cruzan una línea virtual predefinida. Este tipo de análisis es valioso para la planificación del tráfico urbano, estudios de movilidad y la promoción de infraestructura para ciclistas.

## 📂 Estructura del Repositorio

```
bicycle-trip-counter/
│
├── app.py                  # Aplicación web de Streamlit
├── data/
│   └── .gitkeep            # Directorio para videos y frames
├── notebooks/
│   └── training.ipynb      # Notebook para el preprocesamiento y entrenamiento del modelo
├── requirements.txt        # Dependencias de Python
├── bicycle_detection_model.h5  # (Generado por el notebook) Modelo entrenado
└── README.md               # Documentación del proyecto
```

## 🛠️ Requisitos Previos

- Python 3.8+
- `pip` para la gestión de paquetes

## 🚀 Cómo Empezar

### 1. Clonar el Repositorio

```bash
git clone <URL-del-repositorio>
cd bicycle-trip-counter
```

### 2. Instalar Dependencias

Se recomienda crear un entorno virtual para aislar las dependencias del proyecto.

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

Instala las librerías necesarias:

```bash
pip install -r requirements.txt
```

### 3. Entrenamiento del Modelo (Opcional)

El repositorio está diseñado para funcionar con un modelo pre-entrenado que puedes generar. Si deseas entrenar el modelo con tu propio dataset, sigue estos pasos:

1.  **Prepara tu Dataset:**
    *   Coloca tus archivos de video (`.mp4`, `.avi`) en el directorio `data/`.
    *   El notebook `notebooks/training.ipynb` contiene funciones para extraer fotogramas de estos videos.

2.  **Etiqueta tus Datos:**
    *   Después de extraer los fotogramas, necesitas etiquetarlos. El notebook genera un archivo `labels.csv` con las columnas `frame` y `has_bicycle`. Deberás llenar este archivo manualmente (0 para no-bicicleta, 1 para bicicleta).

3.  **Ejecuta el Notebook de Entrenamiento:**
    *   Abre y ejecuta el notebook `notebooks/training.ipynb` utilizando Jupyter.
    *   Este proceso cargará los fotogramas y las etiquetas, entrenará el modelo de CNN y guardará el artefacto resultante como `bicycle_detection_model.h5` en el directorio raíz.

### 4. Ejecutar la Aplicación Streamlit

Una vez que tengas el modelo `bicycle_detection_model.h5` (ya sea que lo hayas entrenado tú mismo o lo hayas descargado), puedes iniciar la aplicación.

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador web. Sube un video y la aplicación procesará el metraje para contar los ciclistas que cruzan la línea virtual y mostrará el resultado.

## 🤖 Cómo Funciona

1.  **Detección:** El modelo de CNN analiza cada fotograma del video para detectar la presencia de una bicicleta.
2.  **Seguimiento (Tracking):** Se utiliza un tracker simple para seguir los objetos detectados a través de fotogramas consecutivos.
3.  **Conteo:** Se define una línea horizontal en el centro del fotograma. Un "viaje" se cuenta cuando el centroide de un objeto rastreado (bicicleta) cruza esta línea de arriba hacia abajo o de abajo hacia arriba.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un *issue* para discutir cambios importantes o envía un *pull request* con tus mejoras.
