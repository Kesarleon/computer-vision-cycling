# Conteo de Viajes de Ciclistas con Visión por Computadora

Este proyecto ofrece una solución profesional y atractiva para detectar y contar ciclistas en videos utilizando visión por computadora. La herramienta se presenta a través de una aplicación web interactiva construida con Streamlit, respaldada por un modelo de Red Neuronal Convolucional (CNN) entrenado con TensorFlow/Keras.

<!-- ![Demostración de la Aplicación](URL_A_LA_IMAGEN_DE_DEMO.png) -->
*<p align="center">Un marcador de posición para una futura demostración de la aplicación.</p>*

## ✨ Características Principales

- **Interfaz de Usuario Interactiva:** Una aplicación web moderna y fácil de usar donde puedes subir tus propios videos.
- **Detección por IA:** Utiliza un modelo de TensorFlow/Keras para identificar ciclistas en cada fotograma.
- **Seguimiento de Objetos:** Implementa un tracker de centroides para seguir a los ciclistas detectados a través del tiempo.
- **Conteo por Línea Virtual:** Cuenta automáticamente a los ciclistas que cruzan una línea virtual en el video.
- **Configuración Personalizable:**
  - **Ajuste del Umbral de Detección:** Controla la sensibilidad del modelo de IA para reducir falsos positivos.
  - **Línea de Conteo Dinámica:** Ajusta la posición vertical de la línea de conteo directamente desde la interfaz.
- **Panel de Resultados en Tiempo Real:** Visualiza el conteo, el tiempo de procesamiento y el progreso del análisis mientras se ejecuta.

## 📂 Estructura del Repositorio Profesional

El código ha sido refactorizado para seguir las mejores prácticas, separando la lógica de la interfaz de usuario para mayor claridad y mantenimiento.

```
.
├── .gitignore
├── app.py                  # Aplicación web principal de Streamlit (UI)
├── src/
│   ├── tracker.py          # Módulo para el seguimiento de centroides
│   └── video_processing.py # Lógica principal de procesamiento de video
├── data/
│   └── .gitkeep            # Directorio para videos de entrada
├── notebooks/
│   └── training.ipynb      # Notebook para entrenar el modelo de detección
├── requirements.txt        # Dependencias de Python
├── packages.txt            # Dependencias del sistema para Streamlit Cloud
└── README.md               # Esta documentación
```
*Nota: El modelo `bicycle_detection_model.h5` se genera al ejecutar el notebook y debe gestionarse con Git LFS.*

## 🚀 Cómo Empezar

### 1. Clonar y Preparar el Entorno

Después de clonar el repositorio, se recomienda crear un entorno virtual:
```bash
# Clona el repositorio (si aún no lo has hecho)
# git clone <URL-del-repositorio>
# cd <nombre-del-repositorio>

# Crea y activa el entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Generar el Modelo (Si no existe)

El modelo `bicycle_detection_model.h5` es necesario para ejecutar la aplicación. Si no lo tienes, puedes generarlo ejecutando el notebook de Jupyter:
- Abre y ejecuta `notebooks/training.ipynb`.
- El notebook incluye una celda de simulación para generar datos de entrenamiento de demostración, por lo que no se requiere etiquetado manual para empezar.
- Al finalizar, el modelo se guardará en el directorio raíz.

### 4. Ejecutar la Aplicación

Con el modelo en su lugar, inicia la aplicación Streamlit:

```bash
streamlit run app.py
```

Tu navegador se abrirá con la aplicación.

## 🤖 Cómo Usar la Aplicación

1.  **Elige una Fuente de Video:** En la barra lateral, selecciona si quieres subir un archivo o usar una URL.
2.  **Ajusta los Parámetros:**
    - **Umbral de Confianza:** Desliza para ajustar la sensibilidad de la detección.
    - **Posición de la Línea:** Desliza para cambiar la altura de la línea de conteo.
3.  **Inicia el Análisis:** Haz clic en el botón "🚀 Procesar Video".
4.  **Observa los Resultados:** El video se procesará y mostrará en el panel principal con métricas en tiempo real.

## ☁️ Despliegue en Streamlit Cloud

Puedes desplegar esta aplicación como una demo pública en Streamlit Cloud. Sigue estos pasos:

### 1. Preparar el Repositorio para Archivos Grandes

El archivo del modelo (`bicycle_detection_model.h5`) es demasiado grande para ser gestionado directamente por Git. Necesitarás usar **Git LFS** (Large File Storage).

```bash
# Instala Git LFS en tu sistema (hazlo una sola vez)
# En Debian/Ubuntu: sudo apt-get install git-lfs
# En macOS: brew install git-lfs
git lfs install

# Indica a Git LFS que rastree los archivos .h5
git lfs track "*.h5"

# Asegúrate de que .gitattributes se añada al repositorio (se crea con el comando anterior)
git add .gitattributes

# Añade tu modelo y súbelo
git add bicycle_detection_model.h5
git commit -m "feat: Add trained model via Git LFS"
git push
```

### 2. Desplegar en Streamlit Cloud

1.  **Sube tu repositorio a GitHub.**
2.  **Inicia sesión en [Streamlit Cloud](https://share.streamlit.io/).**
3.  Haz clic en **"New app"** y selecciona tu repositorio.
4.  Streamlit detectará automáticamente `app.py`, `requirements.txt` y `packages.txt`.
5.  Haz clic en **"Deploy!"**.

La aplicación se construirá y desplegará. Gracias a los archivos de configuración, Streamlit Cloud instalará todas las dependencias de Python y del sistema necesarias.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un *issue* para discutir cambios importantes o envía un *pull request* con tus mejoras.
