# Conteo de Viajes de Ciclistas con Visión por Computadora

Este proyecto ofrece una solución profesional y atractiva para detectar, seguir y contar ciclistas en videos utilizando un modelo de visión por computadora. La herramienta se presenta a través de una aplicación web interactiva construida con Streamlit, que utiliza el detector de objetos **YOLOv3** para un análisis preciso y en tiempo real.

<!-- ![Demostración de la Aplicación](URL_A_LA_IMAGEN_DE_DEMO.png) -->
*<p align="center">Un marcador de posición para una futura demostración de la aplicación.</p>*

## ✨ Características Principales

- **Interfaz de Usuario Interactiva:** Una aplicación web moderna y fácil de usar donde puedes subir tus propios videos.
- **Detección por IA con YOLOv3:** Utiliza un modelo YOLOv3 pre-entrenado para identificar la ubicación exacta de los ciclistas en cada fotograma.
- **Seguimiento de Objetos:** Implementa un tracker de centroides para seguir a los ciclistas detectados a lo largo del video.
- **Conteo por Línea Virtual:** Cuenta automáticamente a los ciclistas que cruzan una línea virtual personalizable (horizontal, vertical o inclinada).
- **Configuración Personalizable:**
  - **Ajuste del Umbral de Detección:** Controla la sensibilidad del modelo para reducir falsos positivos.
  - **Línea de Conteo Dinámica:** Ajusta la posición de la línea de conteo directamente desde la interfaz.
- **Panel de Resultados en Tiempo Real:** Visualiza el conteo, el progreso del análisis y el tiempo transcurrido mientras se procesa el video.

## 📂 Estructura del Repositorio

El código ha sido refactorizado para seguir las mejores prácticas, separando la lógica de la interfaz de usuario para mayor claridad y mantenimiento.

```
bicycle-trip-counter/
│
├── app.py                  # Aplicación web principal de Streamlit (UI)
├── src/
│   ├── tracker.py          # Módulo para el seguimiento de centroides
│   └── video_processing.py # Lógica principal de procesamiento de video
├── data/
│   └── .gitkeep            # Directorio para videos de entrada
├── requirements.txt        # Dependencias de Python
└── README.md               # Esta documentación
```

## 🚀 Cómo Empezar

### 1. Clonar y Preparar el Entorno

```bash
git clone <URL-del-repositorio>
cd bicycle-trip-counter
# Se recomienda crear un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar Dependencias

Asegúrate de tener todas las librerías necesarias.

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación

La aplicación se encarga de todo lo demás. La primera vez que la inicies, **descargará automáticamente los archivos del modelo YOLOv3** (aproximadamente 240 MB). Este proceso solo ocurre una vez.

```bash
streamlit run app.py
```

Tu navegador se abrirá con la aplicación lista para usarse.

## 🤖 Cómo Usar la Aplicación

1.  **Sube un Video:** Usa el cargador de archivos en la barra lateral izquierda.
2.  **Ajusta los Parámetros:**
    - **Umbral de Confianza:** Desliza para ajustar la sensibilidad de la detección. Un valor más alto requiere que el modelo esté más seguro.
    - **Tipo y Posición de la Línea:** Elige entre una línea horizontal, vertical o inclinada y ajusta su posición en el video.
3.  **Inicia el Análisis:** Haz clic en el botón "**🚀 Iniciar Análisis**".
4.  **Observa los Resultados:** El video se procesará y mostrará en el panel principal. Las métricas de conteo y progreso se actualizarán en tiempo real.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un *issue* para discutir cambios importantes o envía un *pull request* con tus mejoras.
