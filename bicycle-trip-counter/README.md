# Bicycle Trip Counter

This project uses Computer Vision to count the number of bicycle trips in a video. It leverages a Convolutional Neural Network (CNN) trained with TensorFlow/Keras to identify cyclists in video frames. A Streamlit application is provided for easy interaction with the model.

## Table of Contents
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)

## Project Objective

The main goal of this project is to provide an automated way to count the number of cyclists in a video. This can be useful for traffic analysis, urban planning, and monitoring of cycling infrastructure.

## Dataset

The model was trained on a custom dataset of videos containing cyclists. For your own training, you should gather a collection of videos and extract frames. The frames should be divided into two classes: `cyclist` and `no_cyclist`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bicycle-trip-counter.git
   cd bicycle-trip-counter
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can use the pre-trained model with the Streamlit application to count cyclists in your own videos.

## Training the Model

The `notebooks/training.ipynb` notebook contains all the steps to train the CNN model.

1. **Exploratory Data Analysis (EDA):** Understand the data distribution and characteristics.
2. **Data Preprocessing:** Prepare the images for training, including resizing and normalization.
3. **Model Training:** Build and train the CNN model.
4. **Model Evaluation:** Evaluate the model's performance on a test set.

## Running the Application

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py
```

This will open a web interface where you can upload a video and see the cyclist count.
