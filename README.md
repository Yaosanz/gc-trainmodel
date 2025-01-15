# UV Index Prediction Model Training

## Overview

This project implements a machine learning model to predict UV Index values using weather data. The system uses TensorFlow to train a neural network model and includes data preprocessing, feature engineering, and model persistence capabilities.

## Features

* Data preprocessing from JSON weather data format
* Feature engineering including temporal features
* Standardized scaling of input features
* Neural network model using TensorFlow
* Model and scaler persistence for later use

## Model Architecture

The neural network consists of:
* Input layer matching feature dimensions
* Dense layer (128 units) with ReLU activation
* Dropout layer (0.2)
* Dense layer (64 units) with ReLU activation
* Dense layer (32 units) with ReLU activation
* Output layer (1 unit) for UV Index prediction

## Prerequisites

* Python 3.x
* Required packages:
  ```
  pandas
  numpy
  tensorflow
  scikit-learn
  joblib
  ```

## Data Format

The training script expects a JSON file with weather data in the following structure:

```json
{
  "data": {
    "weather": [
      {
        "date": "YYYY-MM-DD",
        "hourly": [
          {
            "time": "HHMM",
            "tempC": "XX.X",
            "windspeedKmph": "XX.X",
            "humidity": "XX.X",
            "cloudcover": "XX.X",
            "precipMM": "XX.X",
            "pressure": "XXXX.X",
            "visibility": "XX.X",
            "FeelsLikeC": "XX.X",
            "uvIndex": "XX"
          }
        ]
      }
    ]
  }
}
```

## Features Used

The model uses the following features:
* Temporal features:
  * Month
  * Day
  * Day of week
  * Hour
* Weather features:
  * Temperature (°C)
  * Wind speed (km/h)
  * Humidity (%)
  * Cloud cover (%)
  * Precipitation (mm)
  * Pressure
  * Visibility
  * Feels like temperature (°C)
* Derived features:
  * Is daylight (binary)
  * Daylight hours

## Usage

1. Install required packages:
   ```bash
   pip install pandas numpy tensorflow scikit-learn joblib
   ```

2. Place your weather data JSON file in the project directory

3. Run the training script:
   ```bash
   python train_model.py
   ```

The script will:
1. Create a `model` directory if it doesn't exist
2. Process the JSON data
3. Prepare features and split the dataset
4. Train the neural network model
5. Save the trained model as `model/uv_model_tf.h5`
6. Save the feature scaler as `model/scaler.pkl`

## Key Functions

### `process_json_to_df(json_file)`
Converts JSON weather data to a pandas DataFrame.

### `prepare_features(df)`
Performs feature engineering including:
* Date/time feature extraction
* Daylight feature creation
* Feature selection

### `train_model(json_file)`
Main function that:
* Processes data
* Prepares features
* Trains the model
* Saves model artifacts

## Model Training Parameters

* Test set size: 20%
* Training epochs: 100
* Batch size: 32
* Validation split: 20%
* Optimizer: Adam
* Loss function: Mean Squared Error (MSE)
* Metric: Mean Absolute Error (MAE)

## Output Files

The training process generates two files:
1. `model/uv_model_tf.h5`: Trained TensorFlow model
2. `model/scaler.pkl`: Feature scaler for preprocessing new data

## Notes

* The daylight hours calculation is simplified for Jakarta location
* The model uses mean squared error loss function suitable for regression
* Default random seed (42) is used for reproducibility

## Example Usage

```python
import tensorflow as tf
import joblib
import numpy as np

# Load the model and scaler
model = tf.keras.models.load_model('model/uv_model_tf.h5')
scaler = joblib.load('model/scaler.pkl')

# Prepare your features (example)
features = np.array([[month, day, day_of_week, hour, temp, wind, humidity, 
                     cloud, precip, pressure, visibility, feels_like, 
                     is_daylight, daylight_hours]])

# Scale features
scaled_features = scaler.transform(features)

# Make prediction
prediction = model.predict(scaled_features)
```

---
For questions or issues, please open a GitHub issue in the repository.