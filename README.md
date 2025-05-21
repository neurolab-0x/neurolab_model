# NeuroLab: EEG Data Analysis and Mental State Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)
- [Model Interpretability Features](#model-interpretability-features)

## 🔭 Overview

NeuroLab is a sophisticated EEG (Electroencephalogram) data analysis platform that leverages machine learning to classify mental states in real-time. The system processes EEG signals to identify various mental states such as stress, calmness, and neutrality, making it valuable for applications in mental health monitoring, neurofeedback, and brain-computer interfaces.

## ✨ Features

- **Real-time EEG Processing**: Stream and analyze EEG data in real-time
- **Multiple File Format Support**: Compatible with .edf, .bdf, .gdf, and .csv formats
- **Advanced Signal Processing**: Comprehensive preprocessing and feature extraction pipeline
- **Machine Learning Integration**: Hybrid model approach with automated calibration
- **RESTful API**: FastAPI-powered endpoints for seamless integration
- **Scalable Architecture**: Modular design for easy extension and maintenance
- **Automated Recommendations**: AI-driven insights and recommendations

## 🏗 System Architecture

```
eeg-ds/
├── api/                 # API endpoints and routing
├── config/             # Configuration files
├── data/               # Raw data storage
├── models/             # ML model implementations
├── preprocessing/      # Data preprocessing modules
├── processed/          # Processed data and trained models
├── utils/              # Utility functions
├── main.py            # Application entry point
└── requirements.txt    # Project dependencies
```

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/asimwe1/eeg-ds.git
   cd eeg-ds
   ```

2. **Create a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   ```bash
   cp .env.example .env
   # Configure your .env file with appropriate settings
   ```

## 💻 Usage

### Starting the Server
```bash
uvicorn main:app --reload
```

### Processing EEG Data
```python
import requests

# Upload EEG file for analysis
files = {'file': open('your_eeg_file.csv', 'rb')}
response = requests.post('http://neurai/upload', files=files)

# Real-time processing
data = {'eeg_data': your_eeg_data}
response = requests.post('http://neurai/realtime', json=data)
```

## 📚 API Documentation

### Endpoints

- `POST /upload`: Upload and process EEG files
  - Supports files up to 500MB
  - Returns processed results and analysis

- `POST /realtime`: Real-time EEG data processing
  - Accepts streaming EEG data
  - Returns immediate analysis results

- `POST /retrain`: Retrain the model with new data
  - Requires authenticated access
  - Returns training metrics

- `GET /health`: System health check
  - Monitors system status
  - Returns service metrics

## 🔄 Data Processing Pipeline

1. **Data Loading**
   - File validation and format checking
   - Initial data structure verification

2. **Preprocessing**
   - Artifact removal
   - Signal filtering
   - Normalization

3. **Feature Extraction**
   - Temporal features
   - Frequency domain analysis
   - Statistical measures

4. **State Classification**
   - Mental state prediction
   - Confidence scoring
   - Temporal smoothing

## 🧠 Model Training

### Training Process
1. Data preparation and splitting
2. Feature engineering
3. Model selection and hyperparameter tuning
4. Cross-validation
5. Model calibration
6. Performance evaluation

### Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

AI Model Maintainer - [Mugisha Prosper](mailto:polo1.mugisha@gmail.com)

Project Link: [Neurolabs Inc](https://neurolab.cc)

## Model Interpretability Features

The platform now includes comprehensive model interpretability capabilities:

### SHAP (SHapley Additive exPlanations)
- Explains model predictions by attributing feature importance values
- Helps understand which EEG features contribute most to each mental state classification
- Available via API endpoint: `/interpretability/explain?explanation_type=shap`

### LIME (Local Interpretable Model-agnostic Explanations)
- Provides local explanations for individual predictions
- Explains specific predictions by approximating the model locally
- Available via API endpoint: `/interpretability/explain?explanation_type=lime`
- Can be included in real-time streaming responses with `include_interpretability=true`

### Confidence Calibration
- Ensures model confidence scores accurately reflect true probabilities
- Implements temperature scaling, Platt scaling, and isotonic regression methods
- Available via API endpoint: `/interpretability/calibrate?method=temperature_scaling`
- Improves reliability of mental state classifications

### Reliability Diagrams
- Visual representation of model calibration
- Shows how predicted probabilities match observed frequencies
- Available via API endpoint: `/interpretability/reliability_diagram`

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install interpretability packages (optional)
pip install shap lime
```

## Usage

```python
from utils.interpretability import ModelInterpretability

# Create interpreter with your model
interpreter = ModelInterpretability(model)

# Get SHAP explanations
shap_results = interpreter.explain_with_shap(X_data)

# Get LIME explanations
lime_results = interpreter.explain_with_lime(X_data, sample_idx=0)

# Calibrate model confidence
cal_results = interpreter.calibrate_confidence(X_val, y_val, method='temperature_scaling')

# Make predictions with calibrated confidence
predictions = interpreter.predict_with_calibration(X_test)
```

# Neurolab AI Model Server

A FastAPI-based server for processing EEG data and detecting neurological events.

## Features

- EEG data storage and retrieval
- Session management
- Event detection and storage
- Real-time data processing
- RESTful API endpoints

## Prerequisites

- Python 3.8+
- MongoDB
- InfluxDB

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neurolab_model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the root directory with the following variables:
```env
MONGODB_URL=mongodb://localhost:27017
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token-here
INFLUXDB_ORG=neurolab
```

## Running the Application

1. Start MongoDB and InfluxDB services

2. Run the application:
```bash
python main.py
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## API Endpoints

### EEG Data
- `POST /eeg/data` - Store a single EEG data point
- `POST /eeg/session` - Store a complete EEG session
- `GET /eeg/session/{session_id}` - Retrieve a session
- `GET /eeg/data/{session_id}` - Retrieve EEG data for a time range

### Events
- `POST /events` - Store a detected event
- `GET /events/{session_id}` - Retrieve events for a time range

## Development

The application uses:
- FastAPI for the web framework
- Motor for async MongoDB operations
- InfluxDB Client for time-series data
- Pydantic for data validation

## License

[Your License Here]
