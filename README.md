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
