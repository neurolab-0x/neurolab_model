## EEG Data Analysis and Mental State Classification

## Project Overview
This project aims to analyze EEG (electroencephalogram) data to classify mental states such as stress, calmness, and neutrality. By leveraging machine learning techniques, we can gain insights into how brain activity correlates with different mental conditions.
## Background
Understanding brain activity through EEG data can provide valuable insights into mental states. This project employs machine learning to analyze EEG signals and classify them into different states, which could be useful for applications in mental health, gaming, and brain-computer interfaces.
## Dataset
The dataset used in this project is derived from [CSV](feature_raw.csv), containing EEG readings labeled with corresponding mental states. Each sample includes multiple EEG channel readings.
## Technologies Used
- **Python**: Programming language used for data processing and modeling.
- **Pandas**: Library for data manipulation and analysis.
- **NumPy**: Library for numerical computations.
- **Scikit-Learn**: Machine learning library for model training and evaluation.
- **Matplotlib**: Library for data visualization.

## Installation
To set up this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/asimwe1/eeg-ds.git
   cd eeg-ds
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

## Usage
1. Load the data:
   ```python
   import pandas as pd
   data = pd.read_csv('data/features_raw.csv')
   ```
2. Run the data preprocessing, model training, and evaluation scripts in the order specified in the main script.
   ``` python
   python app.py
   ```
## Data Preprocessing
Data preprocessing steps include:
- Handling missing values
- Normalizing the data
- Adding a mental state column based on average EEG values
- Insighting based on data processed output

## Model Training
The project utilizes a Random Forest classifier for training on the EEG data. Hyperparameter tuning is performed using GridSearchCV to optimize the model's performance.

## Evaluation
After training, the model is evaluated on a test set, with metrics such as accuracy, confusion matrix, and classification report provided.

## Feature Importance
Feature importance is analyzed to understand which EEG channels are most influential in predicting mental states. This analysis aids in refining the model and focusing on significant features.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Feel free to modify any parts of this template to better fit your project! If you have any specific aspects you'd like to include or change, just let me know!
