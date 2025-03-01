from preprocessing.labeling import label_eeg_states
from preprocessing.load_data import load_data

def main():
  data_path = "./data/raw/s01.csv"  # Adjust based on actual file
  data = load_data(data_path)
  data = label_eeg_states(data)
  print(data.columns)

main()