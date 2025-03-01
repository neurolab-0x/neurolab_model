# If your model is in a separate file
import sys
sys.path.append('/content/drive/MyDrive/eeg-ds')
from models import model

model = model()
model.load_weights('./processed/trained_model.h5')  # If using pre-trained weights
