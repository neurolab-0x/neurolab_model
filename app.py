from flask import Flask, request
import requests

app = Flask(__name__)

# Later we will be able to configure pre-processing and processing of EEG data routes from this server after I finish setting up the processing algorithms


if __name__ == "__main__":
  app.run(debug=True)
