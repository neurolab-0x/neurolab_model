# Use the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install any project dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the Python application
CMD ["python", "main.py"]
