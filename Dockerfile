# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file to the working directory
COPY phishing_detector.pkl /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Copy the data file and inference code into the container
COPY Phishing_Legitimate_full.csv .
COPY task4.py .

# Set the command to run the inference script
CMD ["python", "task4.py"]