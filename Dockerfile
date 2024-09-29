# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to the world outside this container
EXPOSE 8080

# Set the command to run the inference script
CMD ["python", "task4.py"]
