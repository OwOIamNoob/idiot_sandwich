ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.02-py3
FROM python:3.10

# Cài đặt libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx


# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements_for_docker.txt .

RUN pip install -r requirements_for_docker.txt

# Copy the rest of the files
COPY . .


# Run bash by default
CMD ["python", "app.py"]
