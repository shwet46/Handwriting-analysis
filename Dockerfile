# Use official TensorFlow base image with Python
FROM tensorflow/tensorflow:2.15.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy requirements and install Python packages
COPY requirements.txt .

# Install requirements and opencv-python-headless
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir opencv-python-headless

# Copy the rest of your project files
COPY . .


CMD ["/bin/bash"]