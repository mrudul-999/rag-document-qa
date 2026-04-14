# Use official Python 3.9 image to match your environment
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install basic system toolchains needed for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies first
# (This caches the pip install step so building is faster in the future)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose both backend and frontend default ports
EXPOSE 8000
EXPOSE 8501
