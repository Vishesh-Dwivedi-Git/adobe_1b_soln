# Use a specific, stable version of Python on a slim base image
# Explicitly set the platform to linux/amd64 for compatibility as required
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for PDF processing and NLP
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Download spaCy model (keeping it lightweight)
RUN python -m spacy download en_core_web_sm

# Copy the source code
COPY ./src .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Specify the command to run when the container starts
CMD ["python", "main.py"]