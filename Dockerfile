# Use a specific Python version for reproducibility
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyMuPDF and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the application source code into the container
COPY src/ /app/src/
# COPY main.py .

# Create directories for input and output files
RUN mkdir -p input output

# Make the main script executable
# RUN chmod +x main.py

# Set the default command to run the main script
CMD ["python", "src/main.py"]