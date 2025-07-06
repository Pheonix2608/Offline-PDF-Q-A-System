# Use official Python image
FROM python:3.10-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    poppler-utils \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "ui_gradio.py"]


# Note: Ensure that the `requirements.txt` file is present in the same directory as this Dockerfile.
# This Dockerfile sets up a Python environment with all necessary dependencies to run the PDF Q&A