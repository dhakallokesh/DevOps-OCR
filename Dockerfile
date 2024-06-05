FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libpango1.0-0 \  
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the entrypoint or command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]