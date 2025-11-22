FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the code
COPY . /code

# Create directory for models and set permissions
RUN mkdir -p /code/models && chmod 777 /code/models

# Set environment variables
ENV HOME=/code

# Run the download script and then the app on port 7860 (HF default)
CMD ["/bin/bash", "-c", "python download_model.py && gunicorn -b 0.0.0.0:7860 app:app --timeout 120"]
