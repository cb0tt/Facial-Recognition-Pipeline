FROM tensorflow/tensorflow:1.15.5-py3

# system deps for dlib/opencv
RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-6 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
