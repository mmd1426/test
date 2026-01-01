FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY . .
