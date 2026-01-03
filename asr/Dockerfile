FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel

WORKDIR /app
    
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
