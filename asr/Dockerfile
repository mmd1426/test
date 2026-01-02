FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel

WORKDIR /app
    
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .
