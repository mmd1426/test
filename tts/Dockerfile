FROM python:3.11.14

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg espeak-ng

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .