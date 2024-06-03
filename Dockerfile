FROM pytorch/torchserve:0.11.0-gpu as builder

WORKDIR /app

COPY . ./

RUN apt-get update && \
    xargs -a packages.txt apt-get install -y && \
    apt-get clean

RUN pip install -r requirements.docker.txt -i https://pypi.tuna.tsinghua.edu.cn/simple