FROM pytorch/torchserve:0.11.0-gpu as builder

WORKDIR /app

COPY . ./

USER root

RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update -y --allow-unauthenticated --fix-missing && \
    add-apt-repository ppa:savoury1/ffmpeg4 -y && \
    apt-get update -y && \
    apt-get update && 	xargs -r -a packages.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean 

RUN pip install -r requirements.docker.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
