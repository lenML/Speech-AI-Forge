FROM fabridamicelli/torchserve:0.12.0-gpu-python3.10

# Set working directory
WORKDIR /app

# Use root to install necessary software
USER root

# Update APT sources to use a faster mirror (Tsinghua), install dependencies, and clean up to reduce image size
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list && \
    apt-get update -y --allow-unauthenticated --fix-missing && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ffmpeg \
        rubberband-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /etc/apt/sources.list.d/cuda.list

# Copy the requirements file for Python dependencies
COPY requirements.txt ./requirements.txt

# Install Python dependencies using Tsinghua's PyPI mirror and clean up cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Set the default command (modify as needed based on your application)
CMD ["bash"]
