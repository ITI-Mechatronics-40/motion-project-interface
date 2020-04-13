# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu

#application port
EXPOSE 5000

#Download pacjages
RUN apt update -- && apt install -y \
    python \
    git \
    python3-pip

# Disable Running performance tests to find the best convolution algorithm for face detection
ENV MXNET_CUDNN_AUTOTUNE_DEFAULT 0

# Install pip requirements
ADD requirements.txt .
RUN pip3 install -r requirements.txt
RUN apt update && apt install -y libsm6 libxext6 libxrender1 
RUN apt install -y python3-setuptools
RUN pip3 install Cython

COPY . /root/

WORKDIR /root/
RUN cd insightface/RetinaFace/rcnn/cython ;  python3 setup.py build_ext --inplace
ENTRYPOINT python3 app.py
# ENTRYPOINT /bin/bash
