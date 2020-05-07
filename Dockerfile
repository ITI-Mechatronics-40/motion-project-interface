FROM ubuntu:18.04

# ignore dialogue
ENV DEBIAN_FRONTEND=noninteractive

# upgrade env
RUN apt update
RUN apt upgrade -y

# install python3 and pip
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install python3-dev -y
RUN pip3 install --upgrade pip

#upgrade ffmpeg for aiortc av dependency
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt-get update && apt-get upgrade -y

#install aiortc dependencies
RUN apt-get install -y \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    pkg-config \
    python3-opencv \
    git

#get python3
RUN apt-get install -y \
    python3-pip

#install aiortc and other python libs
RUN apt-get install -y \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    pkg-config \
    python3-opencv \
    git

#get python3
RUN apt-get install -y \
    python3-pip

#install aiortc and other python libs
RUN pip3 install \
    aiortc \
    aiohttp \
    requests

WORKDIR /home

RUN git clone https://github.com/ITI-Mechatronics-40/motion-project-interface.git
