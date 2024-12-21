FROM openvino/ubuntu24_runtime:latest

SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV CUDNN_VERSION=8.1.1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    DEBIAN_FRONTEND=noninteractive

USER root

# Install system dependencies in logical groups
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic build tools
    build-essential \
    cmake \
    gcc \
    git \
    pkg-config \
    # Development tools
    autoconf \
    automake \
    libtool \
    # System utilities
    apt-utils \
    ca-certificates \
    curl \
    sudo \
    vim \
    wget \
    # Compression tools
    unzip \
    xz-utils \
    # Media related
    yasm \
    libjpeg-dev \
    libavfilter-dev \
    libavutil-dev \
    # Network and security
    libssl-dev \
    libcurl4-openssl-dev \
    libzmq3-dev \
    # Graphics and GUI
    libglu1-mesa-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    # Protobuf
    libprotobuf-dev \
    protobuf-compiler \
    # Additional development libraries
    autopoint \
    bison \
    flex \
    gtk-doc-tools \
    libglib2.0-dev \
    liborc-0.4-0 \
    liborc-0.4-dev \
    swig \
    nlohmann-json3-dev \
    # Python
    python3-pip \
    software-properties-common \
    # Clean up
    && apt-get clean


RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt install -y gcc-9 g++-9 --no-install-recommends

RUN cd /tmp/ \
    && wget https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3.tar.gz \
    && tar -xvf cmake-3.23.3.tar.gz \
    && cd cmake-3.23.3 \
    && ./configure && make -j4 && make install -j4 \
    && rm -rf /tmp/cmake-3.23.3*

RUN pip3 install meson ninja



RUN cd /tmp/ \
    && wget https://gstreamer.freedesktop.org/src/gstreamer/gstreamer-1.22.2.tar.xz \
    && wget https://gstreamer.freedesktop.org/src/gst-plugins-base/gst-plugins-base-1.22.2.tar.xz \
    && wget https://gstreamer.freedesktop.org/src/gst-plugins-good/gst-plugins-good-1.22.2.tar.xz \
    && wget https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-1.22.2.tar.xz \
    && wget https://gstreamer.freedesktop.org/src/gst-libav/gst-libav-1.22.2.tar.xz \
    && tar -xvf gstreamer-1.22.2.tar.xz && cd gstreamer-1.22.2 && meson build --prefix=/usr && ninja -C build install && cd .. \
    && tar -xvf gst-plugins-base-1.22.2.tar.xz && cd gst-plugins-base-1.22.2 && meson build --prefix=/usr && ninja -C build install && cd .. \
    && tar -xvf gst-plugins-good-1.22.2.tar.xz && cd gst-plugins-good-1.22.2 && meson build --prefix=/usr && ninja -C build install && cd .. \
    && tar -xvf gst-plugins-bad-1.22.2.tar.xz && cd gst-plugins-bad-1.22.2 && meson build --prefix=/usr && ninja -C build install && cd .. \
    && tar -xvf gst-libav-1.22.2.tar.xz && cd gst-libav-1.22.2 && meson build --prefix=/usr && ninja -C build install && cd .. \
    && rm -rf *

RUN cd /tmp/ && git clone --recursive https://github.com/whoshuu/cpr.git && cd cpr && mkdir build && cd build && cmake .. && make -j4 && make install \
    && cd /tmp/ && rm -rf cpr
RUN wget https://github.com/opencv/opencv/archive/4.7.0.tar.gz && tar -xvf *.tar.gz && mkdir opencv-4.7.0/build \
    && cd opencv-4.7.0/build && wget https://github.com/opencv/opencv_contrib/archive/4.7.0.tar.gz && tar -xvf *.tar.gz \
    && cmake -D WITH_GTK=ON -D OPENCV_EXTRA_MODULES_PATH=opencv_contrib-4.7.0/modules .. \
    && make -j4 && make install && rm -rf /tmp/opencv-4.7.0* /tmp/*.tar.gz
