FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies including VNC and noVNC
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-wxgtk4.0 \
    python3-matplotlib \
    python3-lxml \
    python3-pygame \
    libxml2-dev \
    libxslt1-dev \
    build-essential \
    ccache \
    g++ \
    gawk \
    make \
    wget \
    libtool \
    automake \
    autoconf \
    libexpat1-dev \
    vim \
    screen \
    # VNC and noVNC dependencies
    x11vnc \
    xvfb \
    fluxbox \
    novnc \
    websockify \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /root

# Clone ArduPilot repository with shallow clone (depth 1) to save space and time
RUN git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/ArduPilot/ardupilot.git

# Set ArduPilot directory as working directory
WORKDIR /root/ardupilot

# Install minimal SITL prerequisites manually
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    pymavlink \
    MAVProxy \
    pexpect \
    future \
    empy \
    pyserial

# Add ardupilot tools to PATH
ENV PATH="/root/ardupilot/Tools/autotest:${PATH}"

# VNC configuration
ENV DISPLAY=:0
ENV VNC_PORT=5900
ENV NOVNC_PORT=6080
ENV RESOLUTION=1920x1080

# Create supervisor config for managing services
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
# 5760-5763 - MAVLink connections
# 14550-14551 - MAVLink UDP
# 6080 - noVNC web interface
EXPOSE 5760 5762 5763 14550 14551 6080

# Set the working directory
WORKDIR /root/ardupilot

# Default command - Start all services with supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
