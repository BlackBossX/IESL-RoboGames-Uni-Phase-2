FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (if any additional ones are needed)
# RUN pip3 install --no-cache-dir <package-name>

# Set working directory
WORKDIR /app

# Copy controller files
COPY ./Webots/controller/ /app/

# Set PYTHONPATH to include Webots controller libraries (will be mounted)

ENV WEBOTS_HOME=/usr/local/webots
ENV PYTHONPATH="${WEBOTS_HOME}/lib/controller/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}"
ENV USER=${USER}

# Expose camera port
EXPOSE 5599

# Default command
CMD ["python3", "ardupilot_vehicle_controller.py", \
     "--motors", "m1_motor, m2_motor, m3_motor, m4_motor", \
     "--camera", "camera", \
     "--camera-port", "5599"]
