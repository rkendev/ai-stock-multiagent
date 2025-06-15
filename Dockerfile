FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# Install system deps and Python 3.10
RUN apt-get update \
 && apt-get install -y \
      build-essential \
      git \
      python3 \
      python3-venv \
      python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Create virtualenv and install Python deps
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip

COPY requirements.txt /tmp/
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"
