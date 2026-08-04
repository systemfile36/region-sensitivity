# For devlopment environment 
# Base image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /

# Install system dependencies from shell script
COPY scripts/install_deps.sh install_deps.sh
RUN bash ./install_deps.sh

# Install python dependencies from txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt