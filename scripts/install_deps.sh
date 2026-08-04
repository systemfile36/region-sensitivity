#!/usr/bin/env bash
# -e: if any of the command lines returns non-zero, the script will terminate
# -u: raise error when using undefined variable
# -x: trace commands 
# -o pipefail: if all commands return zero, the script will return zero, otherwise, return non-zero value
set -euxo pipefail

# Prevent interactive prompt when install dependencies
# Necessary for Container/CI build environment
export DEBIAN_FRONTEND=noninteractive

apt-get update

# Base tools
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  wget \
  unzip \
  build-essential \
  pkg-config \

# Runtime deps commonly needed by opencv-python-headless / matplotlib
apt-get install -y --no-install-recommends \
  libglib2.0-0 \
  libgl1-mesa-glx \

# Clean cache and metadata
apt-get clean
rm -rf /var/lib/apt/lists/*