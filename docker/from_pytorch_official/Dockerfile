# Use an NVIDIA PyTorch image as the base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /workspace

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install system deps
# g++: torch cluster, scatter
# libx*: rdkit pandastools
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
	g++ \
	libxrender1 \
	libxext6 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt
