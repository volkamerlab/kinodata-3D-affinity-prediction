# Use an NVIDIA PyTorch image as the base image
ARG PYTORCH_VERSION
ARG CUDA_NAME
ARG CUDA_SHORT_NAME
ARG CUDNN_VERSION

FROM pytorch/pytorch:${PYTORCH_VERSION}-${CUDA_NAME}-cudnn${CUDNN_VERSION}-runtime
# Set the working directory
WORKDIR /workspace

RUN python -c "print(__import__('platform').python_version())"
RUN echo ${PYTORCH_VERSION}
RUN echo ${CUDA_NAME}
RUN echo ${CUDA_SHORT_NAME}

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
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu121.html
RUN pip install torch_geometric
