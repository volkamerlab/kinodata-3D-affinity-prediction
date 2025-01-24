# Use an NVIDIA PyTorch image as the base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /workspace

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Specify the entrypoint or command
# ENTRYPOINT ["/bin/bash"]

# Note: You can customize the entrypoint or command based on your application needs

