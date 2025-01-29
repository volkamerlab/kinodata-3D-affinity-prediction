docker build --no-cache --build-arg PYTORCH_VERSION=2.3.0 --build-arg CUDA_NAME=cuda12.1 --build-arg CUDA_SHORT_NAME=cu121 -t gjoschka/kinodata3d-b0:2.3.0-cuda12.1 . &> build.log
