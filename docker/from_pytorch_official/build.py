import typer


def main(
    pytorch_version: str = "2.5.1",
    cuda_name: str = "cuda12.1",
    image_name: str = "gjoschka/kinodata3d",
    cudnn_version: str = "9",
    extra_tag: str = "",
):
    cuda_short_name = cuda_name.replace("cuda", "cu").replace(".", "")
    command = [
        "docker build",
        "--no-cache",
        f"--build-arg PYTORCH_VERSION={pytorch_version}",
        f"--build-arg CUDA_NAME={cuda_name}",
        f"--build-arg CUDA_SHORT_NAME={cuda_short_name}",
        f"--build-arg CUDNN_VERSION={cudnn_version}",
        f"-t {image_name}-pt{pytorch_version}-{cuda_short_name}-cudnn{cudnn_version}{extra_tag}",
        ".",
        "&> build.log",
    ]
    print(" ".join(command))


if __name__ == "__main__":
    typer.run(main)
