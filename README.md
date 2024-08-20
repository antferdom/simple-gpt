# Simple GPT Overview
Minimal industrial level implementation (multi-node, DDP, FSDP) GPT training.

# Running with Docker
Using the official Ubuntu Docker image with version tag `22.04` (see [ubuntu Docker Official Image](https://hub.docker.com/_/ubuntu/tags)), which we can pull directly: 
```bash
docker pull ubuntu:22.04
```

```bash
docker run -it -d --net host --runtime=nvidia --gpus all --name simple_ubuntu ubuntu:22.04  bash
docker exec -it simple_ubuntu /bin/bash
```
For limiting the available visible CUDA devices within the container we can modify the flag `--gpus` to just one `1` or specify the device accordingly ([Specialized Configurations with Docker: GPU Enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration)). For example, using one single GPU in the container:
```bash
docker run -it -d --net host --runtime=nvidia --gpus 1 --name simple_ubuntu ubuntu:22.04  bash
```