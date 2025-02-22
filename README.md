# Brainstorm

**Works like brain, as fast as storm**

Brainstorm is a compiler with profile-guided optimization for dynamic neural networks.

# Hardware and Software Requirements

- Software Requirements
  - Larger than CUDA 11.3
  - cuDNN 8.6 for NVIDIA A100 GPU and cuDNN 8.2 for NVIDIA V100 GPUs
  - [Pytorch 1.12.1 with CUDA 11.3](https://pytorch.org/get-started/previous-versions/#v1121)

# Setting up the Environment

We provide two options to set up the environment: bare metal server and docker container.

## Installation on Bare Metal Server

We provide an one-click script to setup the environment on bare metal server. The script will install the required packages and Brainstorm itself.

```bash
bash scripts/setup_bare.sh
```

## Installation with Docker Container

### Building the Docker Image

We also provide a docker image to setup the environment. The docker image can be built by the following command:

```bash
python scripts/docker_gh_build.py --type latest
```

### Starting the Docker Container

The docker image can be run by the following command:

```bash
docker run -name brt_ae -ti brt:latest /bin/bash
```

We also provide an online image on github registry. The image can be run by the following command:

```bash
docker run --name brt_ae -ti ghcr.io/raphael-hao/brt:latest /bin/bash
```

# Preparing the Dataset, Checkpoints, and Kernel Database:

```bash
bash scripts/init_dev.sh
```

# Playing with Brainstorm

**TBD**

# Reference

Please cite Brainstorm in your publications if it helps your research:

```
@inproceedings {brainstorm,
author = {Weihao Cui and Zhenhua Han and Lingji Ouyang and Yichuan Wang and Ningxin Zheng and Lingxiao Ma and Yuqing Yang and Fan Yang and Jilong Xue and Lili Qiu and Lidong Zhou and Quan Chen and Haisheng Tan and Minyi Guo},
title = {Optimizing Dynamic Neural Networks with Brainstorm},
booktitle = {17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)},
year = {2023},
isbn = {978-1-939133-34-2},
address = {Boston, MA},
pages = {797--815},
url = {https://www.usenix.org/conference/osdi23/presentation/cui},
publisher = {USENIX Association},
month = jul,
}
```
