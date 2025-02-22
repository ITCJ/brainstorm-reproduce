#!/usr/bin/env bash
# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BRT_DIR=$(cd "${script_dir}/../../" && pwd)

export BRT_CACHE_PATH=$BRT_DIR/.cache
export BRT_CAPTURE_STATS=False # should be False for brt_dist or tutel
export MOE_LAYER_VENDOR=tutel #_dist #_dist # tutel, brt, or brt_dist

PROC=1
GPU_NUM=$PROC
EXPERT_NUM=$((16 / GPU_NUM))

torchrun --nproc_per_node="$PROC" \
    --nnode=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=6500 \
    benchmark.py --cfg configs/"${EXPERT_NUM}"expert_"${GPU_NUM}"GPU.yaml \
    --batch-size 256 --data-path "${BRT_CACHE_PATH}"/datasets/imagenet22k --output ./results/MoE/ \
    --eval --resume "${BRT_CACHE_PATH}"/ckpt/swin_moe/small_swin_moe_32GPU_16expert/model.pth \
    --gather-ckpt
