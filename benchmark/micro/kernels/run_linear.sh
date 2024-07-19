#!/bin/zsh

export BRT_CACHE_PATH=/root/siton-data-guoguodata/tcj/brainstorm_project/brainstorm/.cache

python3 -u linear_3078_768.py 2>&1 | tee "output_$(date +%Y%m%d_%H%M%S).log"