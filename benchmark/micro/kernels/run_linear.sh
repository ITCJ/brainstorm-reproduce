#!/bin/zsh

export BRT_CACHE_PATH=/root/siton-data-guoguodata/tcj/brainstorm_project/brainstorm/.cache

# python3 -u linear_3078_768.py 2>&1 | tee "output_$(date +%Y%m%d_%H%M%S).log"
python3 -u linear_4096_16384.py 2>&1 | tee "log/output_$(date +%Y%m%d_%H%M%S).log"
# python3 -u layernorm.py  2>&1 | tee "log/layernorm_output_$(date +%Y%m%d_%H%M%S).log"