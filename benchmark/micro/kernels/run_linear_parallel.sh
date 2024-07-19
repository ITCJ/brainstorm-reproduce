#!/bin/zsh

export BRT_CACHE_PATH=/root/siton-data-guoguodata/tcj/brainstorm_project/brainstorm/.cache

# python3 -u linear_3078_768.py 2>&1 | tee "output_$(date +%Y%m%d_%H%M%S).log"

typeset -A gpu_all_bs
gpu_all_bs[0]="2,4"
gpu_all_bs[1]="8,16"
gpu_all_bs[2]="32,64"
gpu_all_bs[3]="128"
gpu_all_bs[4]="224"
gpu_all_bs[5]="320"
gpu_all_bs[6]="416"
gpu_all_bs[7]="512"

# 遍历每个GPU和对应的all_bs列表
for gpu_id in {0..7}; do
    # 设置CUDA_VISIBLE_DEVICES环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id

    # 获取当前GPU对应的all_bs列表
    all_bs_list=${gpu_all_bs[$gpu_id]}
    echo "$gpu_id _ $all_bs_list"

    # 运行Python脚本，并传递修改后的all_bs列表
    # python your_script.py "$all_bs_list"
    python3 -u linear_3078_768.py "$all_bs_list" 2>&1 | tee "output_${all_bs_list}_$(date +%Y%m%d_%H%M%S).log" &
done

wait