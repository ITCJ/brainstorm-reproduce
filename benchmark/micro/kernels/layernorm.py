# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import sys
import torch
from typing import List, Tuple, Union

from torch.utils.benchmark import Timer


from brt.jit import make_jit_kernel

from brt.jit.tvm import TVMTuner

from brt.jit.codegen import ModuleKernel

# all_bs = list(map(int, sys.argv[1].split(',')))
all_bs = [
    # 2,
    # 4,
    # 8,
    # 16,
    # 32,
    # 64,
    # 96,
    # 128,
    # 160,
    192,
    # 224,
    # 320,
    # 416,
    # 512
]


in_out_features = [
    [4096],
    # [16384],
]


for bs in all_bs:
    for in_features in in_out_features:
        input_infos = {"input_0": (bs, in_features[0])}
        output_infos = {"output_0": (bs, in_features[0])} # input output is the same

        parameters = {
            "normalized_shape": in_features[0],
        }

        kernel_name = f"LayerNorm_{bs}_{in_features[0]}"

        layernorm = torch.nn.LayerNorm(in_features[0], elementwise_affine=False).eval()

        tvm_tuner = TVMTuner()

        tvm_tuner.import_pt_netlet(
            "LayerNorm",
            "forward",
            layernorm,
            input_infos,
            output_infos,
            parameters,
        )

        print(f"#### LayerNorm {bs} {in_features[0]}")
        # if tvm_tuner.tune_log_file.exists():
        #     print(tvm_tuner.tune_log_file)
        #     with open(str(tvm_tuner.tune_log_file)) as f:
        #         num_trials = len(f.readlines())
        #         print(tvm_tuner.tune_log_file)
        #     if num_trials < 2000:
        #         print("#### Find incomplete record, continue")
        #         tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
        #         tvm_tuner.tune_netlet()
        #         tvm_tuner.insert_netlet_to_storage()
        #     else:
        #         print("#### Find tuned kernel, pass")
        #         tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
        #         tvm_tuner.tune_netlet()
        #         tvm_tuner.insert_netlet_to_storage()
        # else:
        #     print("#### Start tuning kernel")
        #     tvm_tuner.tune_netlet()
        #     tvm_tuner.insert_netlet_to_storage()
        

        # # #TCJ tune fuse 192,4096,16384 kernel
        # linear0 = torch.nn.Linear(in_features, out_features, bias=False).eval().cuda()
        # x0 = torch.randn((bs, in_features)).cuda()
        # y0 = torch.randn((bs, out_features)).cuda()

        # linear1 = torch.nn.Linear(in_features, out_features, bias=False).eval().cuda()
        # x1 = torch.randn((bs, in_features)).cuda()
        # y1 = torch.randn((bs, out_features)).cuda()

        # modulelist = torch.nn.ModuleList([linear0, linear1])

        # # print(f"type(x){type(x)}")
        # # print(f"type(y){type(y)}")
        # # print(f"type(linear.weight){type(linear.weight)}")

        # # print("----------------make_jit_kernel@linear_4096.py-----------------")
        # linear_kernel = make_jit_kernel(
        #     modules = modulelist,
        #     sample_inputs=[x0,x1],
        #     method="forward",
        #     opt_level="horiz_fuse",
        #     objective_func= "fastest",
        # )
        
        # # # print("----------------execute_kernel@linear_4096.py-----------------")
        # # linear_kernel(x0, linear0.weight, y0, x1, linear1.weight, y1)

        # time = Timer(
        #     setup="import torch",   #前置代码
        #     stmt="model(x0, weight0, y0, x1, weight1, y1)",  # 测量的语句
        #     globals={"model":linear_kernel, "x0": x0, "weight0": linear0.weight, "y0": y0, "x1":x1,  "weight1": linear1.weight, "y1":y1, } 
        # ).timeit(1000).mean * 1e6

        # print(f"eslaped time:{time}")

    
    #TCJ tune rank

    # linear.cuda()
    # # for rank in range(1, 11):
    # for rank in range(1, 2):
    #     # tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
    #     # tvm_tuner.insert_netlet_to_storage(rank=rank)
    #     x = torch.randn((bs, in_features)).cuda()
    #     y = torch.randn((bs, out_features)).cuda()
    #     linear_kernel = make_jit_kernel(
    #         modules = linear,
    #         sample_inputs=x,
    #         rank = rank
    #     )
        
    #     # print(f"linear_kernel{linear_kernel}")
    #     # # 检查和打印参数类型
    #     # print(f"Type of x: {type(x)}")
    #     # print(f"Type of weight: {type(linear.weight)}")
    #     # print(f"Type of y: {type(y)}")
    #     print(f"x.shape:{x.shape}")
    #     print(f"linear.weight.shape:{linear.weight.shape}")
    #     print(f"y.shape:{y.shape}")

    #     # linear_kernel(x, linear.weight, y)

    #     time = Timer(
    #         stmt="model(x, weight, y)",  # 测量的语句
    #         setup="import torch",   #前置代码
    #         globals={"model":linear_kernel, "x": x, "y": y, "weight": linear.weight} 
    #     ).timeit(1000).mean * 1e6

    #     print(f"{rank = }, {time}")
