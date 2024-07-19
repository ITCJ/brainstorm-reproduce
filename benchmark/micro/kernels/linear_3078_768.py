# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import sys
import torch

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
    [768, 3072],
    # [3072, 768]
]


for bs in all_bs:

    for in_features, out_features in in_out_features:

        input_infos = {"input_0": (bs, in_features)}

        output_infos = {"output_0": (bs, out_features)}

        parameters = {
            "in_features": in_features,
            "out_features": out_features,
        }

        kernel_name = f"Linear_{bs}_{in_features}_{out_features}"

        linear = torch.nn.Linear(in_features, out_features, bias=False).eval()

        tvm_tuner = TVMTuner()

        tvm_tuner.import_pt_netlet(
            "Linear",
            "forward",
            linear,
            input_infos,
            output_infos,
            parameters,
            # log_fname,
        )

        print(f"#### # Linerar {bs} {in_features} {out_features}")

        if tvm_tuner.tune_log_file.exists():
            print(tvm_tuner.tune_log_file)
            with open(str(tvm_tuner.tune_log_file)) as f:
                num_trials = len(f.readlines())
                print(tvm_tuner.tune_log_file)
            if num_trials < 2000:
                print("#### Find incomplete record, continue")
                tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
                tvm_tuner.tune_netlet()
                tvm_tuner.insert_netlet_to_storage()
            else:
                print("#### Find tuned kernel, pass")
                #手动读档重新生成代码写入db
                tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
                tvm_tuner.insert_netlet_to_storage()
        else:
            print("#### Start tuning kernel")
            tvm_tuner.tune_netlet()
            tvm_tuner.insert_netlet_to_storage()
        

        
        # #TCJ tune rank
        linear.cuda()

        for rank in range(1, 11):
            # tvm_tuner.task_scheduler.load_log_file = str(tvm_tuner.tune_log_file)
            # tvm_tuner.insert_netlet_to_storage(rank=rank)
            x = torch.randn((bs, in_features)).cuda()
            y = torch.randn((bs, out_features)).cuda()
            linear_kernel = make_jit_kernel(
                modules = linear,
                sample_inputs=x,
                rank = rank
            )
            
            # print(f"linear_kernel{linear_kernel}")
            # # 检查和打印参数类型
            # print(f"Type of x: {type(x)}")
            # print(f"Type of weight: {type(linear.weight)}")
            # print(f"Type of y: {type(y)}")
            # print(f"Type of bias: {type(linear.bias)}")

            time = Timer(
                stmt="y = torch.empty(oshape).cuda(); model(x, weight, y)",  # 测量的语句
                setup="import torch",   #前置代码
                globals={"model":linear_kernel, "x": x, "y": y, "weight": linear.weight, "oshape": (bs, out_features)} 
            ).timeit(1000).mean * 1e6

            print(f"{rank = }, {time}")
