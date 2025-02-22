# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import torch.nn as nn
from brt.runtime.memory_planner import (
    MemoryPlanContext,
    MemoryPlanner,
    pin_memory,
    load_module,
    unload_module,
)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.linear(x)
        x = self.conv(x)
        return x


class MemoryPlanTest(unittest.TestCase):
    def test_weight_loader(self):
        simple_net = SimpleNet()
        in_data = torch.randn(1, 3, 10, 10)
        origin_out_data = simple_net(in_data)
        # init the weight loader
        MemoryPlanContext.init()

        pinned_simple_net = pin_memory(simple_net)
        pinned_out_data = pinned_simple_net(in_data)

        # first load and unload
        new_cuda_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        cuda_simple_net = load_module(pinned_simple_net)

        with torch.cuda.stream(new_cuda_stream):
            cuda_out_data = cuda_simple_net(in_data.cuda(non_blocking=True))

        self.assertTrue(torch.allclose(origin_out_data, cuda_out_data.cpu()))

        unload_simple_net = unload_module(cuda_simple_net)
        unload_out_data = unload_simple_net(in_data)
        # second load and unload
        cuda_simple_net = load_module(unload_simple_net)
        cuda_out_data = cuda_simple_net(in_data.cuda(non_blocking=True))
        unload_simple_net = unload_module(cuda_simple_net)
        unload_out_data = unload_simple_net(in_data)

        self.assertTrue(torch.allclose(origin_out_data, pinned_out_data))
        self.assertTrue(torch.allclose(origin_out_data, cuda_out_data.cpu()))
        self.assertTrue(torch.allclose(origin_out_data, unload_out_data))


if __name__ == "__main__":
    unittest.main()
