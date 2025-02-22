import itertools
from typing import List, Tuple, Union, Literal, Callable, Any

import torch
from torch import autograd
from torch.overrides import (
    handle_torch_function,
    wrap_torch_function,
    has_torch_function,
)

from brt.jit.codegen.hetero_fused import HeteroFusedKernel
from brt.jit.modules.base import FuseModuleInputType
from brt.jit.modules.fused import FusedModule


class HeteroFusedModule(FusedModule):
    def _make_global_kernel(
        self,
        sample_inputs: FuseModuleInputType,
        method: str = "forward",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> HeteroFusedKernel:
        assert self.num_submodule == len(
            sample_inputs
        ), "modules and sample_inputs must have the same length"
        if isinstance(rank, int):
            rank = [rank] * self.num_submodule
        candidates = []
        for jsm, inp, rk in zip(self.jit_submodules, sample_inputs, rank):
            module_kernel = jsm._make_global_kernel(inp, method, objective_func, rk)
            candidates.append(module_kernel)
        fused_kernel = HeteroFusedKernel(candidates)
        return fused_kernel

    def make_function(
        self,
        sample_inputs: FuseModuleInputType,
        mode: Literal["eval", "train"] = "eval",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> autograd.Function:
        jit_kernel = self.make_kernel(
            sample_inputs=sample_inputs,
            method="forward",
            objective_func=objective_func,
            rank=rank,
        )
        (
            input_arg_num,
            total_arg_num,
            input_arg_indices,
            output_arg_indices,
        ) = self._extract_arg_infos("forward")
        out_data = [
            torch.empty(shp).to("cuda")
            for shp in self._get_output_shape("forward", sample_inputs)
        ]

        class JitFunction(autograd.Function):
            @staticmethod
            @wrap_torch_function(lambda *x: x)
            def forward(ctx: Any, *inputs, active_blocks):
                inputs = list(inputs)
                for i, out_index in enumerate(output_arg_indices):
                    inputs.insert(out_index, out_data[i])
                jit_kernel(*inputs, active_blocks=active_blocks)
                outputs = [inputs[i] for i in output_arg_indices]
                return tuple(outputs)

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:
                raise NotImplementedError

        return JitFunction

    def _extract_shared_arg_infos(
        self,
        method: str,
        sample_inputs: FuseModuleInputType,
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    def _extract_arg_infos(
        self,
        method: str,
    ) -> Tuple[int, int, List[int], List[int]]:
        input_arg_num = 0
        total_arg_num = 0
        input_arg_indices = []
        output_arg_indices = []
        for jsm in self.jit_submodules:
            (
                sub_input_arg_num,
                sub_total_arg_num,
                sub_input_arg_indices,
                sub_output_arg_indices,
            ) = jsm._extract_arg_infos(method)
            output_arg_indices.extend(
                [i + total_arg_num for i in sub_output_arg_indices]
            )
            input_arg_indices.extend([i + total_arg_num for i in sub_input_arg_indices])
            total_arg_num += sub_total_arg_num
            input_arg_num += sub_input_arg_num
        return (
            input_arg_num,
            total_arg_num,
            input_arg_indices,
            output_arg_indices,
        )

    def _get_output_shape(
        self, method: str, sample_inputs: FuseModuleInputType
    ) -> List[torch.Size]:
        return list(
            itertools.chain.from_iterable(
                jsm._get_output_shape(method, sample_input)
                for jsm, sample_input in zip(self.jit_submodules, sample_inputs)
            )
        )

    @property
    def module_name(self) -> str:
        return f"HeteroFused_{self.num_submodule}_" + "_".join(
            [jsm.module_name for jsm in self.jit_submodules]
        )
