# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.router.protocol.base import ProtocolBase, register_protocol
from brt.runtime import log

__all__ = ["SwitchTop1Protocol"]

logger = log.get_logger(__file__)

#TCJ SwitchTop1Protocol
@register_protocol("switch_top1")
class SwitchTop1Protocol(ProtocolBase):
    def __init__(
        self,
    ):
        """Top-K protocol

        Args:
            top_k (int, optional): k for top selecting. Defaults to 1.
            supported_capacities (optional): _description_. Defaults to None.
            index_format (str, optional): index tensors according to destination or source. Defaults to "src_index".
            index_gen_opt (bool, optional): whether use optimized GPU kernel. Defaults to True.
        """
        super().__init__()

    def make_route_decision(self, score):
        hot_mask = score.reshape(-1, score.shape[-1]).to(torch.int32, non_blocking=True)
        return hot_mask

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities
