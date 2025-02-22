# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import torch

def throttle_hotmask(
    hotmask: torch.Tensor,
    prefix: torch.Tensor,
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def convert_index_format(
    origin_indices: torch.Tensor,
    loads: torch.Tensor,
    is_to_tag: bool,  # 0 for src_index or 1 for dst_index
) -> torch.Tensor: ...
def generate_indices_and_loads(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor = None,
    capacity_padding: bool = False,
    path_wise_padding: bool = False,
    is_tag_index=False,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def dispatch_with_indices_and_loads(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    gates: torch.Tensor = None,
    tag_generating: bool = False,
    tags: torch.Tensor = None,
    max_path_padding: bool = False,
    max_path_load=0,
    is_1d_routing: bool = True,
    is_tag_index: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...
def split_fused_cells_to_paths(
    in_data: torch.Tensor,
    loads: torch.Tensor,
    max_path_padding: bool = False,
    is_load_split: bool = False,
    is_tag_split: bool = False,
    tags: torch.Tensor = None,
) -> Union[
    Tuple[List[torch.Tensor]],
    Tuple[List[torch.Tensor], List[torch.Tensor]],
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
]: ...
def fuse_split_cells_from_paths(
    in_data: List[torch.Tensor],
    is_load_fuse: bool = False,
    is_tag_fuse: bool = False,
    loads: List[torch.Tensor] = None,
    tags: List[torch.Tensor] = None,
) -> Union[
    Tuple[torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]: ...
def combine_with_indices_and_loads(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    gates: torch.Tensor = None,
    out_data: torch.Tensor = None,
    max_path_padding: bool = False,
    ever_padded=True,
    is_tag_index: bool = False,
    tags: torch.Tensor = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...
