# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union

import torch.distributed as dist
import brt.runtime.distributed as brt_dist
import torch

# pylint: disable=no-name-in-module
from brt._C.router import (
    dispatch_with_indices_and_loads,
    combine_with_indices_and_loads,
)

# pylint: enable=no-name-in-module
from brt.router.fabric.base import register_fabric
from brt.router.fabric.fused import FusedCombineFabric, FusedDispatchFabric


@register_fabric("distributed_fused_dispatch")
class DistributedFusedDispatchFabric(FusedDispatchFabric):
    def __init__(
        self,
        flow_num: int,
        capacity_padding=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        locality_aware: bool = False,
        task_locality_aware: bool = False,
    ):
        self.locality_aware = locality_aware
        self.task_locality_aware = task_locality_aware
        super().__init__(
            flow_num=flow_num,
            capacity_padding=capacity_padding,
            route_logic=route_logic,
            transform=transform,
        )
        self.register_buffer("placement_indices", None)

    def forward(
        self,
        in_flow: torch.Tensor,
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[torch.Tensor]:
        capacity = loads.capacity

        if self.route_logics[0] == "1d":

            if self.transforms[0]:
                out_flow = dispatch_with_indices_and_loads(
                    in_flow,
                    route_indices,
                    loads,
                    gates=score,
                    max_path_padding=self.capacity_padding,
                    max_path_load=capacity,
                )[0]
            else:
                out_flow = dispatch_with_indices_and_loads(
                    in_flow,
                    route_indices,
                    loads,
                    max_path_padding=self.capacity_padding,
                    max_path_load=capacity,
                )[0]
            # print(out_flow)
        elif self.route_logics[0] == "2d":
            out_flow = dispatch_with_indices_and_loads(
                in_flow, route_indices, loads, is_1d_routing=False
            )[0]
        else:
            raise ValueError("route_logic must be 1d or 2d")

        a2a_resuslts = brt_dist.group_asymmetry_a2a(
            out_flow, loads, self.locality_aware
        )
        out_flow = a2a_resuslts[0]

        if self.locality_aware:
            reorder_indices = a2a_resuslts[2]
            origin_shape = out_flow.shape
            out_flow = out_flow.view(reorder_indices.shape[0], -1)
            out_flow = out_flow.index_select(0, reorder_indices)
            out_flow = out_flow.view(origin_shape)
            route_indices, loads, score = brt_dist.batched_exchange(
                [route_indices, loads, score], a2a_resuslts[2]
            )
            brt_dist.set_reorder_indices(a2a_resuslts[2])

        if self.task_locality_aware:
            out_loads = a2a_resuslts[1]
            world_size = dist.get_world_size()
            num_local_tasks = out_loads.size(0) // world_size
            useful_outflows = [[] for _ in range(num_local_tasks)]
            start_index = 0
            for i in range(world_size):
                for j in range(num_local_tasks):
                    useful_outflows[j].append(
                        out_flow[
                            start_index : start_index
                            + out_loads[i * num_local_tasks + j]
                        ]
                    )
                start_index += capacity

            out_flows = [
                torch.cat(useful_outflows[i], dim=0) for i in range(num_local_tasks)
            ]
            new_task_ids = [
                torch.empty(
                    out_flows[i].size(0), dtype=torch.int64, device=out_flow.device
                ).fill_(world_size * num_local_tasks + i)
                for i in range(num_local_tasks)
            ]
            out_flow = torch.cat(out_flows, dim=0)
            out_flow.score = torch.cat(new_task_ids, dim=0)
            return out_flow

        out_flow.route_indices = route_indices
        out_flow.in_loads = loads
        out_flow.out_loads = a2a_resuslts[1]
        out_flow.score = score

        return out_flow


@register_fabric("distributed_fused_combine")
class DistributedFusedCombineFabric(FusedCombineFabric):
    def __init__(
        self,
        flow_num,
        sparse,
        reduction,
        granularity_padding,
        locality_aware: bool = False,
        transform=True,
    ) -> None:
        assert granularity_padding == False
        self.locality_aware = locality_aware
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )
        self.transform = transform

    def forward(self, in_flow: torch.Tensor,) -> List[torch.Tensor]:
        route_indices = in_flow.route_indices
        in_loads = in_flow.in_loads
        out_loads = in_flow.out_loads
        score = in_flow.score
        # print(f"gather in loads: {out_loads}, out loads: {in_loads}")
        # print(f"in flow: {in_flow.sum(1)}")
        in_flow = brt_dist.size_known_group_asymmetry_a2a(in_flow, out_loads, in_loads)
        # print(f"gather out flow: {in_flow.sum(1)}")
        if self.transform:
            out_flow = combine_with_indices_and_loads(
                in_flow, route_indices, in_loads, auto_pad=True, gates=score
            )
        else:
            out_flow = combine_with_indices_and_loads(in_flow, route_indices, in_loads, None)
        out_flow.score = score
        return out_flow


@register_fabric("distributed_placement_dispatch")
class DistributedPlacementDispatchFabric(FusedDispatchFabric):
    def __init__(
        self,
        flow_num: int,
        capacity_padding=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        task_locality: bool = False,
    ):
        self.task_locality = task_locality
        super().__init__(
            flow_num=flow_num,
            capacity_padding=capacity_padding,
            route_logic=route_logic,
            transform=transform,
        )

    def forward(
        self,
        in_flow: torch.Tensor,
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[torch.Tensor]:
        out_flow = dispatch_with_indices_and_loads(in_flow, route_indices, loads)[0]
        out_flow, out_loads, in_loads = brt_dist.group_sparse_a2a(out_flow, loads)
        if self.task_locality:
            world_size = dist.get_world_size()
            world_rank = dist.get_rank()
            num_local_tasks = out_loads.size(0) // world_size
            task_ids = torch.empty(
                out_flow.size(0), dtype=torch.int64, device=out_flow.device
            )
            base_idx = 0
            for i in range(num_local_tasks):
                task_total_load = (
                    out_loads[i * world_size : (i + 1) * world_size].sum().item()
                )
                task_ids[base_idx : base_idx + task_total_load].fill_(
                    world_rank * num_local_tasks + i
                )
                base_idx += task_total_load
            out_flow.score = task_ids
            return out_flow

        out_flow.route_indices = route_indices
        out_flow.in_loads = in_loads
        out_flow.out_loads = out_loads
        out_flow.score = score

        return out_flow


@register_fabric("distributed_placement_combine")
class DistributedPlacementCombineFabric(FusedCombineFabric):
    def __init__(
        self, flow_num, sparse, reduction, granularity_padding, transform=True,
    ) -> None:
        assert granularity_padding == False
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )
        self.transform = transform

    def forward(self, in_flow: torch.Tensor,) -> List[torch.Tensor]:
        route_indices = in_flow.route_indices
        in_loads = in_flow.in_loads
        out_loads = in_flow.out_loads
        score = in_flow.score
        in_flow = brt_dist.size_known_group_sparse_a2a(in_flow, out_loads, in_loads)
        out_flow = combine_with_indices_and_loads(in_flow, route_indices, in_loads, None)
        out_flow.score = score
        return out_flow
