# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import itertools
import pathlib
from collections import OrderedDict
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from brt.router import is_router
from brt.runtime import BRT_CACHE_PATH, log

logger = log.get_logger(__file__)


"""
Currently all placement-related optimizations are implemented for Swin-MoE.
In the future, we have to generalize the placement-related optimizations to
all dynamic models.
"""


def sorted_k_even_partitions(seq, k, length):
    """Returns a list of all unique k-partitions of `seq`.

    Each partition is a list of parts, and each part is a tuple.

    The parts in each individual partition will be sorted in shortlex
    order (i.e., by length first, then lexicographically).

    The overall list of partitions will then be sorted by the length
    of their first part, the length of their second part, ...,
    the length of their last part, and then lexicographically.
    """
    n = len(seq)
    groups = []  # a list of lists, currently empty

    def generate_partitions(i):
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > (k - len(groups)) * length:
                for group in groups:
                    if len(group) < length:
                        group.append(seq[i])
                        yield from generate_partitions(i + 1)
                        group.pop()
                    # else:
                    #     group.append(seq[i])
                    #     yield from generate_partitions(i + 1)
                    #     group.pop()

            if len(groups) < k:
                groups.append([seq[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)

    # Sort the parts in each partition in shortlex order
    result = [sorted(ps, key=lambda p: (len(p), p)) for ps in result]
    # Sort partitions by the length of each part, then lexicographically.
    result = sorted(result, key=lambda ps: (*map(len, ps), ps))
    all_ordered_partitions = []

    for partition in result:
        ordered_partitions = list(itertools.permutations(partition))
        ordered_partitions = [list(p) for p in ordered_partitions]
        all_ordered_partitions.extend(ordered_partitions)

    return all_ordered_partitions


def dump_trace(mod: nn.Module):
    scatter_results = []
    for _m_name, m in mod.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            scatter_results.append(np.array(m.load_history, dtype=object))
    np.save("scatter_results.npy", scatter_results, allow_pickle=True)


def dump_decision(mod: nn.Module):
    scatter_results = []
    for _m_name, m in mod.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            scatter_results.append(
                np.array(m.fabric.cell_decision_history, dtype=object)
            )
    np.save("scatter_results.npy", scatter_results, allow_pickle=True)


def generate_experts_keys(experts_range: Dict[int, int]):
    experts_keys: List[Tuple[int, int]] = []
    for layer_id, block_num in experts_range.items():
        for block_id in range(1, block_num, 2):
            experts_keys.append((layer_id, block_id))
    return experts_keys


def placement_permute_generator(
    placement: List[List[int]],
) -> Iterator[List[List[int]]]:
    np.random.seed(0)
    used_placement = set()
    while True:
        permuted_placement = np.random.permutation(placement)
        tupled_placement = tuple([tuple(p) for p in permuted_placement])
        if tupled_placement in used_placement:
            continue
        used_placement.add(tupled_placement)
        yield permuted_placement


def permute_placement(placement: List[List[int]], seed=0):
    np.random.seed(seed)
    return np.random.permutation(placement)


def deterministic_rand_placement_generator(
    expert_num: int, world_size: int, seed=0
) -> Iterator[List[List[int]]]:
    assert expert_num % world_size == 0
    np.random.seed(seed)
    all_experts = list(range(expert_num))
    used_placement = set()
    while True:
        placement_indice = np.random.permutation(all_experts)
        placement = np.split(placement_indice, world_size)
        placement = [list(np.sort(p)) for p in placement]
        tupled_placement = tuple([tuple(p) for p in placement])
        if tupled_placement in used_placement:
            continue
        used_placement.add(tupled_placement)
        yield placement


def possible_placement_generator(expert_num: int, world_size: int):
    assert expert_num % world_size == 0
    all_experts = list(range(expert_num))
    possible_placement = sorted_k_even_partitions(
        all_experts, world_size, expert_num // world_size
    )
    return possible_placement


def load_searched_placement(
    config, which_one: str, moe_layer_start: int = None, moe_layer_end: int = None
) -> Dict[Tuple[int, int], List[List[int]]]:
    result_path = BRT_CACHE_PATH / "ckpt" / "swinv2_moe_small"
    world_size = dist.get_world_size()
    experts_range = {2: 18, 3: 2}
    experts_keys = generate_experts_keys(experts_range)
    capacity_factor = config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR
    assert which_one in ["best", "worst"]
    if moe_layer_start is None and moe_layer_end is None:
        searched_placement_file = (
            result_path / f"placement/{which_one}_{world_size}_placement.csv"
        )
        moe_layer_start = 0
        moe_layer_end = len(experts_keys) - 1
    else:
        searched_placement_file = (
            result_path
            / f"placement/{moe_layer_start}_{moe_layer_end}.{capacity_factor}.{which_one}_{world_size}_placement.csv"
        )
    logger.info(f"Loading searched placement from {searched_placement_file.as_posix()}")
    searched_placement_list = np.loadtxt(
        searched_placement_file, dtype=np.int32, delimiter=","
    )
    assert len(searched_placement_list) == moe_layer_end - moe_layer_start + 1
    searched_placement = {}
    for i in range(moe_layer_start, moe_layer_end + 1):
        placement = np.split(searched_placement_list[i], world_size)
        placement = [list(p) for p in placement]
        searched_placement[experts_keys[i]] = placement
    return searched_placement


def adaptive_micro_bench_load(
    model: nn.Module,
    new_placements: Dict[Tuple[int, int], List[List[int]]],
    checkpoint_file: str,
    global_expert_num: int = None,
):
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    _experts_keys, rank_placement, placement_indices = generate_rank_placement(
        world_rank, world_size, None, global_expert_num
    )
    for expert_key, placement in new_placements.items():
        rank_placement[expert_key] = list(placement[world_rank])
        placement_index = np.array(list(itertools.chain.from_iterable(placement)))
        placement_indices[expert_key] = torch.from_numpy(placement_index)
    adaptive_load_checkpoint(model, checkpoint_file, rank_placement, placement_indices)


def generate_rank_placement(
    world_rank: int,
    world_size: int,
    placement_file: str = None,
    global_expert_num: int = None,
):
    experts_range = {2: 18, 3: 2}
    experts_keys = generate_experts_keys(experts_range)

    assert placement_file is not None or global_expert_num is not None
    all_placement: List[np.ndarray] = []
    if placement_file is None:
        assert global_expert_num % world_size == 0
        local_expert_num = global_expert_num // world_size
        placement = np.repeat(np.arange(world_size), local_expert_num)
        all_placement = [placement for i in range(len(experts_keys))]
    else:
        all_placement = np.load(placement_file, allow_pickle=True)
    rank_placement: "OrderedDict[Tuple[int, int], List[int]]" = OrderedDict()
    placement_indices: "OrderedDict[Tuple[int, int], torch.Tensor]" = OrderedDict()
    for idx, placement in enumerate(all_placement):
        expert_key = experts_keys[idx]
        placement_index = []
        for rank_idx in range(world_size):
            placement_index.append(np.where(placement == rank_idx)[0])
            if rank_idx == world_rank:
                rank_placement[expert_key] = placement_index[-1].tolist()
        placement_index = np.concatenate(placement_index, axis=None)
        placement_indices[expert_key] = torch.from_numpy(placement_index)

    # if placement_file is None:
    #     placement_indices = None

    return experts_keys, rank_placement, placement_indices


def adaptive_load(
    model: nn.Module,
    checkpoint_file: str,
    enable_locality=False,
    placement_file: str = None,
    global_expert_num: int = None,
):
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    experts_keys, rank_placement, placement_indices = generate_rank_placement(
        world_rank, world_size, placement_file, global_expert_num
    )
    adaptive_load_checkpoint(model, checkpoint_file, rank_placement, placement_indices)
    # debug_helper(model, placement_indices)
    locality_enabled_router = {"scatter": [], "gather": []}
    if enable_locality:
        locality_enabled_router["scatter"].append(experts_keys[0])
        locality_enabled_router["gather"].append(experts_keys[-1])
        print(f"locality_enabled_router: {locality_enabled_router}")
    adaptive_set_locality(model, locality_enabled_router)


def adaptive_load_checkpoint(
    model: nn.Module,
    checkpoint_file: str,
    rank_placement: "OrderedDict[Tuple[int, int], List[int]]",
    placement_indices: "OrderedDict[Tuple[int, int], torch.Tensor]" = None,
):
    ckpt_filepath = pathlib.Path(checkpoint_file)
    checkpoint = torch.load(ckpt_filepath, map_location="cpu")
    all_expert_state_dict = {}
    state_dict = {}
    for k in checkpoint.keys():
        if "._moe_layer.experts." in k:
            k_var_id = k.split("._moe_layer.experts.0.")
            k_list = k_var_id[0].split(".")
            var_name, expert_id = k_var_id[1].split(".")
            expert_k = (int(k_list[1]), int(k_list[3]))
            expert_id = int(expert_id)
            if expert_k not in all_expert_state_dict:
                all_expert_state_dict[expert_k] = {}
            if expert_id not in all_expert_state_dict[expert_k]:
                all_expert_state_dict[expert_k][expert_id] = {}
            all_expert_state_dict[expert_k][expert_id][var_name] = checkpoint[k]
        else:
            state_dict[k] = checkpoint[k]

    for k in rank_placement.keys():
        tensors_to_concat = {}
        for expert_id in rank_placement[k]:
            tensor_entries = all_expert_state_dict[k][expert_id].keys()
            for entry in tensor_entries:
                full_entry_name = (
                    f"layers.{k[0]}.blocks.{k[1]}.mlp._moe_layer.experts.0.{entry}"
                )
                if full_entry_name not in tensors_to_concat:
                    tensors_to_concat[full_entry_name] = []
                tensors_to_concat[full_entry_name].append(
                    all_expert_state_dict[k][expert_id][entry]
                )
        for entry in tensors_to_concat.keys():
            if len(tensors_to_concat[entry]) == 1:
                state_dict[entry] = tensors_to_concat[entry][0]
            else:
                if "_bias" in entry:
                    state_dict[entry] = torch.stack(
                        [torch.unsqueeze(x, dim=0) for x in tensors_to_concat[entry]]
                    )
                else:
                    state_dict[entry] = torch.stack(tensors_to_concat[entry])

    if placement_indices is not None:
        for k in placement_indices.keys():
            # print(f"setting placement for {k} with {placement_indices[k]}")
            origin_wg_weight = state_dict[
                f"layers.{k[0]}.blocks.{k[1]}.mlp._moe_layer.gates.0.wg.weight"
            ]
            new_wg_weight = torch.index_select(
                origin_wg_weight, 0, placement_indices[k]
            )
            state_dict[
                f"layers.{k[0]}.blocks.{k[1]}.mlp._moe_layer.gates.0.wg.weight"
            ] = new_wg_weight

    model.load_state_dict(state_dict, strict=False)
    # logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()


def adaptive_set_locality(
    model: nn.Module, locality_enabled_router: Dict[str, List[Tuple[int, int]]]
):
    for _m_name, m in model.named_modules():
        if is_router(m):
            expert_key = tuple([int(_m_name.split(".")[i]) for i in [1, 3]])
            if (
                "scatter" in m._router_type
                and expert_key in locality_enabled_router["scatter"]
            ):
                print(
                    f"orginal locality for {_m_name}.fabric is {m.fabric.locality_aware}"
                )
                m.fabric.locality_aware = True
            if (
                "gather" in m._router_type
                and expert_key in locality_enabled_router["gather"]
            ):
                print(
                    f"orginal locality for {_m_name}.fabric is {m.fabric.locality_aware}"
                )
                m.fabric.locality_aware = True


def debug_helper(
    model: nn.Module,
    placement_indices: "OrderedDict[Tuple[int, int], torch.Tensor]" = None,
):
    for _m_name, m in model.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            expert_key = tuple([int(_m_name.split(".")[i]) for i in [1, 3]])
            print(f"setting debug info for {_m_name} with {expert_key}")
            m.fabric.expert_key = expert_key
