# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import csv

import torch.nn as nn
import numpy as np
from brt.router import GatherRouter
from brt import Annotator

__all__ = ["Backbone"]

from cell import Cell
from ops import Conv2dNormAct, ShapeSpec, kaiming_init_module


class Backbone(nn.Module):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    # the properties below are not used any more

    @property
    def out_features(self):
        """deprecated"""
        return self._out_features

    @property
    def out_feature_strides(self):
        """deprecated"""
        return {f: self._out_feature_strides[f] for f in self._out_features}

    @property
    def out_feature_channels(self):
        """deprecated"""
        return {f: self._out_feature_channels[f] for f in self._out_features}


class DynamicStem(nn.Module):
    def __init__(
        self,
        in_channels=3,
        mid_channels=64,
        out_channels=64,
        input_res=None,
        sept_stem=True,
        norm="BN",
        affine=True,
    ):
        """
        Build basic STEM for Dynamic Network.
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()

        # start with 3 stem layers down-sampling by 4.
        self.stem_1 = Conv2dNormAct(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            bias=False,
            norm=nn.SyncBatchNorm(mid_channels),
            activation=nn.ReLU(),
        )

        # stem 2
        input_res = input_res // 2
        if not sept_stem:
            self.stem_2 = Conv2dNormAct(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.SyncBatchNorm(mid_channels),
                activation=nn.ReLU(),
            )

        else:
            self.stem_2 = nn.Sequential(
                Conv2dNormAct(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=mid_channels,
                    bias=False,
                ),
                Conv2dNormAct(
                    mid_channels,
                    mid_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    norm=nn.SyncBatchNorm(mid_channels),
                    activation=nn.ReLU(),
                ),
            )
        # stem 3
        if not sept_stem:
            self.stem_3 = Conv2dNormAct(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=nn.SyncBatchNorm(out_channels),
                activation=nn.ReLU(),
            )

        else:
            self.stem_3 = nn.Sequential(
                Conv2dNormAct(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=mid_channels,
                    bias=False,
                ),
                Conv2dNormAct(
                    mid_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                    norm=nn.SyncBatchNorm(out_channels),
                    activation=nn.ReLU(),
                ),
            )

        self.out_res = input_res // 2
        self.out_cha = out_channels
        # using Kaiming init
        for layer in [self.stem_1, self.stem_2, self.stem_3]:
            kaiming_init_module(layer, mode="fan_in")

    def forward(self, x):
        x = self.stem_1(x)
        x = self.stem_2(x)
        x = self.stem_3(x)
        return x

    @property
    def out_channels(self):
        return self.out_cha

    @property
    def stride(self):
        return 4

    @property
    def out_resolution(self):
        return self.out_res

    @property
    def flops(self):
        return self.real_flops


class DynamicNetwork(Backbone):
    """
    This module implements Dynamic Routing Network.
    It creates dense connected network on top of some input feature maps.
    """

    def __init__(
        self,
        init_channel,
        input_shape,
        cell_num_list,
        layer_num,
        norm="",
        cal_flops=True,
        cell_type="",
        max_stride=32,
        sep_stem=True,
        using_gate=False,
        small_gate=False,
        gate_bias=1.5,
        drop_prob=0.0,
        device=None,
    ):
        super(DynamicNetwork, self).__init__()
        self.device = device
        # set affine in BatchNorm
        if "Sync" in norm:
            self.affine = True
        else:
            self.affine = False
        # set scheduled drop path
        self.drop_prob = drop_prob
        if self.drop_prob > 0.0001:
            self.drop_path = True
        else:
            self.drop_path = False
        self.cal_flops = cal_flops
        self._size_divisibility = max_stride
        input_res = np.array(input_shape[1:3])

        self.stem = DynamicStem(
            3,
            out_channels=init_channel,
            input_res=input_res,
            sept_stem=sep_stem,
            norm=norm,
            affine=self.affine,
        )
        self._out_feature_strides = {"stem": self.stem.stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}
        self._out_feature_resolution = {"stem": self.stem.out_resolution}
        assert self.stem.out_channels == init_channel
        self.all_cell_list = nn.ModuleList()
        self.all_cell_type_list = []
        self.cell_num_list = cell_num_list[:layer_num]
        self._out_features = []
        # using the initial layer
        input_res = input_res // self.stem.stride
        in_channel = out_channel = init_channel
        self.annotator = Annotator([0],gridding=False)
        self.init_layer = Cell(
            C_in=in_channel,
            C_out=out_channel,
            norm=norm,
            allow_up=False,
            allow_down=True,
            input_size=input_res,
            cell_type=cell_type,
            cal_flops=False,
            using_gate=using_gate,
            small_gate=small_gate,
            gate_bias=gate_bias,
            affine=self.affine,
        )

        # add cells in each layer
        for layer_index in range(len(self.cell_num_list)):
            layer_cell_list = nn.ModuleList()
            layer_cell_type = []
            for cell_index in range(self.cell_num_list[layer_index]):
                # channel multi, when stride:4 -> channel:C, stride:8 -> channel:2C ...
                channel_multi = pow(2, cell_index)
                in_channel_cell = in_channel * channel_multi
                # add res and dim switch to each cell
                allow_up = True
                allow_down = True
                # add res up and dim down by 2
                if cell_index == 0 or layer_index == layer_num - 1:
                    allow_up = False
                # dim down and resolution up by 2
                if cell_index == 3 or layer_index == layer_num - 1:
                    allow_down = False
                res_size = input_res // channel_multi
                layer_cell_list.append(
                    Cell(
                        C_in=in_channel_cell,
                        C_out=in_channel_cell,
                        norm=norm,
                        allow_up=allow_up,
                        allow_down=allow_down,
                        input_size=res_size,
                        cell_type=cell_type,
                        cal_flops=cal_flops,
                        using_gate=using_gate,
                        small_gate=small_gate,
                        gate_bias=gate_bias,
                        affine=self.affine,
                    )
                )
                # allow dim change in each aggregation
                dim_up, dim_down, dim_keep = False, False, True

                # dim up and resolution down by 2
                if cell_index > 0:
                    dim_up = True
                # dim down and resolution up by 2
                if (
                    cell_index < self.cell_num_list[layer_index] - 1
                ) and layer_index > 2:
                    dim_down = True
                elif (
                    cell_index < self.cell_num_list[layer_index] - 2
                ) and layer_index <= 2:
                    dim_down = True
                # dim keep unchanged
                if (
                    layer_index <= 2
                    and cell_index == self.cell_num_list[layer_index] - 1
                ):
                    dim_keep = False
                # allowed cell operations
                layer_cell_type.append([dim_up, dim_keep, dim_down])
                if layer_index == len(self.cell_num_list) - 1:
                    name = "layer_" + str(cell_index)
                    self._out_feature_strides[name] = channel_multi * self.stem.stride
                    self._out_feature_channels[name] = in_channel_cell
                    self._out_feature_resolution[name] = res_size
                    self._out_features.append(name)
            self.all_cell_list.append(layer_cell_list)
            self.all_cell_type_list.append(layer_cell_type)
        self.final_gathers = nn.ModuleList(
            GatherRouter(fabric_type="single_cell_combine")
            for _ in range(self.cell_num_list[-1])
        )

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        h_l1 = self.stem(x)
        h_l1 = self.annotator(h_l1)

        # the initial layer
        h_l1_list = self.init_layer([h_l1])
        prev_out_list = [h_l1_list]
        # build forward outputs
        for layer_index in range(len(self.cell_num_list)):
            layer_input, layer_output = [], []
            # aggregate cell input
            for cell_index in range(len(self.all_cell_type_list[layer_index])):
                cell_input = []

                if self.all_cell_type_list[layer_index][cell_index][0]:
                    cell_input.extend(prev_out_list[cell_index - 1][2])
                if self.all_cell_type_list[layer_index][cell_index][1]:
                    cell_input.extend(prev_out_list[cell_index][1])
                if self.all_cell_type_list[layer_index][cell_index][2]:
                    cell_input.extend(prev_out_list[cell_index + 1][0])

                layer_input.append(cell_input)

            # calculate each cell
            for _cell_index in range(len(self.all_cell_type_list[layer_index])):
                cell_output = self.all_cell_list[layer_index][_cell_index](
                    h_l1=layer_input[_cell_index]
                )

                layer_output.append(cell_output)

            # update layer output
            prev_out_list = layer_output
            # print(f"layer_index: {layer_index}, layer_output:")
            # for cell_id, cell_output in enumerate(layer_output):  # cell index
            #     for outputs in cell_output:
            #         if outputs:
            #             print(f"cell_id: {cell_id} [",end =" ")
            #             for id, output in enumerate(outputs):
            #                 print(f"output_{id}: {output.numel()},",end =" ")
            #             print("]")
            # print("")
        final_out_list = []
        for out_idx in range(len(prev_out_list)):
            final_out = self.final_gathers[out_idx](prev_out_list[out_idx][1])
            final_out_list.append(final_out)
        return final_out_list

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                height=self._out_feature_resolution[name][0],
                width=self._out_feature_resolution[name][0],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def input_shape(self):
        pass


def build_dynamic_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a Dynamic Backbone from config.
    Args:
        cfg: a dl_lib CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = DynamicNetwork(
        init_channel=cfg.MODEL.BACKBONE.INIT_CHANNEL,
        input_shape=input_shape,
        cell_num_list=cfg.MODEL.BACKBONE.CELL_NUM_LIST,
        layer_num=cfg.MODEL.BACKBONE.LAYER_NUM,
        norm=cfg.MODEL.BACKBONE.NORM,
        cal_flops=cfg.MODEL.CAL_FLOPS,
        cell_type=cfg.MODEL.BACKBONE.CELL_TYPE,
        max_stride=cfg.MODEL.BACKBONE.MAX_STRIDE,
        sep_stem=cfg.MODEL.BACKBONE.SEPT_STEM,
        using_gate=cfg.MODEL.GATE.GATE_ON,
        small_gate=cfg.MODEL.GATE.SMALL_GATE,
        gate_bias=cfg.MODEL.GATE.GATE_INIT_BIAS,
        drop_prob=cfg.MODEL.BACKBONE.DROP_PROB,
        device=cfg.MODEL.DEVICE,
    )

    return backbone
