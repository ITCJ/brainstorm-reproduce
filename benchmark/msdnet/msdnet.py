# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List

import torch.nn as nn
import torch
import math

from brt import Annotator
from brt.router import ScatterRouter, GatherRouter

is_measure = False


def write_json(nn: nn.Module, shape_in, shape_out):
    file_json = open("msdnet.json", "a+")
    file_json.write("{")
    conv_node = nn[0]
    bias = "false"
    if (
        isinstance(conv_node, torch.nn.modules.conv.Conv2d)
        and conv_node.bias is not None
    ):
        bias = "true"
    if isinstance(conv_node, torch.nn.modules.pooling.MaxPool2d):
        file_json.write(
            f'"module_name": "MaxPool2d", "kernel_size": {conv_node.kernel_size}, "stride": {conv_node.stride}, "padding": {conv_node.padding}, "input_shape": {list(shape_in)}, "output_shape": {list(shape_out)}'
        )
    elif isinstance(conv_node, torch.nn.modules.pooling.AvgPool2d):
        file_json.write(
            f'"module_name": "AvgPool2d", "kernel_size": {conv_node.kernel_size}, "input_shape": {list(shape_in)}, "output_shape": {list(shape_out)}'
        )

    elif isinstance(conv_node, torch.nn.modules.conv.Conv2d):
        file_json.write(
            f'"module_name": "Conv2d", "in_channels": {conv_node.in_channels}, "out_channels": {conv_node.out_channels}, "kernel_size": {conv_node.kernel_size[0]}, "stride": {conv_node.stride[0]}, "padding": {conv_node.padding[0]}, "dilation": null, "groups": 1, "bias": {bias}, "padding_mode": "zeros", "norm": "SyncBatchNorm", "activation": "ReLU", "input_shape": {list(shape_in)}, "output_shape": {list(shape_out)}'
        )
    else:
        raise ValueError("not support")
    file_json.write("}" + "\n")

    file_json.close()
    return


def write_shape(shape, name):
    file_json = open("msdnet.json", "a+")
    if name == "input":
        file_json.write("input_shape:" + str(shape) + "\n")
    else:
        file_json.write("output_shape:" + str(shape) + "\n")
    file_json.close()
    return


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1, padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                nIn,
                nOut,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True),
        )

    def forward(self, x):
        if self.training == True:
            self.net[0].input_shape = x.size()
            self.net[0].output_shape = self.net(x).size()
            write_json(self.net, x.size(), self.net(x).size())
        return self.net(x)


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        print(x)
        return x


# from brt.trace.leaf_node import register_leaf_node
# @register_leaf_node
class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck, bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        bottleneck_layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(
                nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False)
            )
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
        if type == "normal":
            layer.append(
                nn.Conv2d(nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False)
            )
        elif type == "down":
            layer.append(
                nn.Conv2d(nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False)
            )
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))

        layer.append(nn.ReLU(True))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        if self.training == False:
            return self.net(x)
        else:
            if len(self.net) == 3:
                self.net[0].input_shape = x.size()
                self.net[0].output_shape = self.net(x).size()
                write_json(self.net, x.size(), self.net(x).size())
                return self.net(x)
            input = x.clone()
            for i in range(len(self.net)):
                x = self.net[i](x)
                if i == 2:
                    mid = x.clone()
                    self.net[0].input_shape = input.size()
                    self.net[0].output_shape = x.size()
                    write_json(self.net, input.size(), x.size())
                if i == 5:
                    self.net[3].input_shape = mid.size()
                    self.net[3].output_shape = x.size()
                    write_json(self.net[3:], mid.size(), x.size())

            return x


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, "down", bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, "normal", bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1], self.conv_down(x[0]), self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, "normal", bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0], self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith("cifar"):
            self.layers.append(
                ConvBasic(nIn, nOut * args.grFactor[0], kernel=3, stride=1, padding=1)
            )
        elif args.data == "ImageNet":
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * args.grFactor[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
            self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        for i in range(1, args.nScales):
            self.layers.append(
                ConvBasic(nIn, nOut * args.grFactor[i], kernel=3, stride=2, padding=1)
            )
            nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            if self.training:
                if i == 0:
                    input = x.clone()
                    for j in range(len(self.layers[i])):
                        x = self.layers[i][j](x)
                        if j == 2:
                            mid = x.clone()
                            self.layers[i][0].input_shape = input.size()
                            self.layers[i][0].output_shape = x.size()
                            write_json(self.layers[i], input.size(), x.size())
                        if j == 3:
                            self.layers[i][3].input_shape = mid.size()
                            self.layers[i][3].output_shape = x.size()
                            write_json(self.layers[i][3:], mid.size(), x.size())
                    res.append(x)
                    continue
            x = self.layers[i](x)
            res.append(x)

        return res


class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(
                ConvDownNormal(
                    nIn1,
                    nIn2,
                    _nOut,
                    args.bottleneck,
                    args.bnFactor[self.offset - 1],
                    args.bnFactor[self.offset],
                )
            )
        else:
            self.layers.append(
                ConvNormal(
                    nIn * args.grFactor[self.offset],
                    nOut * args.grFactor[self.offset],
                    args.bottleneck,
                    args.bnFactor[self.offset],
                )
            )

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            self.layers.append(
                ConvDownNormal(
                    nIn1,
                    nIn2,
                    _nOut,
                    args.bottleneck,
                    args.bnFactor[i - 1],
                    args.bnFactor[i],
                )
            )

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """

    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        if self.training:
            input = x[-1].clone()
            input_reserved = x[-1].clone()
            for i in range(len(self.m)):
                input = self.m[i](input)
                if i == 1:
                    mid = input.clone()
                if i == 2:
                    self.m[2].input_shape = mid.size()
                    self.m[2].output_shape = input.size()
                    write_json(self.m[2:], mid.size(), input.size())
        res = self.m(x[-1])
        res = res.view(res.size(0), self.linear.in_features)
        return self.linear(res)


class MSDNet(nn.Module):
    def __init__(self, args, _is_measure=False):
        super(MSDNet, self).__init__()

        if _is_measure:
            self.train(True)
        else:
            self.train(False)
        
        self.annotator = Annotator([0])

        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.scatters = nn.ModuleList()

        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args

        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(
                args.step if args.stepmode == "even" else args.step * i + 1
            )
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(
                " ********************** Block {} "
                " **********************".format(i + 1)
            )
            m, nIn = self._build_block(
                nIn, args, self.steps[i], n_layers_all, n_layer_curr
            )
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith("cifar100"):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 100)
                )
            elif args.data.startswith("cifar10"):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 10)
                )
            elif args.data == "ImageNet":
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 1000)
                )
            else:
                raise NotImplementedError

        if args.init_routers:
            assert args.thresholds is not None
            self.build_routers(args.thresholds)
        else:
            self.routers_initilized = False

        for m in self.blocks:
            if hasattr(m, "__iter__"):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, "__iter__"):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):
        layers = [MSDNFirstLayer(3, nIn, args)] if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = args.nScales
            outScales = args.nScales
            if args.prune == "min":
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == "max":
                interval = math.ceil(1.0 * n_layer_all / args.nScales)
                inScales = args.nScales - math.floor(
                    1.0 * (max(0, n_layer_curr - 2)) / interval
                )
                outScales = args.nScales - math.floor(
                    1.0 * (n_layer_curr - 1) / interval
                )
            else:
                raise ValueError

            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))
            print(
                "|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|".format(
                    inScales, outScales, nIn, args.growthRate
                )
            )

            nIn += args.growthRate
            if args.prune == "max" and inScales > outScales and args.reduction > 0:
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(
                        nIn,
                        math.floor(1.0 * args.reduction * nIn),
                        outScales,
                        offset,
                        args,
                    )
                )
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print(
                    "|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|".format(
                        _t, math.floor(1.0 * args.reduction * _t)
                    )
                )
            elif (
                args.prune == "min"
                and args.reduction > 0
                and (
                    (n_layer_curr == math.floor(1.0 * n_layer_all / 3))
                    or n_layer_curr == math.floor(2.0 * n_layer_all / 3)
                )
            ):
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(
                        nIn,
                        math.floor(1.0 * args.reduction * nIn),
                        outScales,
                        offset,
                        args,
                    )
                )

                nIn = math.floor(1.0 * args.reduction * nIn)
                print("|\t\tTransition layer inserted! (min)\t|")
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(
                ConvBasic(
                    nIn * args.grFactor[offset + i],
                    nOut * args.grFactor[offset + i],
                    kernel=1,
                    stride=1,
                    padding=0,
                )
            )
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )

        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )

        return ClassifierModule(conv, nIn, num_classes)

    def build_routers(self, thresholds: List[float]):
        assert len(thresholds) == self.nBlocks - 1
        self.routers_initilized = True
        self.scatters = nn.ModuleList()

        for i in range(self.nBlocks - 1):
            self.scatters.append(
                ScatterRouter(
                    protocol_type="binary_threshold",
                    # fabric_type="single_cell_dispatch",
                    protocol_kwargs={"threshold": thresholds[i]},
                    fabric_kwargs={
                        "flow_num": 5,
                        "route_logic": ["1d", "1d", "1d", "1d", "1d"],
                        "transform": [False, False, False, False, False],
                    },
                    # capturing=True,
                    # capture_mode="max",
                )
            )
        self.final_gather = GatherRouter(
            # capturing=True,
            # fabric_type="single_cell_combine",
            # fabric_kwargs={"sparse": True}
            # fabric_type="combine", fabric_kwargs={"sparse": True}
        )
        # if "combine" not in self.final_gather.captured_fabric_type:
        #     self.final_gather.captured_fabric_type.append("combine")

    def forward(self, x, _is_measure=False):
        res = []
        if not self.training and self.routers_initilized:
            ##the actual dynamic routing
            x = self.annotator(x)
            for i in range(self.nBlocks):
                ## scatter all of the image
                x = self.blocks[i](x)
                tmp_res = self.classifier[i](x)

                if i < self.nBlocks - 1:
                    # x : 4 256 96 56 56 tmp_res : 256 1000
                    ##build a tesnsor with the size(0) of bs
                    sccater_input = []
                    result_scatter = []
                    for j, output in enumerate(x):
                        sccater_input.append(output)
                        result_scatter.append([])

                    for m in range(j + 1, 4):
                        sccater_input.append(torch.zeros_like(output))
                        result_scatter.append([])
                    sccater_input.append(tmp_res)
                    ##scatter
                    (
                        result_scatter[0],
                        result_scatter[1],
                        result_scatter[2],
                        result_scatter[3],
                        routed_res,
                    ) = self.scatters[i](sccater_input, tmp_res)
                    result = []
                    # differ the path by 0 and 1
                    for j, output in enumerate(x):
                        result.append(result_scatter[j][1])
                    routed_res = routed_res[0]
                    x = result
                else:
                    routed_res = tmp_res
                res.append(routed_res)
            ret = self.final_gather(res)
            return ret
        else:
            ## raw testing
            for i in range(self.nBlocks):
                x = self.blocks[i](x)
                res.append(self.classifier[i](x))
            return res
