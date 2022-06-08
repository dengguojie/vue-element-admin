#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
transdata op
"""
import abc
import operator
from functools import reduce

from tbe import tvm
from tbe.tvm.tensor import Tensor
from tbe.common.utils.shape_util import shape_to_list


class Op(abc.ABC):
    """
    Base packed operation
    """

    def __init__(self, producer=None, rule=None, dst_shape=None, op_name=None):
        # rule(producer) -> tensor
        self.producer = producer
        self.rule = rule
        self.dst_shape = dst_shape
        self.op_name = op_name

        self.tensor = None
        self._shape = None
        self._dtype = None

    @property
    def shape(self):
        if isinstance(self.tensor, Op):
            return shape_to_list(self.tensor.tensor.shape)
        if isinstance(self.tensor, Tensor):
            return shape_to_list(self.tensor.shape)
        return self._shape

    @property
    def dtype(self):
        if isinstance(self.tensor, Op):
            return self.tensor.tensor.dtype
        if isinstance(self.tensor, Tensor):
            return self.tensor.dtype
        return self._dtype

    @abc.abstractmethod
    def operation(self, *args):
        """"""

    def infer_axes(self, parent, index, half=1):
        """
        According to index based on parent to seek index based on child.
        :param parent: src-tensor
        :param index: index based on parent
        :param half: choose index[0] or index[1] if index is tuple that size is 2
        """
        producers = [self, ] + self.get_all_producers(self)
        producers = list(reversed(producers[:producers.index(parent)]))
        for tensor in producers:
            index = tensor.axis_mapping(index, half)
        return index

    def get_all_producers(self, tensor):
        # get all producers of tensor
        producers = list()
        if tensor.producer:
            producers.append(tensor.producer)
            producers.extend(self.get_all_producers(tensor.producer))
        return producers

    def axis_mapping(self, index, half):
        # base axis_mapping, special op need override
        index = self.rule[index]
        if isinstance(index, (list, tuple)):
            index = index[half] if len(index) == 2 else index[0]
        return index

    def calc_pad_axes(self):
        # calc pad info
        axes = []
        for i in self.rule:
            if isinstance(i, (list, tuple)):
                axes.append(i[0])
        return axes


class PlaceholderOp(Op):
    """
    PlaceholderOp
    """

    def __init__(self, tensor: Tensor, op_name="placeholder"):
        super().__init__(op_name=op_name)
        self.tensor = tensor

    def operation(self):
        """"""


class PadSReshapeOp(Op):
    """
    PadSReshapeOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, op_name="pad_s_reshape"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation()

    def operation(self):
        # make [N,H,C] -> [N,Hx,C] -> [N,H1,H0,C]
        def pad_s_reshape(idx):
            mapped_idx = []
            pad_axises = []
            for k, axis in enumerate(self.rule):
                if isinstance(axis, int):
                    mapped_idx.append(idx[axis])
                elif isinstance(axis, (tuple, list)):
                    s, e = axis[0], axis[-1] + 1
                    strides = (math_prod(self.dst_shape[(i + 1):e]) for i in axis)
                    mapped_idx.append(sum(a * b for a, b in zip(idx[s:e], strides)))
                    pad_axises.append(k)

            cond = tvm.all(*[mapped_idx[j] < self.producer.shape[j] for j in pad_axises])
            value = self.producer.tensor(*mapped_idx)
            return tvm.select(cond, value, None)

        with tvm.tag_scope("transdata|s_reshape"):
            self.tensor = tvm.compute(self.dst_shape, lambda *i: pad_s_reshape(i),
                                      name=self.op_name, attrs={"axes": self.rule})


class TransposeOp(Op):
    """
    TransposeOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, op_name="transpose"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation()

    def axis_mapping(self, index, half):
        # transpose axis_mapping by src and dst.
        return self.rule.index(index)

    def operation(self):
        # make [N,H,C1,C0] -> [N,C1,H,C0]
        def transpose(i):
            return self.producer.tensor(*[x for _, x in sorted(zip(self.rule, i))])

        with tvm.tag_scope("transdata|transpose"):
            self.tensor = tvm.compute(self.dst_shape, lambda *i: transpose(i),
                                      name=self.op_name, attrs={"permute": self.rule})


class PadOp(Op):
    """
    PadOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, pad_value=0, op_name="pad"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation(pad_value)

    def operation(self, pad_value):
        # make [N,H,C] -> [N,H,Cx]
        axis = self.calc_pad_axes()[0]

        def pad(idx, axis_, tensor_):
            cond = idx[axis_] >= tensor_.shape[axis_]
            var = tvm.const(pad_value, dtype=tensor_.dtype)
            value = tensor_[idx]
            return tvm.select(cond, var, value)

        with tvm.tag_scope("transdata|pad"):
            self.tensor = tvm.compute(self.dst_shape, lambda *i: pad(i, axis, self.producer.tensor),
                                      name=self.op_name, attrs={"axes": axis})


class DePadOp(Op):
    """
    DePadOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, op_name="depad"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation()

    def operation(self):
        # make [N,H,CX] -> [N,H,C]
        axis = self.calc_pad_axes()[0]
        with tvm.tag_scope("transdata|depad"):
            self.tensor = tvm.compute(self.dst_shape, lambda *i: self.producer.tensor[i],
                                      name=self.op_name, attrs={"axes": axis})


class SReshapeOp(Op):
    """
    SReshapeOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, op_name="s_reshape"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation()

    def operation(self):
        # make [N, H1*H0, C] as [N, H1, H0, C]
        def s_reshape(idx):
            mapped_idx = []
            for axis in self.rule:
                if isinstance(axis, int):
                    mapped_idx.append(idx[axis])
                elif isinstance(axis, (tuple, list)):
                    s, e = axis[0], axis[-1] + 1
                    strides = (math_prod(self.dst_shape[(i + 1):e]) for i in axis)
                    mapped_idx.append(sum(a * b for a, b in zip(idx[s:e], strides)))
            return self.producer.tensor(*mapped_idx)

        with tvm.tag_scope("transdata|s_reshape"):
            self.tensor = tvm.compute(self.dst_shape, lambda *i: s_reshape(i),
                                      name=self.op_name, attrs={"axes": self.rule})


class FReshapeOp(Op):
    """
    FReshapeOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, op_name="f_reshape"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation()

    def axis_mapping(self, index, half):
        # input -> output [0,[1,2],3].
        # 2 in input mapping 1 in output.
        for k, v in enumerate(self.rule):
            if isinstance(v, (list, tuple)) and index in v:
                return k
            if index == v:
                return k
        return None

    def operation(self):
        # make [N, H1, H0, C] as [N, H1*H0, C]
        def f_reshape(idx):
            mapped_idx = []
            for i, axis in enumerate(self.rule):
                if isinstance(axis, int):
                    mapped_idx.append(idx[i])
                elif isinstance(axis, (tuple, list)):
                    remained = idx[i]
                    for x in axis:
                        stride = math_prod(self.producer.shape[(x + 1):(axis[-1] + 1)])
                        mapped_idx.append(remained // stride)
                        remained = remained % stride
            return self.producer.tensor(*mapped_idx)

        with tvm.tag_scope("transdata|f_reshape"):
            self.tensor = tvm.compute(self.dst_shape, lambda *i: f_reshape(i),
                                      name=self.op_name, attrs={"axes": self.rule})


class SetValueOp(Op):
    """
    SetValueOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, cond, pad_value=0, op_name="set_value"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation(pad_value, cond)

    def operation(self, pad_value, cond):
        # Pad [N,H,C] that H belong to cond (eg: H in [2,10])
        axis = self.calc_pad_axes()[0]
        var = tvm.const(pad_value, self.producer.dtype)
        lambda_func = lambda *indice: tvm.select(cond(*indice), var, self.producer.tensor(*indice))
        with tvm.tag_scope("transdata|pad"):
            self.tensor = tvm.compute(self.dst_shape, lambda_func, name=self.op_name, attrs={"axes": axis})


class DataMoveOp(Op):
    """
    DataMoveOp
    """

    def __init__(self, tensor: Op, rule, dst_shape, op_name="res"):
        super().__init__(tensor, rule, dst_shape, op_name)
        self.operation()

    def operation(self):
        # ub2gm
        with tvm.tag_scope("transdata|res"):
            self.tensor = tvm.compute(self.dst_shape, lambda *i: self.producer.tensor[i], name=self.op_name)


def set_align(dim, factor):
    # set align by factor
    return tvm.floordiv(dim + factor - 1, factor) * factor


def math_prod(iterable):
    # calc prod for iter
    return reduce(operator.mul, iterable, 1)
