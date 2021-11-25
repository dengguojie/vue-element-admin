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
general_transdata
"""
import operator
from copy import copy
from functools import reduce

from tbe import tvm
from tbe.common.utils.shape_util import shape_to_list

from .transdata_compute import TransdataComputation


class GeneralForwardComputation(TransdataComputation):
    """
    GeneralForwardComputation
    """
    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._axes_map = dict(sorted(axes_map.items()))

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return "general.forward"

    def do_compute(self):
        """
        Process
        """
        # example: (N, C, H, W) -> (N, C1*C0, H, W)
        padded_tensor = self._pad(self._tensor)

        # example: (N, C1*C0, H, W) -> (N, C1, C0, H, w)
        reshaped_tensor = self._reshape(padded_tensor)

        # example: (N, C1, C0, H, W) -> (N, C1, H, W, C0)
        transposed_tensor = self._transpose(reshaped_tensor)

        return transposed_tensor

    def _pad(self, tensor):
        def func(idx, pad_axis, tensor_):
            pad_cond = idx[pad_axis] >= self._src_shape[pad_axis]
            pad_value = tvm.const(self._pad_value, dtype=self._dtype)
            origin_value = tensor_[idx]
            return tvm.select(pad_cond, pad_value, origin_value)

        # pad one by one from lower dim
        shapes, axes = self._calc_pad()
        for j, (shape, axis) in enumerate(zip(shapes, axes)):
            with tvm.tag_scope("transdata|pad"):
                tensor = tvm.compute(shape, lambda *i: func(i, axis, tensor), name="pad_" + str(j),
                                     attrs={"axes": (axis,)})
        return tensor

    def _reshape(self, tensor):
        def func(idx):
            mapped_idx = []
            for axis in axes:
                if isinstance(axis, int):
                    mapped_idx.append(idx[axis])
                elif isinstance(axis, (tuple, list)):
                    s, e = axis[0], axis[-1] + 1
                    strides = (_math_prod(shape[(i + 1):e]) for i in axis)
                    mapped_idx.append(sum(a * b for a, b in zip(idx[s:e], strides)))
            return mapped_idx

        shape, axes = self._calc_reshape()
        with tvm.tag_scope("transdata|s_reshape"):
            tensor = tvm.compute(shape, lambda *i: tensor(*func(i)), name="reshape",
                                 attrs={"axes": axes})
        return tensor

    def _transpose(self, tensor):
        shape, permute = self._dst_shape, self._calc_permute()
        with tvm.tag_scope("transdata|transpose"):
            tensor = tvm.compute(shape, lambda *i: tensor(*[x for _, x in sorted(zip(permute, i))]),
                                 name="transpose", attrs={"permute": permute})

        return tensor

    def _calc_pad(self):
        shapes, axes = [self._src_shape], []
        _map_items = sorted(self._axes_map.items(), reverse=True)

        for k, v in _map_items:
            if isinstance(v, (tuple, list)):
                shape = copy(shapes[-1])
                shape[k] = _math_prod(self._dst_shape[i] for i in v)
                shapes.append(shape)
                axes.append(k)
        return shapes[1:], axes

    def _calc_reshape(self):
        shape, axes = [], []
        for k, v in self._axes_map.items():
            if isinstance(v, int):
                shape.append(self._src_shape[k])
                axes.append(len(shape) - 1)
            elif isinstance(v, (tuple, list)):
                shape.extend(self._dst_shape[i] for i in v)
                start, stop = len(shape) - len(v), len(shape)
                axes.append(tuple(range(start, stop)))
        return shape, axes

    def _calc_permute(self):
        permute = []
        for _, v in self._axes_map.items():
            if isinstance(v, int):
                permute.append(v)
            elif isinstance(v, (tuple, list)):
                permute.extend(v)
        return [x for _, x in sorted(zip(permute, range(0, len(permute))))]


class GeneralBackwardComputation(TransdataComputation):
    """
    GeneralBackwardComputation
    """
    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._axes_map = dict(sorted(axes_map.items(), key=lambda x: x[1]))

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return "general.backward"

    def do_compute(self):
        """
        Process
        """
        # example: (N, C1, H, W, C0) -> (N, C1, C0, H, W)
        transposed_tensor = self._transpose(self._tensor)

        # example: (N, C1, C0, H, W) -> (N, C1*C0, H, W)
        reshaped_tensor = self._reshape(transposed_tensor)

        # example: (N, C1*C0, H, W) -> (N, C, H, W)
        depadded_tensor = self._depad(reshaped_tensor)

        return depadded_tensor

    def _transpose(self, tensor):
        """
        eg: (N,C,H,W) -> (H,W,C,N), axes_map is {0:3, 1:0, 2:1, 3:2}, value of dict is base on dst,
        while sort axes_map by value, axs_map is {1:0, 2:1, 3:2, 0:3}, key of dict is base on src.
        """
        permute = self._get_permute()
        shape = tuple(self._src_shape[i] for i in permute)
        with tvm.tag_scope("transdata|transpose"):
            tensor = tvm.compute(shape, lambda *i: tensor(*[x for _, x in sorted(zip(permute, i))]),
                                 name="transpose", attrs={"permute": permute})
        return tensor

    def _reshape(self, tensor):
        def func(idx):
            mapped_idx = []
            for i, axis in enumerate(fused_axes):
                if isinstance(axis, int):
                    mapped_idx.append(idx[i])
                elif isinstance(axis, (tuple, list)):
                    remained = idx[i]
                    for x in axis:
                        stride = _math_prod(crt_shape[(x + 1):(axis[-1] + 1)])
                        mapped_idx.append(remained // stride)
                        remained = remained % stride
            return mapped_idx

        crt_shape = shape_to_list(tensor.shape)
        fused_shape, fused_axes = self._calc_reshape()
        with tvm.tag_scope("transdata|f_reshape"):
            reshape_tensor = tvm.compute(fused_shape, lambda *i: tensor(*func(i)), name="reshape",
                                         attrs={"axes": fused_axes})
        return reshape_tensor

    def _depad(self, tensor):
        shapes, axes = self._calc_depad(list(tensor.shape))
        for j, (shape, axis) in enumerate(zip(shapes, axes)):
            with tvm.tag_scope("transdata|depad"):
                tensor = tvm.compute(shape, lambda *i: tensor[i], name="depad_" + str(j),
                                     attrs={"axes": axis})
        return tensor

    def _get_permute(self):
        permute = []
        for k, _ in self._axes_map.items():
            if isinstance(k, int):
                permute.append(k)
            elif isinstance(k, (tuple, list)):
                permute.extend(k)
        return permute

    def _calc_reshape(self):
        fused_shape, fused_axes = [], []
        i = 0
        for k, v in self._axes_map.items():
            if isinstance(k, int):
                fused_shape.append(self._src_shape[k])
                fused_axes.append(i)
                i += 1
            elif isinstance(k, (tuple, list)):
                fused_shape.append(_math_prod(self._src_shape[j] for j in k))
                stop = i + len(k)
                fused_axes.append(tuple(range(i, stop)))
                i = stop
        return fused_shape, fused_axes

    def _calc_depad(self, crt_shape):
        shapes, axes = [crt_shape], []
        for k, v in self._axes_map.items():
            if isinstance(k, (tuple, list)):
                shape = copy(shapes[-1])
                shape[v] = self._dst_shape[v]
                shapes.append(shape)
                axes.append(v)
        return shapes[1:], axes


def _math_prod(iterable):
    return reduce(operator.mul, iterable, 1)
