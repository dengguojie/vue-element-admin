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
performance transdata
"""
import operator
from copy import copy
from functools import reduce

from tbe import tvm
from tbe.common.utils.shape_util import shape_to_list
from tbe.dsl.base import operation
from tbe.dsl.base.classifier.transdata.constants import DTYPE_BYTE
from tbe.dsl.base.classifier.transdata.constants import BLOCK
from tbe.dsl.base.classifier.transdata.constants import BORROW_N_B8B16_BACKWARD, BORROW_N_B8B16_FORWARD
from .transdata_compute import TransdataComputation


class BorrowNBackwardComputation(TransdataComputation):
    """
    BorrowNBackwardComputation
    """

    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._axes_map = dict(sorted(axes_map.items(), key=lambda x: x[1]))

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return BORROW_N_B8B16_BACKWARD

    def _get_permute(self):
        permute = []
        for k in self._axes_map.keys():
            if isinstance(k, int):
                permute.append(k)
            elif isinstance(k, (tuple, list)):
                permute.extend(k)
        return permute

    def _calc_depad(self, src_shape, dst_shape):
        shapes, axes = [src_shape], []
        for k, v in self._axes_map.items():
            if isinstance(k, (tuple, list)):
                shape = copy(shapes[-1])
                shape[v] = dst_shape[v]
                shapes.append(shape)
                axes.append(v)
        return shapes[1:], axes

    def _calc_f_reshape_axes(self):
        _fused_axes = []
        i = 0
        for k, v in self._axes_map.items():
            if isinstance(k, int):
                _fused_axes.append(i)
                i += 1
            elif isinstance(k, (tuple, list)):
                stop = i + len(k)
                _fused_axes.append(tuple(range(i, stop)))
                i = stop
        return _fused_axes

    def _backward_preprocess(self):
        # example: (N, C1, H, W, C0) -> (Nx, C1, C0, H, W)
        pad_factor = BLOCK // DTYPE_BYTE.get(self._tensor.dtype, 1)
        padded_tensor = _pad(self._tensor, 0, pad_factor)

        # example: (Nx, C1, H, W, C0) -> (Nx.o, 16, C1, H, W, C0)
        axes = [[0, 1] if x == 0 else x + 1 for x in range(len(padded_tensor.shape))]
        s_reshaped_tensor = _s_reshape(padded_tensor, axes, pad_factor)

        # example: (Nx.o, 16, C1, H, W, C0) -> (Nx.o, C1, H, W, C0, 16)
        perm = [0, ] + list(range(2, len(s_reshaped_tensor.shape))) + [1, ]
        transposed_tensor = _transpose(s_reshaped_tensor, perm)

        # While borrow-n is work, axes-map should be adjusted.
        self._axes_map[len(transposed_tensor.shape) - 1] = len(self._axes_map)
        return transposed_tensor

    def _backward_process(self, tensor):
        # example: (Nx.o, C1, H, W, C0, 16) -> (Nx.o, H, W, C1, C0, 16)
        perm = self._get_permute()
        transposed_tensor = _transpose(tensor, perm)

        # example: (Nx.o, H, W, C1, C0, 16) -> (Nx.o, H, W, Cx, 16)
        axes = self._calc_f_reshape_axes()
        f_reshaped_tensor = _f_reshape(transposed_tensor, axes)

        # example: (Nx.o, H, W, Cx, 16) -> (Nx.o, H, W, C, 16)
        src_tensor = f_reshaped_tensor
        src_shape = shape_to_list(src_tensor.shape)
        dst_shape = src_shape.copy()
        for k, v in self._axes_map.items():
            if isinstance(k, (list, tuple)):
                dst_shape[v] = self._dst_shape[v]
        shapes, axes = self._calc_depad(src_shape, dst_shape)
        for shape, axis in zip(shapes, axes):
            src_tensor = _depad(src_tensor, axis, shape)
        return src_tensor

    def _backward_postprocess(self, tensor):
        # example: (Nx.o, H, W, C, 16) -> (Nx.o, 16, H, W, C)
        perm = [0, len(tensor.shape) - 1] + list(range(1, len(tensor.shape) - 1))
        backward_transposed_tensor = _transpose(tensor, perm)

        # example: (Nx.o, 16, H, W, C) -> (Nx, H, W, C)
        axes = [[0, 1], ] + list(range(2, len(backward_transposed_tensor.shape)))
        backward_f_reshaped_tensor = _f_reshape(backward_transposed_tensor, axes)

        # example: (Nx, H, W, C) -> (N, H, W, C)
        return _data_move(self._dst_shape, backward_f_reshaped_tensor)

    def do_compute(self):
        """
        Main Process
        """
        tensor = self._backward_preprocess()

        tensor = self._backward_process(tensor)

        return self._backward_postprocess(tensor)


class BorrowNForwardComputation(TransdataComputation):
    """
    BorrowNForwardComputation
    """

    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._axes_map = dict(sorted(axes_map.items()))

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return BORROW_N_B8B16_FORWARD

    def _get_permute(self):
        permute = []
        for _, v in self._axes_map.items():
            if isinstance(v, int):
                permute.append(v)
            elif isinstance(v, (tuple, list)):
                permute.extend(v)
        return permute

    def _calc_pad(self):
        axes = []
        for k, v in self._axes_map.items():
            if isinstance(v, (tuple, list)):
                axes.append(k)
        axes.sort(reverse=True)
        return axes

    def _calc_s_reshape(self):
        shape, axes = [], []
        for k, v in self._axes_map.items():
            if isinstance(v, int) or (isinstance(v, (tuple, list)) and len(v) == 1):
                shape.append(1)
                axes.append(len(shape) - 1)
            else:
                shape.extend([1, ] * len(v))
                start, stop = len(shape) - len(v), len(shape)
                axes.append(tuple(range(start, stop)))
        return axes

    def _preprocess(self):
        # example: (N, H, C) -> (Nx, H, C)
        pad_factor = BLOCK // DTYPE_BYTE.get(self._tensor.dtype, 1)
        padded_tensor = _pad(self._tensor, 0, pad_factor)

        # example: (Nx, H, C) -> (Nx.o, 16, H, C)
        axes = [[0, 1] if x == 0 else x + 1 for x in range(len(padded_tensor.shape))]
        s_reshaped_tensor = _s_reshape(padded_tensor, axes, pad_factor)

        # example: (Nx.o, 16, H, C) -> (Nx.o, H, C, 16)
        perm = [0, ] + list(range(2, len(s_reshaped_tensor.shape))) + [1, ]
        transposed_tensor = _transpose(s_reshaped_tensor, perm)

        # While borrow-n is work, axes-map should be adjusted.
        self._axes_map[len(self._axes_map)] = len(transposed_tensor.shape)
        return transposed_tensor

    def _process(self, tensor):
        # example: (Nx.o, H, C, 16) -> (Nx.o, HX, CX, 16)
        pad_factor = operation.get_compile_info().get("_pad_factor")
        for axis in self._calc_pad():
            tensor = _pad(tensor, axis, pad_factor)

        # example: (Nx.o, HX, CX, 16) -> (Nx,o, HX, C1, C0, 16)
        axes = self._calc_s_reshape()
        s_reshaped_tensor = _s_reshape(tensor, axes, pad_factor)

        # example: (Nx.o, HX, C1, C0, 16) -> (Nx.o, C1, HX, C0, 16)
        return _transpose(s_reshaped_tensor, self._get_permute())

    def _postprocess(self, tensor):
        # example: (Nx.o, C1, HX, C0, 16) -> (Nx.o, 16, C1, HX, C0)
        perm = [0, len(tensor.shape) - 1] + list(range(1, len(tensor.shape) - 1))
        transposed_tensor = _transpose(tensor, perm)

        # example: (Nx.o, 16, C1, HX, C0) -> (Nx, C1, HX, C0)
        axes = [[0, 1], ] + list(range(2, len(transposed_tensor.shape)))
        f_reshaped_tensor = _f_reshape(transposed_tensor, axes)

        # example: (Nx, H, W, C) -> (N, H, W, C)
        return _data_move(self._dst_shape, f_reshaped_tensor)

    def do_compute(self):
        """
        Main Process
        """
        tensor = self._preprocess()

        tensor = self._process(tensor)

        return self._postprocess(tensor)


def _pad(tensor, pad_axis, pad_factor, pad_value=0, name="pad"):
    """
    :param tensor: src-tensor
    :param pad_axis: int, index of pad-axes
    :param pad_factor: pad-align-var
    :param pad_value: filled var
    :return: dst-tensor
    """

    def func(idx, axis_, tensor_):
        pad_cond = idx[axis_] >= tensor_.shape[axis_]
        pad_var = tvm.const(pad_value, dtype=tensor_.dtype)
        origin_value = tensor_[idx]
        return tvm.select(pad_cond, pad_var, origin_value)

    shape = list(tensor.shape)
    shape[pad_axis] = _set_align(shape[pad_axis], pad_factor)
    with tvm.tag_scope("transdata|pad"):
        tensor = tvm.compute(shape, lambda *i: func(i, pad_axis, tensor), name=name, attrs={"axes": pad_axis})
    return tensor


def _depad(tensor, depad_axis, dst_shape, name="depad"):
    """
    :param tensor: src-tensor
    :param depad_axis: int, index of depad axes
    :param dst_shape: dst-tensor's shape
    :return: dst-tensor
    """
    with tvm.tag_scope("transdata|depad"):
        tensor = tvm.compute(dst_shape, lambda *i: tensor[i], name=name, attrs={"axes": depad_axis})
    return tensor


def _s_reshape(tensor, axes, s_factor, name="reshape"):
    """
    :param tensor: src-tensor
    :param axes: record which axes need to split
    :param s_factor: dim split as [floordiv(dim, s_factor), s_factor]
    :return: dst-tensor
    """

    def func(idx):
        mapped_idx = []
        for axis in axes:
            if isinstance(axis, int):
                mapped_idx.append(idx[axis])
            elif isinstance(axis, (tuple, list)):
                s, e = axis[0], axis[-1] + 1
                strides = (_math_prod(dst_shape[(i + 1):e]) for i in axis)
                mapped_idx.append(sum(a * b for a, b in zip(idx[s:e], strides)))
        return mapped_idx

    dst_shape = []
    for k, v in enumerate(axes):
        if isinstance(v, int):
            dst_shape.append(tensor.shape[k])
        else:
            dst_shape.extend([tvm.floordiv(tensor.shape[k], s_factor), s_factor])
    with tvm.tag_scope("transdata|s_reshape"):
        tensor = tvm.compute(dst_shape, lambda *i: tensor(*func(i)), name=name, attrs={"axes": axes})
    return tensor


def _transpose(tensor, perm, name="transpose"):
    """
    :param tensor: src-tensor
    :param perm: permute for transpose
    :return: dst-tensor
    """
    shape = tuple(tensor.shape[i] for i in perm)
    with tvm.tag_scope("transdata|transpose"):
        tensor = tvm.compute(shape, lambda *i: tensor(*[x for _, x in sorted(zip(perm, i))]),
                             name=name, attrs={"permute": perm})
    return tensor


def _f_reshape(tensor, axes, name="reshape"):
    """
    :param tensor: src-tensor
    :param axes: record which axes need to fuse
    :return: dst-tensor
    """

    def func(idx):
        mapped_idx = []
        for i, axis in enumerate(axes):
            if isinstance(axis, int):
                mapped_idx.append(idx[i])
            elif isinstance(axis, (tuple, list)):
                remained = idx[i]
                for x in axis:
                    stride = _math_prod(crt_shape[(x + 1):(axis[-1] + 1)])
                    mapped_idx.append(remained // stride)
                    remained = remained % stride
        return mapped_idx

    dst_shape = []
    crt_shape = shape_to_list(tensor.shape)
    for k, v in enumerate(axes):
        if isinstance(v, int):
            dst_shape.append(crt_shape[v])
        else:
            dst_shape.append(_math_prod(crt_shape[j] for j in v))
    with tvm.tag_scope("transdata|f_reshape"):
        reshape_tensor = tvm.compute(dst_shape, lambda *i: tensor(*func(i)), name=name, attrs={"axes": axes})
    return reshape_tensor


def _data_move(dst_shape, tensor, name="res"):
    with tvm.tag_scope("transdata|res"):
        tensor = tvm.compute(dst_shape, lambda *i: tensor[i], name=name)
    return tensor


def _set_align(dim, factor):
    return tvm.floordiv(dim + factor - 1, factor) * factor


def _math_prod(iterable):
    return reduce(operator.mul, iterable, 1)
