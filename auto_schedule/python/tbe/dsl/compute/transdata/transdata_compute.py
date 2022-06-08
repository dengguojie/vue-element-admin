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
transdata compute
"""
import abc
from copy import copy

from tbe import tvm
from tbe.dsl.base import operation
from tbe.common.utils.shape_util import shape_to_list
from .transdata_op import math_prod
from .transdata_op import set_align

_computes = {}


def transdata(tensor, dst_shape, axes_map, pad_value):
    """
    transdata interface that choose different computes by _transdata_category
    """
    category = operation.get_context().get_current_compute().get("_transdata_category")
    return _computes.get(category)(tensor, dst_shape, axes_map, pad_value).do_compute()


class TransdataComputation(abc.ABC):
    """
    TransdataComputation
    """

    def __init_subclass__(cls, **kwargs):
        _computes[cls.get_category()] = cls

    @abc.abstractmethod
    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        self._tensor = tensor
        self._dst_shape = shape_to_list(dst_shape)
        self._axes_map = axes_map
        self._pad_value = pad_value
        self._src_shape = shape_to_list(self._tensor.shape)
        self._dtype = self._tensor.dtype
        self._pad_mode = operation.get_compile_info().get("_src_pad_mode")
        self._pad_var = operation.get_compile_info().get("_src_pad_var")

        self.x_idx = None
        self.x_align_var = None

    @classmethod
    @abc.abstractmethod
    def get_category(cls):
        """"""

    @abc.abstractmethod
    def do_compute(self):
        """"""

    def calc_align_borrow_axis(self):
        # Borrow axis need be padded by factor.
        # Eg: n is 15 in bn, align n by 16.
        # pad H to Hx, pad N to Nx
        perm = list(range(len(self._tensor.shape)))
        perm[self.x_idx] = [perm[self.x_idx], ]
        dst_shape = copy(self._tensor.shape)
        dst_shape[self.x_idx] = set_align(dst_shape[self.x_idx], self.x_align_var)
        return perm, dst_shape

    def calc_pad(self, tensor):
        # pad one by one from lower dim
        shapes, perms = [tensor.shape, ], [list(range(len(tensor.shape))), ]
        for k, v in enumerate(reversed(self._pad_mode)):
            k = len(self._pad_mode) - 1 - k
            var = self._pad_var[k]
            # borrow-axis and don't do pad
            if k != self.x_idx and v != 0:
                i = tensor.infer_axes(self._tensor, k)
                shape = copy(shapes[-1])
                shape[i] = set_align(shape[i], var)
                perm = copy(perms[0])
                perm[i] = [perm[i], ]
                shapes.append(shape)
                perms.append(perm)
        return perms[1:], shapes[1:]

    def calc_depad(self, tensor, mode="bh"):
        """
        De-Pad one by one from higher dim.
        :param tensor: source_tensor before de-pad.
        :param mode: bh||bn transpose i1||i0 to the end, but i0||i1 would be de-pad.
        :return: perms, dst_shapes
        """
        shapes, perms = [tensor.shape, ], [list(range(len(tensor.shape))), ]
        # bh keep h0, bn keep n1
        x = tensor.infer_axes(self._tensor, self.x_idx, half=1 if mode == "bh" else 0)
        for k, v in enumerate(self._pad_mode):
            if v == 0 or k == x:
                # don't de-pad and borrowed_axes delay de-pad
                continue
            shape = copy(shapes[-1])
            shape[k] = self._dst_shape[k]
            perm = copy(perms[0])
            perm[k] = [perm[k], ]
            shapes.append(shape)
            perms.append(perm)
        return perms[1:], shapes[1:]

    def calc_transpose_0(self, tensor, half=0):
        """
        Transpose the borrowed axis to the end.
        :param tensor: src_tensor before transpose.
        :param half: 0||1 mean transpose i1||i0 to the end.
        :return: perm, dst_shape.
        """
        i = tensor.infer_axes(self._tensor, self.x_idx, half=half)
        perm = list(range(len(tensor.shape)))
        perm.pop(i)
        perm.append(i)
        return perm, tuple(tensor.shape[i] for i in perm)

    def calc_transpose_1(self, tensor, is_forward=True):
        """
        Transpose by origin-rule that from user defined.
        :param tensor: src_tensor before transpose.
        :param is_forward: forward or not.
        :return: perm, dst_shape.
        """
        perm = []
        for k, v in self._axes_map.items():
            var = v if is_forward else k
            if isinstance(var, int):
                perm.append(var)
            elif isinstance(var, (tuple, list)):
                perm.extend(var)
        perm.append(len(tensor.shape) - 1)
        return perm, tuple(tensor.shape[i] for i in perm)

    def calc_transpose_2(self, tensor, mode="bh"):
        """
        Transpose borrowed axis to the origin
        :param tensor: src_tensor before transpose
        :param mode: "bh" transpose i1, "bn" transpose i0.
        :return: perm, dst_shape
        """
        perm = list(range(len(tensor.shape)))
        i1 = tensor.infer_axes(self._tensor, self.x_idx, half=0)
        i0 = tensor.infer_axes(self._tensor, self.x_idx, half=1)
        if mode == "bh":
            perm.pop(i1)
            perm.insert(i0, i1)
        else:
            perm.pop(i0)
            perm.insert(i1 + 1, i0)
        return perm, tuple(tensor.shape[i] for i in perm)

    def calc_s_reshape(self, tensor, i, factor):
        """
        Split Reshape from H to [H1,H0].
        :param tensor: src_tensor before s_reshape.
        :param i: index of H based on tensor.
        :param factor: value of H0.
        :return: perm, dst_shape.
        """
        perm = [x if x <= i else x + 1 for x in range(len(tensor.shape))]
        perm[i] = [perm[i], perm[i] + 1]
        shape = copy(tensor.shape)
        shape.insert(i, tvm.floordiv(shape[i], factor))
        shape[i + 1] = factor
        return perm, shape

    def calc_f_reshape(self, tensor, i):
        """
        Fuse Reshape from [H1,H0] as H.
        :param tensor: src_tensor before f_reshape.
        :param i: index of H1 based on tensor.
        :return: perm, dst_shape.
        """
        # fuse H1 and ho_i as H1 * ho_i
        perm = list(range(len(tensor.shape)))
        perm.pop(i)
        perm[i] = [i, i + 1]
        shape = copy(tensor.shape)
        shape.pop(i)
        shape[i] = math_prod(tensor.shape[j] for j in perm[i])
        return perm, shape
