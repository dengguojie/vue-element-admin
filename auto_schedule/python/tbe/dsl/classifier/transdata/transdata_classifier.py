#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
TransdataClassify
"""
import copy
import operator
from abc import ABC
from abc import abstractmethod
from functools import reduce
from typing import Any
from typing import Dict
from typing import Optional

from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

from .. import shape_classifier
from .constants import B32
from .constants import B64
from .constants import BLOCK
from .constants import BORROW_H_B8B16_BACKWARD
from .constants import BORROW_H_B8B16_FORWARD
from .constants import BORROW_N_B8B16_BACKWARD
from .constants import BORROW_N_B8B16_FORWARD
from .constants import DO_NOTHING
from .constants import DO_PAD
from .constants import DO_TRANSPOSE_PAD
from .constants import DTYPE_BYTE
from .constants import GENERAL_BACKWARD
from .constants import GENERAL_FORWARD
from .constants import PAD_8
from .constants import PAD_16
from .constants import PAD_32
from .constants import REINTERPRET_DTYPE
from .constants import RESERVED_SPACE
from .constants import UNKNOWN_DIM

_classifies = {}


@shape_classifier.register_classifier(shape_classifier.TRANSDATA)
def classify(ins: list, extra_params: Optional[Dict[str, Any]]):
    from tbe.common.buildcfg import get_current_build_config
    if get_current_build_config("enable_op_prebuild"):
        return [ins]

    src_shape = ins[0].get("shape")
    dst_shape = ins[1]

    if len(src_shape) == len(dst_shape):
        dict_args = {"errCode": "E90001",
                     "detailed_cause": "Pad or DePad is required in transdata,"
                                       "but length of src is equal to dst"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    category = "forward" if len(src_shape) < len(dst_shape) else "backward"
    return _classifies.get(category)(ins).classify()


class TransdataClassify(ABC):
    """
    TransdataClassify
    """

    def __init__(self, ins):
        # base-info
        self._ins = copy.deepcopy(ins)
        self._axes_map = self._ins[2]
        self._dst_shape = self._ins[1]
        self._src_shape = self._ins[0].get("shape", None)
        self._dtype = self._ins[0].get("dtype", None)

        # other-info
        self.index_c = None
        self.is_transpose = False
        self.is_last_transpose = False
        self.is_forward = True if self.get_category().find("forward") != -1 else False

        # branch-info
        self.base_branch = GENERAL_FORWARD
        self.bn_branch = BORROW_N_B8B16_FORWARD
        self.bh_branch = BORROW_H_B8B16_FORWARD
        self.update_branch()

        # pad-info
        self.src_pad_mode = []
        self.src_pad_var = []

        # transpose-info
        self.src, self.dst = [], []
        self.pad, self.de_pad = [], []
        self.combine = None

        # reinterpret fp32 as fp16-mode
        self.soc_ub_size = get_soc_spec("UB_SIZE")
        self.ori_bit = DTYPE_BYTE.get(self._dtype, None)
        self.bit = self.ori_bit
        if self.bit in [B32, B64]:
            self.bit = DTYPE_BYTE.get(REINTERPRET_DTYPE.get(self._dtype, None), None)
        self.block_size = BLOCK // self.ori_bit

    def __init_subclass__(cls, **kwargs):
        _classifies[cls.get_category()] = cls

    @classmethod
    @abstractmethod
    def get_category(cls):
        """"""

    @abstractmethod
    def classify(self):
        """"""

    def update_branch(self):
        # update default-branch
        if not self.is_forward:
            self.base_branch = GENERAL_BACKWARD
            self.bn_branch = BORROW_N_B8B16_BACKWARD
            self.bh_branch = BORROW_H_B8B16_BACKWARD

    def add_const_compile_info(self, shape, axes_map):
        """
        Const's input from Classify would be tiling's input.
        Func make const-shape matched with compileInfo.
        """
        src_pad_mode, src_pad_var, permute = [], [], []
        index = 0 if not self.is_forward else 1
        for var in axes_map.items():
            if isinstance(var[index], int):
                src_pad_mode.append(DO_NOTHING)
                src_pad_var.append(1)
                permute.append(var[index])
            elif isinstance(var[index], (list, tuple)) and len(var[index]) == 1:
                src_pad_mode.append(DO_PAD)
                src_pad_var.append(_pad_refined(shape[var[index][0]]))
                permute.extend(var[index])
            else:
                src_pad_mode.append(DO_TRANSPOSE_PAD)
                src_pad_var.append(_pad_refined(shape[var[index][-1]]))
                permute.extend(var[index])

        operation.add_compile_info_inner("_src_pad_mode", src_pad_mode)
        operation.add_compile_info_inner("_src_pad_var", src_pad_var)
        operation.add_compile_info_inner("_permute", permute)

    def add_dynamic_compile_info(self, src_fuse, permute, shape):
        """
        Convert origin compileInfo that from use-defined to tiling.
        """
        index = 0 if not self.is_forward else 1
        for var in self._axes_map.items():
            if isinstance(var[index], int):
                self.src_pad_mode.append(DO_NOTHING)
                self.src_pad_var.append(1)
            elif isinstance(var[index], (list, tuple)) and len(var[index]) == 1:
                self.src_pad_mode.append(DO_PAD)
                self.src_pad_var.append(_pad_refined(shape[var[index][0]]))
            else:
                self.src_pad_mode.append(DO_TRANSPOSE_PAD)
                self.src_pad_var.append(_pad_refined(shape[var[index][-1]]))
                self.index_c = len(self.src_pad_mode) - 1

        operation.add_compile_info_inner("_src_pad_mode", self.src_pad_mode)
        operation.add_compile_info_inner("_src_pad_var", self.src_pad_var)
        operation.add_compile_info_inner("_permute", permute)
        operation.add_compile_info_inner("_src_fuse", src_fuse)

    def create_base_result(self, category):
        # return base result
        self._ins[0]["bit"] = self.ori_bit
        self._ins[0]["ori_bit"] = self.ori_bit
        self._ins[0]["shape"] = self._src_shape
        self._ins[0]["range"] = [[1, None] if x == UNKNOWN_DIM else [x, x] for x in self._src_shape]
        self._ins[0]["is_forward"] = self.is_forward
        self._ins[0]["transdata_category"] = category
        self._ins[1] = self._dst_shape
        self._ins[2] = self._axes_map

    def create_bn_bh_result(self, category):
        # choose BH||BN
        src_shape = self._src_shape.copy()
        dst_shape = self._dst_shape.copy()
        axes_map = self._axes_map.copy()
        dtype = self._dtype

        # Reinterpret fp32-tensor by fp16-mode
        if self.ori_bit != self.bit:
            dtype = REINTERPRET_DTYPE.get(dtype, None)
            src_shape = src_shape + [self.ori_bit // self.bit, ]
            dst_shape = dst_shape + [self.ori_bit // self.bit, ]
            if self.is_forward:
                axes_map[len(axes_map)] = len(dst_shape) - 1
            else:
                axes_map[len(src_shape) - 1] = len(axes_map)

        ins = copy.deepcopy(self._ins)
        ins[0]["bit"] = self.bit
        ins[0]["ori_bit"] = self.ori_bit
        ins[0]["shape"] = src_shape
        ins[0]["dtype"] = dtype
        ins[0]["range"] = [[1, None] if x == UNKNOWN_DIM else [x, x] for x in src_shape]
        ins[0]["transdata_category"] = category
        ins[1] = dst_shape
        ins[2] = axes_map
        return ins

    def choose_base_branch_by_model(self, src):
        # Current support(Template)
        # Don't use high-mode(dynamic, const)
        # Current only support NHC->NC1HC0
        self.is_transpose = src != sorted(src)
        self.is_last_transpose = src[-1] != len(src) - 1
        return (not self.is_transpose) or self.is_last_transpose

    def check_burst_len(self, shape, perm):
        # Can borrow any axis except C.
        # Axis that do transpose could be borrowed as burst_len just like (H in NHC).
        # Axis that do nothing could be borrowed as n_burst just like (N in NHC).
        # NCH's idx is [0,1,2] that mapping to perm should be [0,1,3].
        idx = [x + 1 if x > self.index_c else x for x in range(len(shape))]
        ub, burst_len_is_ok = 1, False
        for k, v in enumerate(reversed(idx)):
            index = len(idx) - 1 - k
            is_last_axis = index == len(idx) - 1
            is_c_axis = v == self.index_c
            is_transpose_axis = perm.index(v) != v

            used_buffer = ub
            if is_last_axis or is_c_axis or is_transpose_axis:
                used_buffer *= shape[index]
            if used_buffer >= self.block_size:
                burst_len_is_ok = True
                break
            ub *= shape[index]
        return burst_len_is_ok

    def const_strategy(self):
        # In the func, bh and bn are legal, only choose better.
        if self.is_forward:
            src = self.dst
            shape = self._src_shape
        else:
            src = self.src
            shape = self._dst_shape

        first_dim_is_n = src[0] == 0
        if not first_dim_is_n:
            return self.bh_branch

        # NHC||NCH
        num, buffer_size, target = 1, 0, 0
        ub_size = self.soc_ub_size // self.ori_bit // 2
        borrow_factor = BLOCK // self.bit
        for k, v in enumerate(reversed(shape)):
            k = len(shape) - 1 - k
            if k not in [self.index_c, len(shape) - 1]:
                if v >= borrow_factor:
                    used_buffer = v // borrow_factor * borrow_factor * num
                    if used_buffer >= ub_size:
                        used_buffer = ub_size
                else:
                    used_buffer = v * num
                    if used_buffer >= ub_size // borrow_factor * v:
                        used_buffer = ub_size // borrow_factor * v

                if used_buffer > buffer_size:
                    target = k
                    buffer_size = used_buffer
            num *= v
        return self.bn_branch if target == 0 else self.bh_branch

    def choose_branch_by_value(self, shape, perm):
        # choose base\bh\bn
        last_dim = shape[-1]
        if last_dim % self.block_size == 0 or (not self.check_burst_len(shape, perm)):
            return self.base_branch
        # Empirical method
        last_dim_limit = 1024 if self.is_forward else 128
        if last_dim >= last_dim_limit:
            return self.base_branch

        src = self.dst if self.is_forward else self.src
        base = math_prod(shape[self.index_c + 1:])
        base *= set_align(shape[self.index_c], self.src_pad_var[self.index_c])
        ub_size = (self.soc_ub_size - RESERVED_SPACE) // self.ori_bit // 2
        factor = BLOCK // self.bit

        # BH in backward and forward is different.
        first_dim_is_n = src[0] == 0
        bn_is_legal = ub_size >= factor * base and first_dim_is_n
        bh_is_legal = ub_size >= factor * base
        if not self.is_forward:
            bh_is_legal = ub_size >= factor * factor * base

        if bn_is_legal and bh_is_legal:
            return self.const_strategy()
        elif bn_is_legal:
            return self.bn_branch
        elif bh_is_legal:
            return self.bh_branch
        else:
            return self.base_branch


def _pad_refined(factor):
    # buffer_align has problem(Template-Code)
    for i in [PAD_32, PAD_16, PAD_8]:
        if factor % i == 0:
            return i
    return PAD_16 if factor == -1 else factor


def math_prod(iterable):
    return reduce(operator.mul, iterable, 1)


def set_align(dim, factor):
    return (dim + factor - 1) // factor * factor
