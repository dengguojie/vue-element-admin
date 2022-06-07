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
Information containers for tuple reduce schedule
"""

# Standard Packages
from typing import List
from typing import AnyStr
from typing import Iterable
from functools import reduce

# Platform Information
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform.platform_info import ASCEND_910B
from tbe.common.platform.platform_info import ASCEND_910
from tbe.common.platform.platform_info import ASCEND_310P
from tbe.common.platform.platform_info import ASCEND_310B
from tbe.common.platform.platform_info import SHORT_SOC_VERSION
from tbe.common.platform.platform_info import CORE_NUM
from tbe.common.platform.platform_info import UB_SIZE

# Local Packages
from tbe import tvm
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_compile_info
from tbe.common.utils import op_tiling
from tbe.dsl.unify_schedule import util
from tbe.dsl.unify_schedule.pattern_parser import _dfs_compute
from tbe.dsl.unify_schedule.pattern_parser import ComputeType
from tbe.tvm.tensor import Tensor
from tbe.tvm.schedule import Stage
from ...constants import INSN_MAPPING
from ...constants import DTYPE_BYTE_MAPPING

# Tuple Reduce Packages
from . import tuple_reduce_schedule_helper

LOCAL_UB = "local.UB"
OP_TYPE_AUTO_TILING = "AutoTiling"


def ceil_div(x, y): return (x + y - 1) // y
def product(lst): return reduce(lambda x, y: x * y, lst)


def check_atomic_add_support(version, dtype):
    if version not in [ASCEND_910B, ASCEND_910, ASCEND_310P, ASCEND_310B]:
        return False
    if dtype != "float32":
        return False
    return True


class Info:
    """
    Information Container
    """

    def __init__(self, outs: Iterable[Tensor]):
        # SOC INFORMATION
        self.version = get_soc_spec(SHORT_SOC_VERSION)
        self.core_num = get_soc_spec(CORE_NUM)
        self.ub_size = get_soc_spec(UB_SIZE)
        self.block_size = 32

        # COMPUTE GRAPH
        self.outs = outs
        self.compute_type_size_map, self.compute_type_tensor_map = _dfs_compute(
            outs)
        self.tensors: List[Tensor] = self.compute_type_tensor_map.get(
            ComputeType.ANY)
        self.placeholder: List[Tensor] = self.compute_type_tensor_map.get(
            ComputeType.PLACEHOLDER)
        self.reduce_tensor: List[Tensor] = self.compute_type_tensor_map.get(
            ComputeType.REDUCE)
        self.broadcast_tensor: List[Tensor] = self.compute_type_tensor_map.get(
            ComputeType.BROADCAST)
        self.elewise_tensor: List[Tensor] = self.compute_type_tensor_map.get(
            ComputeType.ELEWISE)

        # KEY FEATURES
        self.max_shape = util.shape_to_list(
            self.reduce_tensor[0].op.input_tensors[0].shape)
        self.dim_var_code = self.dim_var_encode(self.max_shape)
        self.min_dtype, self.max_dtype = self.min_max_dtype()
        self.reduce_axis_var = [
            axis.var for axis in self.reduce_tensor[0].op.body[0].axis]
        self.all_axis_var = list(
            self.reduce_tensor[0].op.body[0].source[0].args)
        self.reduce_axis = [self.all_axis_var.index(
            axis) for axis in self.reduce_axis_var]
        self.last_reduce = False if max(self.reduce_axis) < len(
            self.max_shape) - 1 else True
        self.keep_dims = True
        self.reduce_mode = self.reduce_axis[-1] == len(self.max_shape) - 1
        self.atomic_support = check_atomic_add_support(
            self.version, self.reduce_tensor[0].dtype)
        self.atomic_threshold = self.core_num * 64

        # TILING INFORMATION
        self.init_tiling_information()

        # SCHEDULE SWITCHES
        # Initialize the switches
        self.mem_unique = False
        self.double_buffer = False
        self.compute_root = False
        self.transpose_reduce = False
        self.align_pad = False
        # mem_unique
        if len(self.placeholder) == 1:
            self.mem_unique = True
            self.unique_tensors = self.placeholder
        # double buffer (const only)
        # compute root (broadcast branch only)

        # calc buffer count
        self.calc_buffer_count()

        # ADD COMPILE INFO
        self.add_compile_info()

        # CONST TILING IF NECESSARY
        if self.is_const:
            self.const_tiling()

    @staticmethod
    def get_insn(stage: Stage):
        tag = stage.op.tag
        if stage.op.tag.find("|") != -1:
            insn = tag.split("|")[0]
        else:
            insn = tag
        return INSN_MAPPING.get(insn, insn)

    @staticmethod
    def get_dtype_size(dtype: AnyStr):
        return DTYPE_BYTE_MAPPING.get(dtype)

    @staticmethod
    def dim_var_encode(shape):
        one_hot = [1 if isinstance(
            thisaxis, tvm.expr.Var) else 0 for thisaxis in shape]
        binary = 0
        for v in one_hot[::-1]:
            binary = 2 * binary + v
        return binary

    def init_tiling_information(self):
        self.is_const = False if self.dim_var_code else True
        self.ub_size -= 2048
        self.ub_block = self.ub_size // self.block_size
        self.graph = tuple_reduce_schedule_helper.Compute(self.outs)
        self.short_tensors = set(self.graph.broadcast_branch).union(
            self.graph.reduce_tensors)
        self.grande_tensors = set(self.graph.tensors) - self.short_tensors
        self.unique_tensors = {}

    def calc_buffer_count(self):
        # calc buffer count
        self.buffer_count = self.graph.buffer_count(
            self.grande_tensors, self.short_tensors, self.unique_tensors)

        # dichotomy
        self.buffer_count[0] += 1

        # calc buffer size
        if self.is_const:
            self.short_buffer_size = 0
            for tensor in self.short_tensors:
                prod = ceil_div(product(util.shape_to_list(
                    tensor.shape)) * DTYPE_BYTE_MAPPING.get(self.max_dtype), self.block_size)
                if self.short_buffer_size < prod:
                    self.short_buffer_size = prod
            if ceil_div(self.ub_block,
                        self.graph.ratio * self.buffer_count[0] + self.buffer_count[1]) < self.short_buffer_size:
                self.short_buffer_size = ceil_div(
                    self.ub_block, self.graph.ratio * self.buffer_count[0] + self.buffer_count[1])
        else:
            self.short_buffer_size = ceil_div(
                self.ub_block, self.graph.ratio * self.buffer_count[0] + self.buffer_count[1])
        self.grande_buffer_size = self.ub_block - \
            self.short_buffer_size * self.buffer_count[1]
        self.grande_buffer_size = self.grande_buffer_size // self.buffer_count[0]

        self.short_buffer_size = self.short_buffer_size * \
            self.block_size // DTYPE_BYTE_MAPPING.get(self.max_dtype)
        self.grande_buffer_size = self.grande_buffer_size * \
            self.block_size // DTYPE_BYTE_MAPPING.get(self.max_dtype)

    def min_max_dtype(self):
        min_dtype, max_dtype = self.outs[0].dtype, self.outs[0].dtype
        for tensor in self.tensors:
            if DTYPE_BYTE_MAPPING.get(tensor.dtype) > DTYPE_BYTE_MAPPING.get(max_dtype):
                max_dtype = tensor.dtype
            if DTYPE_BYTE_MAPPING.get(tensor.dtype) < DTYPE_BYTE_MAPPING.get(min_dtype):
                min_dtype = tensor.dtype
        return min_dtype, max_dtype

    def add_compile_info(self):
        # prepare compile info
        is_const = self.is_const
        common_info = [self.core_num,
                       self.ub_size,
                       self.block_size,
                       self.atomic_support,
                       self.dim_var_code,
                       self.atomic_threshold,
                       self.compute_root,
                       self.double_buffer,
                       self.mem_unique,
                       self.transpose_reduce,
                       self.align_pad]
        graph_info = [len(self.placeholder),
                      self.grande_buffer_size,
                      DTYPE_BYTE_MAPPING.get(self.max_dtype),
                      DTYPE_BYTE_MAPPING.get(self.min_dtype),
                      DTYPE_BYTE_MAPPING.get(self.reduce_tensor[0].dtype),
                      self.keep_dims]

        # add compile info
        add_compile_info_inner("_is_const", is_const)
        add_compile_info_inner("_common_info", common_info)
        add_compile_info_inner("_runtime", True)
        add_compile_info_inner("_graph_info", graph_info)

    def const_tiling(self):
        add_compile_info_inner("_runtime", False)
        inputs = [{"shape": util.shape_to_list(
            ph.shape), "dtype": ph.dtype} for ph in self.placeholder]
        outputs = [{"shape": util.shape_to_list(
            tensor.shape), "dtype": tensor.dtype} for tensor in self.outs]
        run_info = op_tiling.do_op_tiling(
            OP_TYPE_AUTO_TILING, get_compile_info(), inputs, outputs)
        tiling_data = op_tiling.decode(run_info.get("tiling_data"),
                                       {"block_tiling_factor": "int", "ub_tiling_factor": "int"})
        self.block_factor = tiling_data.get("block_tiling_factor")
        self.ub_factor = tiling_data.get("ub_tiling_factor")
        self.tiling_key = run_info.get("tiling_key")
        # modify runtime to False
        add_compile_info_inner("_runtime", True)
