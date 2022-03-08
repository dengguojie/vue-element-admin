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
from typing import Set
from typing import List
from typing import Tuple
from typing import AnyStr
from typing import Iterable
from enum import Enum
from enum import auto

# Platform Information
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform.platform_info import ASCEND_920A
from tbe.common.platform.platform_info import ASCEND_910
from tbe.common.platform.platform_info import ASCEND_710
from tbe.common.platform.platform_info import SOC_VERSION
from tbe.common.platform.platform_info import CORE_NUM
from tbe.common.platform.platform_info import UB_SIZE

# Local Packages
from tbe import tvm
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import get_compile_info
from tbe.common.utils import op_tiling
from tbe.dsl.unify_schedule import util
from tbe.dsl.unify_schedule.pattern_parser import _dfs_compute
from tbe.dsl.unify_schedule.pattern_parser import ComputeType
from tbe.tvm.tensor import PlaceholderOp
from tbe.tvm.tensor import Tensor
from tbe.tvm.schedule import Stage
from ...constants import CompileInfo
from ...constants import INSN_MAPPING
from ...constants import DTYPE_BYTE_MAPPING

# Tuple Reduce Packages
from . import tuple_reduce_schedule_helper

LOCAL_UB = "local.UB"
OP_TYPE_AUTO_TILING = "AutoTiling"


def check_atomic_add_support(version, dtype):
    if version not in [ASCEND_920A, ASCEND_910, ASCEND_710]:
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
        self.version = get_soc_spec(SOC_VERSION)
        self.core_num = get_soc_spec(CORE_NUM)
        self.ub_size = get_soc_spec(UB_SIZE)
        self.block_size = 32

        # COMPUTE GRAPH
        self.outs = outs
        self.compute_type_size_map, self.compute_type_tensor_map = _dfs_compute(outs)
        self.tensors: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.ANY)
        self.placeholder: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.PLACEHOLDER)
        self.reduce_tensor: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.REDUCE)
        self.broadcast_tensor: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.BROADCAST)
        self.elewise_tensor: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.ELEWISE)

        # KEY FEATURES
        self.max_shape = util.shape_to_list(self.reduce_tensor[0].op.input_tensors[0].shape)
        self.dim_var_code = self.dim_var_encode(self.max_shape)
        self.min_dtype, self.max_dtype = self.min_max_dtype()
        self.reduce_axis_var = [axis.var for axis in self.reduce_tensor[0].op.body[0].axis]
        self.all_axis_var = list(self.reduce_tensor[0].op.body[0].source[0].args)
        self.reduce_axis = [self.all_axis_var.index(axis) for axis in self.reduce_axis_var]
        self.keep_dims = True
        self.reduce_mode = self.reduce_axis[-1] == len(self.max_shape) - 1
        self.atomic_support = check_atomic_add_support(self.version, self.reduce_tensor[0].dtype)

        # TILING INFORMATION
        self.is_const = False if self.dim_var_code else True
        self.buffer_count = self.ub_buffer_count(tuple_reduce_schedule_helper.Schedule(outs)) + 4
        self.buffer_size = self.ub_size // self.buffer_count // DTYPE_BYTE_MAPPING.get(self.max_dtype)

        # ADD COMPILE INFO
        self.add_compile_info()

        # CONST TILING IF NECESSARY
        if self.is_const:
            self.const_tiling()
    
    def min_max_dtype(self):
        min_dtype, max_dtype = self.outs[0].dtype, self.outs[0].dtype
        for tensor in self.tensors:
            if DTYPE_BYTE_MAPPING.get(tensor.dtype) > DTYPE_BYTE_MAPPING.get(max_dtype):
                max_dtype = tensor.dtype
            if DTYPE_BYTE_MAPPING.get(tensor.dtype) < DTYPE_BYTE_MAPPING.get(min_dtype):
                min_dtype = tensor.dtype
        return min_dtype, max_dtype
    
    @staticmethod
    def dim_var_encode(shape):
        one_hot = [1 if isinstance(thisaxis, tvm.expr.Var) else 0 for thisaxis in shape]
        binary = 0
        for v in one_hot[::-1]:
            binary = 2 * binary + v
        return binary
    
    def add_compile_info(self):
        # prepare compile info
        is_const = self.is_const
        common_info = [self.core_num,
                       self.ub_size,
                       self.block_size,
                       self.atomic_support,
                       self.dim_var_code]
        graph_info = [len(self.placeholder),
                      self.buffer_count,
                      DTYPE_BYTE_MAPPING.get(self.max_dtype),
                      DTYPE_BYTE_MAPPING.get(self.min_dtype),
                      self.keep_dims]
        
        # add compile info
        add_compile_info_inner("_is_const", is_const)
        add_compile_info_inner("_common_info", common_info)
        add_compile_info_inner("_runtime", True)
        add_compile_info_inner("_graph_info", graph_info)
    
    def const_tiling(self):
        add_compile_info_inner("_runtime", False)
        inputs = [{"shape": util.shape_to_list(ph.shape), "dtype": ph.dtype} for ph in self.placeholder]
        outputs = [{"shape": util.shape_to_list(tensor.shape), "dtype": tensor.dtype} for tensor in self.outs]
        run_info = op_tiling.do_op_tiling(OP_TYPE_AUTO_TILING, get_compile_info(), inputs, outputs)
        tiling_data = op_tiling.decode(run_info.get("tiling_data"),
                                       {"block_tiling_factor": "int", "ub_tiling_factor": "int"})
        self.block_factor = tiling_data.get("block_tiling_factor")
        self.ub_factor = tiling_data.get("ub_tiling_factor")
        self.tiling_key = run_info.get("tiling_key")
        # modify runtime to False
        add_compile_info_inner("_runtime", True)

    @staticmethod
    def ub_buffer_count(sch: tuple_reduce_schedule_helper.Schedule) -> int:
        def _dfs(root: Stage, idx: int = 0, base: int = 0):
            v = base + idx
            if root in base_map:
                v = max(base + idx, base_map.get(root))
            base_map.update({root: v})
            producers = sch.producer(root)
            for i, p in enumerate(producers):
                _dfs(p, i, base_map.get(root))
        
        root = sch.stage_map[sch.real_outputs[0]]
        base_map = {}
        _dfs(root)
        buffer_count = {}
        for stage in base_map:
            if stage.op.tag.find("tuple_reduce_sum") != -1:
                buffer_count.update({stage: base_map.get(stage) + len(stage.op.input_tensors) + 2})
            else:
                buffer_count.update({stage: base_map.get(stage) + len(stage.op.input_tensors) + 1})
        max_buffer_count = 0
        for k in buffer_count:
            max_buffer_count = max(max_buffer_count, buffer_count.get(k))
        return max_buffer_count
    
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
