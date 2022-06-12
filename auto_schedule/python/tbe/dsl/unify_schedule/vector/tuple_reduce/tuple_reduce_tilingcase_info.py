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
from typing import AnyStr
from typing import Callable
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
from .tuple_reduce_schedule_helper import Compute


def ceil_div(x, y): return (x + y - 1) // y
def product(lst): return reduce(lambda x, y: x * y, lst)


class SoC:
    """
    SOC INFORMATION
    """

    def __init__(self) -> None:
        """
        Get SoC version from 'get_soc_spec'
        """
        self.version = get_soc_spec(SHORT_SOC_VERSION)
        self.core_num = get_soc_spec(CORE_NUM)
        self.ub_size = get_soc_spec(UB_SIZE)
        self.block_size = 32
        self.atomic_capability = [ASCEND_910B, ASCEND_910, ASCEND_310P, ASCEND_310B]

    def atomic(self) -> bool:
        """
        check whether current SoC support atomic add on gm
        """
        return self.version in self.atomic_capability


class ComputeGraph:
    """
    Compute Graph
    """

    def __init__(self, outs: Iterable[Tensor]) -> None:
        """
        traversal the whole graph, categorize tensors
        """
        self.outs: Iterable[Tensor] = outs
        self.compute_type_size_map, self.compute_type_tensor_map = _dfs_compute(outs)
        self.tensors: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.ANY)
        self.placeholder: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.PLACEHOLDER)
        self.reduce_tensor: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.REDUCE)
        self.broadcast_tensor: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.BROADCAST)
        self.elewise_tensor: List[Tensor] = self.compute_type_tensor_map.get(ComputeType.ELEWISE)
        if not self.broadcast_tensor:
            self.broadcast_tensor = []
        if not self.elewise_tensor:
            self.elewise_tensor = []


class Switch:
    """
    Schedule opt mode switches
    """

    def __init__(self, graph: ComputeGraph, is_const: bool) -> None:
        """
        initialize switches
        """
        self.graph: ComputeGraph = graph
        self.is_const: bool = is_const

        self.mem_unique: bool = False
        self.double_buffer: bool = False
        self.compute_root: bool = False
        self.transpose_reduce: bool = False
        self.align_pad: bool = False

    def check(self) -> None:
        """
        exec all member function whose name starts with 'check_'
        decide switches state for current graph
        """
        check_funcs = filter(lambda func_name: isinstance(getattr(self, func_name), Callable)
                             and func_name.startswith("check_"), self.__dir__())
        for func_name in check_funcs:
            getattr(self, func_name)()

    def check_mem_unique(self) -> None:
        """
        Compile-time
        """
        if len(self.graph.placeholder) == 1:
            self.mem_unique = True

    def check_double_buffer(self) -> None:
        """
        Disable
        """
        pass

    def check_compute_root(self) -> None:
        """
        Compile-time
        """
        if len(self.graph.broadcast_tensor) > 0:
            self.compute_root = True

    def check_transpose_reduce(self) -> None:
        """
        Runtime
        """
        pass

    def check_align_pad(self) -> None:
        """
        Runtime
        """
        pass


class BufferSize:
    """
    Estimate buffer size for each stage
    """

    def __init__(self, soc: SoC, graph: Compute, switches: Switch) -> None:
        """
        initialize ub size, categorize tensors and tensors weights
        """
        self.soc: SoC = soc
        self.graph: Compute = graph
        self.switches: Switch = switches
        self.ub_size = soc.ub_size - 2048
        self.ub_block = self.ub_size // soc.block_size

        self.unique_tensors: Set[Tensor] = set()
        self.short_tensors: Set[Tensor] = set()
        self.grande_tensors: Set[Tensor] = set()
        self.grande_buffer_size = 0
        self.short_buffer_size = 0
        self.categorize_tensors()
        self.tensors_weight()

        self.buffer_count = []

    def categorize_tensors(self) -> None:
        """
        category tensors into short tensors / grande tensors / unique tensors for buffer count
        """
        self.unique_tensors = set(self.graph.placeholder) if self.switches.mem_unique else set()
        if self.switches.compute_root:
            self.short_tensors = set(self.graph.broadcast_branch).union(self.graph.reduce_tensors)
        else:
            self.short_tensors = set(self.graph.reduce_tensors)
        self.grande_tensors = set(self.graph.tensors) - self.short_tensors

    def tensors_weight(self) -> None:
        """
        claim tensors weight such as this reduce tensor cost 2 extra buffer...
        """
        pass

    def const_estimate(self, dtype_size: int) -> None:
        """
        estimate buffer size more accurate in const mode
        """
        maximum_short_tensor_numel = 0
        for tensor in self.short_tensors:
            tensor_shape = util.shape_to_list(tensor.shape)
            align_granularity = self.soc.block_size // dtype_size
            tensor_shape[-1] = ceil_div(tensor_shape[-1], align_granularity) * align_granularity
            tensor_numel = product(tensor_shape)
            maximum_short_tensor_numel = max(maximum_short_tensor_numel, tensor_numel)
        maximum_short_tensor_blocks = ceil_div(maximum_short_tensor_numel * dtype_size, self.soc.block_size)

        total_short_buffer_count = self.graph.ratio * self.buffer_count[0] + self.buffer_count[1]
        estimated_short_buffer_blocks = ceil_div(self.ub_block, total_short_buffer_count)
        short_buffer_blocks = min(maximum_short_tensor_blocks, estimated_short_buffer_blocks)

        remaining_blocks = self.ub_block - short_buffer_blocks * self.buffer_count[1]
        grande_buffer_blocks = remaining_blocks // self.buffer_count[0]

        self.grande_buffer_size = grande_buffer_blocks * self.soc.block_size // dtype_size
        self.short_buffer_size = short_buffer_blocks * self.soc.block_size // dtype_size

    def dynamic_estimate(self, dtype_size: int) -> None:
        """
        estimate buffer size by ratio = grande / short
        """
        total_short_buffer_count = self.graph.ratio * self.buffer_count[0] + self.buffer_count[1]
        short_buffer_blocks = ceil_div(self.ub_block, total_short_buffer_count)
        remaining_blocks = self.ub_block - short_buffer_blocks * self.buffer_count[1]
        grande_buffer_blocks = remaining_blocks // self.buffer_count[0]

        self.grande_buffer_size = grande_buffer_blocks * self.soc.block_size // dtype_size
        self.short_buffer_size = short_buffer_blocks * self.soc.block_size // dtype_size

    def estimate(self, dtype_size: int) -> None:
        """
        calculate minimum maximum buffer needed
        estimate buffer size for different kinds of tensors
        """
        self.buffer_count = self.graph.buffer_count(self.grande_tensors, self.short_tensors, self.unique_tensors)
        # dichotomy
        self.buffer_count[0] += 1
        # transpose reduce
        if self.switches.transpose_reduce:
            self.buffer_count[0] += 1
        # align pad
        if self.switches.align_pad:
            self.buffer_count[0] += 1

        if self.switches.is_const:
            self.const_estimate(dtype_size)
        else:
            self.dynamic_estimate(dtype_size)


class CompileInfo:
    """
    Compile Info Bundle
    """

    def __init__(self, shape: List, reduce_axis: List) -> None:
        self.classify_key = self.calc_classify_key(shape, reduce_axis)
        self.pattern = self.calc_reduce_pattern(shape, reduce_axis)

    @staticmethod
    def calc_reduce_pattern(shape: List, reduce_axis: List) -> int:
        """
        calculate reduce pattern
        """
        one_hot = [0 for _ in shape]
        for i in reduce_axis:
            one_hot[i] = 1
        pattern = 0
        for v in one_hot:
            pattern = 2 * pattern + v
        pattern = 10 * pattern + len(one_hot)
        return pattern

    def calc_classify_key(self, shape: List, reduce_axis: List) -> AnyStr:
        """
        calculate classify key
        """
        pattern = self.calc_reduce_pattern(shape, reduce_axis)
        return "_" + str(pattern)

    def add_dim_var_code(self, dim_var_code) -> None:
        if "_dim_var_code" not in get_compile_info():
            add_compile_info_inner("_dim_var_code", {})
        dim_var_code_context = get_compile_info()["_dim_var_code"]
        dim_var_code_context.update({self.pattern: dim_var_code})
        add_compile_info_inner("_dim_var_code", dim_var_code_context)


def check_atomic_add_support(soc: SoC, dtype: AnyStr) -> bool:
    """
    check if current case support atomic
    """
    if soc.atomic() and dtype == "float32":
        return True
    return False


def dim_var_encode(shape: List) -> int:
    """
    To record which axis is unknown
    @param shape:
    @return:
    """
    one_hot = [1 if isinstance(thisaxis, tvm.expr.Var) else 0 for thisaxis in shape]
    binary = 0
    for v in one_hot[::-1]:
        binary = 2 * binary + v
    return binary


def min_max_dtype_size(graph: ComputeGraph):
    """
    Get maximum dtype size and minimum dtype size (in Byte)
    @param graph:
    @return:
    """
    min_dtype_size = DTYPE_BYTE_MAPPING.get(graph.outs[0].dtype)
    max_dtype_size = DTYPE_BYTE_MAPPING.get(graph.outs[0].dtype)
    for tensor in graph.tensors:
        min_dtype_size = min(min_dtype_size, DTYPE_BYTE_MAPPING.get(tensor.dtype))
        max_dtype_size = max(max_dtype_size, DTYPE_BYTE_MAPPING.get(tensor.dtype))
    return min_dtype_size, max_dtype_size


def get_reduce_axis(graph) -> List[int]:
    """
    Basic Prerequisites is this compute graph has one and only one reduce tensor (tuple reduce include)
    Get the reduce axis from graph
    @param graph:
    @return:
    """
    reduce_tensor = graph.reduce_tensor[0]
    reduce_axis_var = [axis.var for axis in reduce_tensor.op.body[0].axis]
    all_axis_var = list(reduce_tensor.op.body[0].source[0].args)
    reduce_axis = [all_axis_var.index(axis) for axis in reduce_axis_var]
    return reduce_axis


class Info:
    """
    Information Container
    """

    def __init__(self, outs: Iterable[Tensor]):
        # SOC INFORMATION
        self.soc: SoC = SoC()

        # COMPUTE GRAPH
        self.outs: Iterable[Tensor] = outs
        self.graph: ComputeGraph = ComputeGraph(outs)

        # KEY FEATURES
        self.max_shape: List = util.shape_to_list(self.graph.reduce_tensor[0].op.input_tensors[0].shape)
        self.dim_var_code: int = dim_var_encode(self.max_shape)
        self.min_dtype_size, self.max_dtype_size = min_max_dtype_size(self.graph)
        self.reduce_dtype_size: int = DTYPE_BYTE_MAPPING.get(self.graph.reduce_tensor[0].dtype)
        self.reduce_axis: List[int] = get_reduce_axis(self.graph)
        self.last_reduce: bool = False if max(self.reduce_axis) < len(self.max_shape) - 1 else True
        self.atomic_support: bool = check_atomic_add_support(self.soc, self.graph.reduce_tensor[0].dtype)
        self.atomic_threshold: int = self.soc.core_num * 64
        self.is_graph_const: bool = False if self.dim_var_code else True
        self.is_const: bool = get_compile_info().get("_is_const")

        # TILING
        self.switches: Switch = Switch(self.graph, self.is_const)
        self.switches.check()
        self.buffer_size: BufferSize = BufferSize(self.soc, Compute(self.outs), self.switches)
        self.buffer_size_list = []
        self.buffer_size.estimate(self.max_dtype_size)
        self.buffer_size_list.append(self.buffer_size.grande_buffer_size)

        self.buffer_size.switches.transpose_reduce = True
        self.buffer_size.switches.align_pad = False
        self.buffer_size.estimate(self.max_dtype_size)
        self.buffer_size_list.append(self.buffer_size.grande_buffer_size)

        self.buffer_size.switches.transpose_reduce = False
        self.buffer_size.switches.align_pad = True
        self.buffer_size.estimate(self.max_dtype_size)
        self.buffer_size_list.append(self.buffer_size.grande_buffer_size)

        # COMPILE INFO
        self.add_compile_info()
        self.compile_info: CompileInfo = CompileInfo(self.max_shape, self.reduce_axis)
        self.compile_info.add_dim_var_code(self.dim_var_code)

        # CONST TILING
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

    def add_compile_info(self) -> None:
        common_info = [
            self.soc.core_num,
            self.soc.block_size,
            self.atomic_support,
            self.atomic_threshold]
        graph_info = [
            len(self.graph.placeholder),
            self.min_dtype_size,
            self.max_dtype_size,
            self.reduce_dtype_size]
        add_compile_info_inner("_common_info", common_info)
        add_compile_info_inner("_graph_info", graph_info)
        add_compile_info_inner("_runtime", True)
        add_compile_info_inner("_buffer_size", self.buffer_size_list)

    def const_tiling(self) -> None:
        add_compile_info_inner("_runtime", False)
        inputs = [{"shape": util.shape_to_list(ph.shape), "dtype": ph.dtype} for ph in self.graph.placeholder]
        outputs = [{"shape": util.shape_to_list(tensor.shape), "dtype": tensor.dtype} for tensor in self.outs]

        run_info = op_tiling.do_op_tiling("AutoTiling", get_compile_info(), inputs, outputs)
        tiling_data_fmt = {"block_tiling_factor": "int", "ub_tiling_factor": "int"}
        tiling_data = op_tiling.decode(run_info.get("tiling_data"), tiling_data_fmt)

        self.block_factor = tiling_data.get("block_tiling_factor")
        self.ub_factor = tiling_data.get("ub_tiling_factor")
        self.tiling_key = run_info.get("tiling_key")

        add_compile_info_inner("_runtime", True)
