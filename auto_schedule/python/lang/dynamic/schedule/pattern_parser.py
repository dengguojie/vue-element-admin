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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""
from enum import Enum, auto  # pylint: disable=E0611

from te import tvm
from te.lang.base import operation

from . import Pattern
from . import util
from .reduce_schedule import ReduceSchedule

ELEWISE_COMPUTE = {
    "elewise_binary_add", "elewise_binary_sub", "elewise_binary_div",
    "elewise_binary_mul", "elewise_binary_min", "elewise_binary_max",
    "elewise_binary_and", "elewise_binary_or", "elewise_binary_vcmpv_le",
    "elewise_binary_vcmpv_lt", "elewise_binary_vcmpv_ge",
    "elewise_binary_vcmpv_gt", "elewise_binary_vcmpv_ne",
    "elewise_binary_vcmpv_eq", "emit_insn_elewise_binary_cmp",
    "elewise_binary_logic", "elewise_single_log", "elewise_single_exp",
    "elewise_single_rec", "elewise_single_VS_add", "elewise_single_VS_mul",
    "elewise_single_VS_max", "elewise_single_VS_min", "elewise_single_abs",
    "elewise_single_relu", "elewise_single_not", "elewise_single_sqrt",
    "elewise_single_rsqrt", "elewise_multiple_mla",
    "elewise_multiple_madd", "elewise_multiple_maddrelu",
    "elewise_multiple_sel", "elewise_binary_scalar_axpy",
    "elewise_binary_cmpsel_gt", "elewise_binary_cmpsel_ge",
    "elewise_binary_cmpsel_lt", "elewise_binary_cmpsel_le",
    "elewise_binary_cmpsel_eq", "elewise_binary_cmpsel_ne",
    "elewise_binary_vcmpv_gt", "elewise_binary_vcmpv_ge",
    "elewise_binary_vcmpv_lt", "elewise_binary_vcmpv_le",
    "elewise_binary_vcmpv_eq", "elewise_binary_vcmpv_ne",
}

CAST_COMPUTE = {
    "elewise_single_cast", "elewise_single_ceil", "elewise_single_floor",
    "elewise_single_trunc", "elewise_single_round", "elewise_single_round_d",
}

BROADCAST_COMPUTE = {
    "unified_broadcast", "broadcast", "unknown_broadcast"
}

REDUCE_COMPUTE = {
    "reduce_min", "reduce_max", "reduce_sum",
    "reduce_prod", "tuple_reduce_sum",
}

CONV2D_COMPUTE = {
    "conv_vector_remove_pad",
    "convolution_C"
}

CONV2D_BP_INPUT_COMPUTE = {
    "conv2d_backprop_input",
    "conv2d_backprop_input_opti"
}

CONV2D_BP_FILTER_COMPUTE = {
    "conv2d_backprop_filterdw_ddr"
}


class ComputeType(Enum):
    """
    ComputeType
    """
    ANY = auto()
    UNKNOWN = auto()
    PLACEHOLDER = auto()
    ELEWISE = auto()
    BROADCAST = auto()
    REDUCE = auto()
    CAST = auto()
    CONV2D = auto()
    CONV2D_BP_INPUT = auto()
    CONV2D_BP_FILTER = auto()


def get_pattern(outs):
    """
    :param outs:
    :return:
    """
    pattern = operation.get_context().get_pattern()
    if pattern is None:
        return _parse_pattern(outs)
    if callable(pattern):
        return pattern(outs)
    return pattern


def _parse_pattern(outs):
    # key: compute type, @see enum(ComputeType)
    # value: size of the special compute type
    # the "any" key means total of compute
    compute_type_size_map = _dfs_compute(outs)

    if ComputeType.CONV2D in compute_type_size_map:
        return Pattern.CONV2D
    if ComputeType.CONV2D_BP_INPUT in compute_type_size_map:
        return Pattern.CONV2D_BACKPROP_INPUT
    if ComputeType.CONV2D_BP_FILTER in compute_type_size_map:
        return Pattern.CONV2D_BACKPROP_FILTER
    if _is_elewise(compute_type_size_map):
        return Pattern.ELEMWISE
    if _is_reduce(compute_type_size_map):
        return Pattern.REDUCE

    return Pattern.OPAQUE


def _is_elewise(compute_type_size_map: dict):
    ph_size = compute_type_size_map.get(ComputeType.PLACEHOLDER, 0)
    elewise_size = compute_type_size_map.get(ComputeType.ELEWISE, 0)
    broadcast_size = compute_type_size_map.get(ComputeType.BROADCAST, 0)
    cast_size = compute_type_size_map.get(ComputeType.CAST, 0)
    total = compute_type_size_map.get(ComputeType.ANY, 0)
    return ph_size + elewise_size + broadcast_size + cast_size == total


def _is_reduce(compute_type_size_map):
    placeholder_size = compute_type_size_map.get(ComputeType.PLACEHOLDER, 0)
    elewise_size = compute_type_size_map.get(ComputeType.ELEWISE, 0)
    cast_size = compute_type_size_map.get(ComputeType.CAST, 0)
    reduce_size = compute_type_size_map.get(ComputeType.REDUCE, 0)
    total = compute_type_size_map.get(ComputeType.ANY, 0)
    return placeholder_size + elewise_size + reduce_size + cast_size == total


def _dfs_compute(outs) -> dict:
    outs = list(outs) if isinstance(outs, (tuple, list)) else [outs]
    visited = set()
    compute_type_size_map = {}
    for out in outs:
        _dfs_compute_inner(out, visited, compute_type_size_map)
    return compute_type_size_map


def _dfs_compute_inner(tensor: tvm.tensor.Tensor, visited: set,
                       compute_type_size_map: dict):
    if tensor in visited:
        return
    visited.add(tensor)

    compute_type = _get_compute_type(tensor)
    compute_type_size_map[compute_type] = compute_type_size_map.get(
        compute_type, 0) + 1
    compute_type_size_map[ComputeType.ANY] = compute_type_size_map.get(
        ComputeType.ANY, 0) + 1

    for tensor_i in tensor.op.input_tensors:
        _dfs_compute_inner(tensor_i, visited, compute_type_size_map)


def _get_compute_type(tensor: tvm.tensor.Tensor) -> ComputeType:
    if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
        return ComputeType.PLACEHOLDER

    tag = tensor.op.tag
    if tag is None:
        return ComputeType.UNKNOWN

    insn = util.get_dsl_insn(tensor)
    insn_compute_type_mapping = [
        (ELEWISE_COMPUTE, ComputeType.ELEWISE),
        (BROADCAST_COMPUTE, ComputeType.BROADCAST),
        (CAST_COMPUTE, ComputeType.CAST),
        (REDUCE_COMPUTE, ComputeType.REDUCE),
        (CONV2D_COMPUTE, ComputeType.CONV2D),
        (CONV2D_BP_INPUT_COMPUTE, ComputeType.CONV2D_BP_INPUT),
        (CONV2D_BP_FILTER_COMPUTE, ComputeType.CONV2D_BP_FILTER)
    ]
    for insns, compute_type in insn_compute_type_mapping:
        if insn in insns:
            return compute_type

    return ComputeType.UNKNOWN
