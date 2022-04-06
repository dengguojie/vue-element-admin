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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""
from typing import Tuple

from tbe import tvm
from tbe.common.register import get_operator
from tbe.dsl.base import operation
from tbe.dsl.unify_schedule.constants import COMPUTE_TYPE_INSN_MAPPING
from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.pattern_manager import parse

from . import Pattern
from . import util


def _get_custom_pattern():
    """
    get custom pattern
    :return:
    """
    pattern = None
    op_type = operation.get_context().get_op_type()
    if op_type and get_operator(op_type):
        pattern = get_operator(op_type).get_pattern()

    return pattern


def get_pattern(outs):
    """
    :param outs:

    """
    pattern = _get_custom_pattern()
    if pattern is None:
        current_pattern = operation.get_context().get_pattern()
        if current_pattern:
            return current_pattern
        return _parse_pattern(outs)
    if callable(pattern):
        return pattern(outs)
    return pattern


def _parse_pattern(outs):
    # compute_type_size_map
    # key: compute type, @see enum(ComputeType)
    # value: size of the special compute type
    # the "any" key means total of compute
    # compute_type_tensor_map
    # key: compute type, @see enum(ComputeType)
    # value: the special compute type tensor
    compute_type_size_map, compute_type_tensor_map = _dfs_compute(outs)

    if ComputeType.CONV3D in compute_type_size_map:
        return Pattern.CONV3D
    if ComputeType.CONV2D in compute_type_size_map:
        return Pattern.CONV2D
    if ComputeType.CONV2D_BP_INPUT in compute_type_size_map:
        return Pattern.CONV2D_BACKPROP_INPUT
    if ComputeType.CONV2D_BP_FILTER in compute_type_size_map:
        return Pattern.CONV2D_BACKPROP_FILTER
    if ComputeType.CONV3D_BP_INPUT in compute_type_size_map:
        return Pattern.CONV3D_BACKPROP_INPUT
    if ComputeType.MAT_MUL in compute_type_size_map:
        return Pattern.MAT_MUL

    pattern = parse(outs, compute_type_size_map, compute_type_tensor_map)
    if pattern is not None:
        return pattern

    if ComputeType.CONV3D_BP_FILTER in compute_type_size_map:
        return Pattern.CONV3D_BACKPROP_FILTER

    return Pattern.OPAQUE


def _dfs_compute(outs) -> Tuple[dict, dict]:
    outs = list(outs) if isinstance(outs, (tuple, list)) else [outs]
    visited = set()
    compute_type_size_map = {}
    compute_type_tensor_map = {}
    for out in outs:
        _dfs_compute_inner(out, visited, compute_type_size_map, compute_type_tensor_map)
    return compute_type_size_map, compute_type_tensor_map


def _dfs_compute_inner(tensor: tvm.tensor.Tensor, visited: set,
                       compute_type_size_map: dict, compute_type_tensor_map: dict):
    if tensor in visited:
        return
    visited.add(tensor)

    compute_type = _get_compute_type(tensor)
    compute_type_size_map[compute_type] = compute_type_size_map.get(
        compute_type, 0) + 1
    if compute_type not in compute_type_tensor_map:
        compute_type_tensor_map[compute_type] = []
    compute_type_tensor_map[compute_type].append(tensor)

    compute_type_size_map[ComputeType.ANY] = compute_type_size_map.get(
        ComputeType.ANY, 0) + 1
    if ComputeType.ANY not in compute_type_tensor_map:
        compute_type_tensor_map[ComputeType.ANY] = []
    compute_type_tensor_map[ComputeType.ANY].append(tensor)

    for tensor_i in tensor.op.input_tensors:
        _dfs_compute_inner(tensor_i, visited, compute_type_size_map, compute_type_tensor_map)


def _get_compute_type(tensor: tvm.tensor.Tensor) -> ComputeType:
    if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
        return ComputeType.PLACEHOLDER

    tag = tensor.op.tag
    if tag is None:
        return ComputeType.UNKNOWN

    insn = util.get_dsl_insn(tensor)
    for compute_type, insns in COMPUTE_TYPE_INSN_MAPPING.items():
        if insn in insns:
            return compute_type

    return ComputeType.UNKNOWN
