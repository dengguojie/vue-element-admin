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
classifier of shape in tuple reduce
"""
# Standard Packages
from typing import Dict
from typing import List
from typing import Tuple
from typing import AnyStr
from functools import reduce
from itertools import groupby
from itertools import product
from itertools import combinations
import copy
# Ascend Packages
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_context
from . import util

# Constants
TUPLE_REDUCE = "tuple_reduce"
MAX_DIM_LEN = 8


def _raise_error(message: AnyStr):
    """
    Raise error
    @param message:
    @return:
    """
    dict_args = {"errCode": "E90001", "detailed_cause": message}
    raise RuntimeError(dict_args, get_error_message(dict_args))


def _binary_mode_detection(ins: List) -> List[List]:
    """
    Generate (-1, ...) if find -2
    @param ins:
    @return:
    """
    inputs = [list(tensor.get('shape')) for tensor in ins]
    for idx, input_shape in enumerate(inputs):
        if -2 in input_shape:
            inputs[idx] = [-1 for _ in range(MAX_DIM_LEN)]
    return inputs


def _get_broadcast_axes(inputs, extra_params):
    """
    Get broadcast axes from extra_params
    @param inputs:
    @param extra_params:
    @return:
    """
    broadcast_axes = [[] for _ in inputs]
    if "compile_broadcast_axis" not in extra_params:
        return broadcast_axes
    
    compile_broadcast_axis = extra_params.get("compile_broadcast_axis")
    for k, v in compile_broadcast_axis.items():
        shape_length = len(inputs[k].get("shape"))
        v = [value + shape_length if value < 0 else value for value in v]
        broadcast_axes[k] = v
    
    return broadcast_axes


def classify(ins: List, extra_params: Dict) -> List[List]:
    """
    classify
    This module has the following steps:
    1. Make the shape length consistent by adding one in front of the shape.
    2. Merge consecutive axes of the same type according to the reduce and broadcast axes.
    3. For the last unknown axis of non-reduce non-broadcast type, generate a scenario where the axis is 1.
    @param ins: inputs list, the last element defaults to the reduce_axis.
    @param extra_params:
        "compile_broadcast_axis": dict, key is input index, value is compile broadcast axes.
        "disable_fuse_axes": axes that cannot be fused with others.
    @return: list of all possible inputs scenarios.
    """
    # ParamCheck
    if extra_params is not None and not isinstance(extra_params, Dict):
        _raise_error("extra_params must be a dict or None when mode is {mode}".format(mode=TUPLE_REDUCE))
    if extra_params is None:
        extra_params = {}
    if len(ins) < 2 or not isinstance(ins[-1], (List, Tuple)):
        _raise_error("The last element in the {mode} classify must be a List or Tuple"
                     "which is reduce_axis".format(mode=TUPLE_REDUCE))
    
    # Check disable_fuse_axes
    disable_fuse_axes = []
    if "disable_fuse_axes" in extra_params:
        disable_fuse_axes = extra_params.get("disable_fuse_axes")
    add_compile_info_inner("_disable_fuse_axes", disable_fuse_axes[:])

    # Get reduce axis from ins
    _ins = copy.deepcopy(ins)
    reduce_axis = _ins.pop()
    add_compile_info_inner("_reduce_axis", reduce_axis[:])
    
    # Get broadcast axis from extra_params
    broadcast_axes = _get_broadcast_axes(_ins, extra_params)

    # Adjust shape length
    inputs = _binary_mode_detection(_ins)
    max_shape_len = len(max(inputs, key=lambda i: len(i)))
    shapes_length = []
    for idx, input_shape in enumerate(inputs):
        delta = max_shape_len - len(input_shape)
        shapes_length.append(len(input_shape))
        broadcast_axis = broadcast_axes[idx]
        if broadcast_axis:
            broadcast_axes[idx] = list(range(delta)) + [x + delta for x in broadcast_axis]
        inputs[idx] = [1] * delta + input_shape
    add_compile_info_inner("_shapes_length", shapes_length)
    add_compile_info_inner("_max_shape_len", max(shapes_length))

    # Deduce from broadcast axes
    for input_idx, broadcast_axis in enumerate(broadcast_axes):
        for axis_idx in broadcast_axis:
            inputs[input_idx][axis_idx] = 1
    
    # Construct instance
    instance = TupleReduceClassifier(inputs, broadcast_axes, reduce_axis, disable_fuse_axes, ins)
    return instance.classify(extra_params)


class TupleReduceClassifier:
    """
    Tuple Reduce Classifier
    """

    def __init__(self, inputs, broadcast_axes, reduce_axis, disable_fuse_axes, ins):
        self.inputs = inputs
        self.broadcast_axes = broadcast_axes
        self.reduce_axis = reduce_axis
        self.disable_fuse_axes = disable_fuse_axes
        self.ins = ins
        self.classify_outs = []
    
    def classify(self, extra_params):
        self.onehot_encode()
        self.fuse_axes()
        self.add_compile_info()
        self.deduce()
        self.tail_optimization()
        return self.classify_outs
    
    def onehot_encode(self):
        reduce_code = [1 if i in self.reduce_axis else 0 for i, _ in enumerate(self.inputs[0])]
        disable_fuse_code = [i if i in self.disable_fuse_axes else 0 for i, _ in enumerate(self.inputs[0])]
        broadcast_code = [-1 for _ in reduce_code]
        for broadcast_axis in self.broadcast_axes:
            for axis_index in broadcast_axis:
                broadcast_code[axis_index] += 1
        for i, v in enumerate(broadcast_code):
            if v >= sum(len(val) > 0 for val in self.broadcast_axes):
                broadcast_code[i] = 1
            elif v > 0:
                broadcast_code[i] = 2 + i
            else:
                broadcast_code[i] = 0
        
        fusible_code = [int(''.join(map(str, [k, j, i]))) for i, j, k in
                        zip(reduce_code, broadcast_code, disable_fuse_code)]
        
        self.reduce_code = reduce_code
        self.broadcast_code = broadcast_code
        self.disable_fuse_code = disable_fuse_code
        self.fusible_code = fusible_code
    
    def fuse_axes(self):
        fuse_rules = [(k, list(g)) for k, g in groupby(tuple(enumerate(self.fusible_code)), lambda i: i[1])]
        self.fused_reduce_axis, self.fused_broadcast_axis, self.fused_disable_fuse_axes = [], [], []
        for idx, (k, g) in enumerate(fuse_rules):
            if k > 99:
                self.fused_disable_fuse_axes.append(idx)
            if k % 10 == 1:
                self.fused_reduce_axis.append(idx)
            if (k // 10) % 10 >= 1:
                self.fused_broadcast_axis.append(idx)
        
        self.fused_shapes = [[] for _ in self.inputs]
        for idx, _shape in enumerate(self.inputs):
            fused_shape = []
            for k, g in fuse_rules:
                group_index = [_pair[0] for _pair in g]
                group_values = [_shape[_idx] for _idx in group_index]
                value = -1 if -1 in group_values else reduce((lambda x, y: x * y), group_values)
                fused_shape.append(value)
            self.fused_shapes[idx] = fused_shape
    
    def add_compile_info(self):
        add_compile_info_inner("_fusible_code", self.fusible_code[:])
        add_compile_info_inner("_fused_reduce_axis", self.fused_reduce_axis[:])
        add_compile_info_inner("_fused_broadcast_axis", self.fused_broadcast_axis[:])
        add_compile_info_inner("_fused_disable_fuse_axes", self.fused_disable_fuse_axes[:])
    
    def deduce(self):
        inputs_num = len(self.fused_shapes)
        axis_num = len(self.fused_shapes[0])
        for j in range(axis_num):
            if j in self.fused_broadcast_axis:
                continue
            axes = [_shape[j] for _shape in self.fused_shapes]
            if len(set(axes)) > 2:
                _raise_error("The inputs might not meet the constraints of Tuple-Reduce Template")
            max_value = max(axes)
            for i in range(inputs_num):
                self.fused_shapes[i][j] = max_value
    
    def tail_optimization(self):
        # basic version
        _ins = copy.deepcopy(self.ins)
        _ins[-1] = self.fused_reduce_axis
        for i, v in enumerate(self.fused_shapes):
            _ins[i].update({"shape": v})
            _ins[i].update({"range": [(1, None) if value == -1 else (value, value) for value in v]})
        self.classify_outs.append(_ins)

        # optimized version
        if -1 not in [_shape[-1] for _shape in self.fused_shapes]:
            return
