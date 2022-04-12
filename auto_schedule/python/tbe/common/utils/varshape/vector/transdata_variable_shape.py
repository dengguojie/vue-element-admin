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
transdata variable shape
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation
from tbe.tvm import api as tvm


@register_variable("transdata")
def variable_shape(inputs):
    # type: (list) -> list
    """
    Eg:
    src is [30, 1024]
    dst is [64, 32, 16]
    map is {0: [1, ],
            1: [0, 2]}
    In map has three model: int:int, tuple:int, int:tuple
    int:int is do nothing
    tuple:int is do de-pad
    int:tuple is do pad
    Return src_shape, dst_shape
    """
    if len(inputs) != 3:
        dict_args = {"errCode": "E90001", "detailed_cause": "inputs' size error"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def is_const_model(_input):
        return -1 not in _input

    def reversed_map(_map):
        result = {}
        for key, value in _map.items():
            value = tuple(value) if isinstance(value, list) else value
            key = list(key) if isinstance(key, tuple) else key
            result[value] = key
        return result

    def init_shape(_input):
        if is_const:
            return _input

        result = []
        for k, v in enumerate(_input):
            # set pad-dim-n as var but not 1 while dynamic
            result.append(v if v != -1 and [k, v] != [0, 1] else operation.var_inner(f"_dim_{k}", [1, None]))
        return result

    def src_infer_dst(in_shape, out_shape, _map):
        for key, value in enumerate(in_shape):
            map_value = _map.get(key)
            if isinstance(map_value, int):
                out_shape[map_value] = value
            elif isinstance(map_value, (list, tuple)):
                if len(map_value) == 1:
                    out_shape[map_value[0]] = tvm.floordiv(value + pad_factor - 1, pad_factor) * pad_factor
                else:
                    out_shape[map_value[0]] = tvm.floordiv(value + pad_factor - 1, pad_factor)
                    out_shape[map_value[1]] = pad_factor

        return out_shape

    def infer_shape_main(_input, _output, _map):
        _input = init_shape(_input)
        _output = src_infer_dst(_input, _output, _map)
        return _input, _output

    axes_map = inputs[2]
    is_forward = inputs[0].get("is_forward")
    src_shape = inputs[0].get("shape")
    dst_shape = inputs[1]

    if not is_forward:
        src_shape, dst_shape = dst_shape, src_shape
        axes_map = reversed_map(axes_map)

    current_compute = operation.get_context().get_current_compute()
    # transdata_category that help to choose different computation.
    # 32bit-Tensor would be reinterpret as 16bit-Tensor in schedule(ori_bit).
    pad_factor = operation.get_compile_info().get("_pad_factor")
    is_const = is_const_model(src_shape)
    current_compute.add("_pad_factor", pad_factor)
    current_compute.add("_const_model", is_const)
    current_compute.add("_transdata_category", inputs[0].get("transdata_category", None))
    current_compute.add("_ori_bit", inputs[0].get("ori_bit", None))

    src_shape, dst_shape = infer_shape_main(src_shape, dst_shape, axes_map)
    if not is_forward:
        src_shape, dst_shape = dst_shape, src_shape

    return src_shape, dst_shape
