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
provides the function of generating the tiling key for tik optimization
"""

import hashlib


def _sort_unique_key_params(params):
    """
    sort_unique_key_params
    """
    if not isinstance(params, dict):
        raise RuntimeError("params must be dict, but get %s" % params)

    sort_params_keys = list(params.keys())
    sort_params_keys.sort()
    new_params = {}
    for k in sort_params_keys:
        val = params.get(k)
        if isinstance(val, list):
            val = tuple(val)
        new_params[k] = val
    return new_params


def _clear_unique_key_params(params):
    """
    Delete unnecessary keys
    """
    allow_key = [
        "index",
        "dtype",
        "ori_format",
        "ori_shape",
        "format",
        "shape",
        "slice_offset",
        "split_index",
        "total_shape",
        "valid_shape",
        "l1_addr_offset",
        "l1_fusion_type",
        "l1_workspace_size",
        "addr_type"]

    if params:
        new_params = []
        for param in params:
            new_param = {}
            for k in param:
                if k.lower() in allow_key:
                    new_param[k] = param.get(k)
            new_param = _sort_unique_key_params(new_param)
            new_params.append(new_param)
        return new_params


def _check_unique_key_params(name, params):
    """
    check get_op_compile_unique_key func params
    """
    if not isinstance(params, (list, tuple)) and params:
        raise RuntimeError("The %s must be list or tuple, but get %s" % (name, type(params)))


def get_op_compile_unique_key(op_type, inputs, outputs, attrs, extra_params):
    """
    get_op_compile_unique_key
    """
    if isinstance(op_type, str):
        op_type = op_type.lower()
    else:
        raise RuntimeError("The op_type must be str, but get %s" % type(op_type))

    _check_unique_key_params("inputs", inputs)
    _check_unique_key_params("outputs", outputs)
    _check_unique_key_params("extra_params", extra_params)

    op_compile_params = {}

    op_inputs = _clear_unique_key_params(inputs)
    op_outputs = _clear_unique_key_params(outputs)

    if op_inputs:
        op_compile_params["inputs"] = op_inputs

    if outputs:
        op_compile_params["outputs"] = op_outputs

    if attrs:
        new_attrs = []
        for attr in attrs:
            new_attr = _sort_unique_key_params(attr)
            new_attrs.append(new_attr)
        op_compile_params["attrs"] = new_attrs

    if extra_params:
        extra_params.sort()
        op_compile_params["extra_params"] = extra_params

    op_compile_params = str(op_compile_params)
    op_compile_params_sha = hashlib.sha256(op_compile_params.encode("utf-8")).hexdigest()

    return "%s_%s" % (op_type, op_compile_params_sha)
