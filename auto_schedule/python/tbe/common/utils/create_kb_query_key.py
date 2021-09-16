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
from tbe.common.utils import log


def _check_unique_key_params(name, params):
    """
    check get_op_compile_unique_key func params
    """
    if not isinstance(params, (list, tuple)) and params:
        raise RuntimeError("The get_op_compile_unique_key api param %s must be list or tuple,"
                           " but get %s" % (name, type(params)))


def _sort_unique_key_params(params):
    """
    sort_unique_key_params
    """
    if not isinstance(params, dict):
        log.warn("The get_op_compile_unique_key api param must be nest dict, but get %s" % params)
        return params

    new_params = {}
    for k in sorted(params.keys()):
        new_params[k] = params.get(k)
    return new_params


def _get_new_inputs_outputs(name, params):
    """
    Delete unnecessary keys
    """
    if not params:
        log.warn("The get_op_compile_unique_key api param %s is empty.", name)
        return params

    new_params = []
    for param in params:
        # param may be None, (), {}, at this param no process.
        if not param:
            new_params.append(param)
        # param is dict, need sort the param
        elif isinstance(param, dict):
            new_param = _sort_unique_key_params(param)
            new_params.append(new_param)
        # param is list or tuple, get nest dict
        elif isinstance(param, (list, tuple)):
            new_param = []
            for param_i in param:
                # param_i must be dict, need sort the param_i
                if isinstance(param_i, dict):
                    new_param_i = _sort_unique_key_params(param_i)
                    new_param.append(new_param_i)
                else:
                    log.warn("The get_op_compile_unique_key api param %s, must be (dict, (dict, ...))", name)
            new_params.append(new_param)
        else:
            log.warn("The get_op_compile_unique_key api param %s,"
                     " must be (dict, dict,...) or (dict, (dict, ...))", name)
            return []
    return new_params


def _get_new_attrs(attrs):
    new_attrs = []
    # attrs must be list or tuple.
    for attr in attrs:
        # attr must be dict, need sort attr.
        if isinstance(attr, dict):
            new_attr = _sort_unique_key_params(attr)
            new_attrs.append(new_attr)
        else:
            log.warn("The get_op_compile_unique_key api param attrs, must be (dict, dict,...)")
    return new_attrs


def _get_new_extra_params(extra_params):
    new_extra_params = {}
    # extra_params must be dict, need sort the extra_params.
    for k in sorted(extra_params.keys()):
        v = extra_params.get(k)
        if isinstance(v, dict):
            new_v = _sort_unique_key_params(v)
            new_extra_params[k] = new_v
        else:
            new_extra_params[k] = v
    return new_extra_params


def get_op_compile_unique_key(op_type, inputs, outputs, attrs, extra_params):
    """
    get_op_compile_unique_key
    """
    if isinstance(op_type, str):
        op_type = op_type.lower()
    else:
        raise RuntimeError("The get_op_compile_unique_key api param op_type must be str, but get %s" % type(op_type))

    op_compile_params = {}
    if inputs:
        _check_unique_key_params("inputs", inputs)
        op_compile_params["inputs"] = _get_new_inputs_outputs("inputs", inputs)
    if outputs:
        _check_unique_key_params("outputs", outputs)
        op_compile_params["outputs"] = _get_new_inputs_outputs("outputs", outputs)
    if attrs:
        _check_unique_key_params("attrs", attrs)
        op_compile_params["attrs"] = _get_new_attrs(attrs)
    if extra_params and isinstance(extra_params, dict):
        op_compile_params["extra_params"] = _get_new_extra_params(extra_params)

    op_compile_params = str(op_compile_params)
    log.debug("kb_query_key_params_str: %s:%s", op_type, op_compile_params)
    op_compile_params_sha = hashlib.sha256(op_compile_params.encode("utf-8")).hexdigest()

    return "%s_%s" % (op_type, op_compile_params_sha)
