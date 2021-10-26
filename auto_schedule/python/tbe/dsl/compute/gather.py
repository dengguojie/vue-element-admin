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
gather compute
"""

from tbe import tvm
from tbe.common.testing.dsl_source_info import source_info_decorator
from .util import dtype_check_decorator

NAME_INDEX = [0]


@source_info_decorator()
@dtype_check_decorator
def gather(params, indices, axis=None, batch_dims=0):
    """
    :param params:
    :param indices:
    :param axis:
    :param batch_dims:
    :return:
    """
    p_shape = params.shape
    i_shape = indices.shape
    p_name = params.name
    i_name = indices.name
    i_dtype = indices.dtype

    # check indices dytpe
    _check_gather_indices_dtype(i_dtype)

    g_shape = p_shape[:axis] + i_shape[batch_dims:] + p_shape[axis + 1:]

    def index(i):
        end = axis + len(i_shape) - batch_dims
        idx_pos = i[:batch_dims] + i[axis:end]
        return list(i[:axis]) + [indices[idx_pos]] + list(i[end:])

    op_name = "gather"
    compute_name = op_name + _get_next_name_index()
    with tvm.tag_scope(op_name):
        g_tensor = tvm.compute(g_shape, lambda *i: params(*index(i)), name=compute_name,
                               attrs={"params_name": p_name, "indices_name": i_name})

    return g_tensor


@source_info_decorator()
@dtype_check_decorator
def gather_nd(params, indices, batch_dims=0):
    """
    :param params:
    :param indices:
    :param batch_dims:
    :return:
    """
    p_shape = params.shape
    i_shape = indices.shape
    p_name = params.name
    i_name = indices.name
    i_dtype = indices.dtype

    # check indices dytpe
    _check_gather_indices_dtype(i_dtype)

    rank_value = i_shape[-1].value if isinstance(i_shape[-1], tvm.expr.ConstExpr) else i_shape[-1]

    g_shape = i_shape[:-1] + p_shape[batch_dims + rank_value:]

    def index(i):
        p_pos = []
        p_pos.extend(i[:batch_dims])
        axis = len(i_shape) - 1
        for j in range(rank_value):
            p_pos.append(indices(*i[:axis], j))
        p_pos.extend(i[axis:])
        return p_pos

    op_name = "gather_nd"
    compute_name = op_name + _get_next_name_index()
    with tvm.tag_scope(op_name):
        g_tensor = tvm.compute(g_shape, lambda *i: params(*index(i)), name=compute_name,
                               attrs={"params_name": p_name, "indices_name": i_name})

    return g_tensor


def _check_gather_indices_dtype(i_dtype):
    """
    :param i_dtype: indices dtype str
    :return:
    """
    if i_dtype not in ("int32, int64"):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "indices tensors dtype must be int32 or int64! " \
                                      "while dtype is [%s], " % (i_dtype,)
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _get_next_name_index():
    """
    :return: next name index as string
    """
    NAME_INDEX[0] += 1
    return str(NAME_INDEX[0])
