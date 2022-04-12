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
tuple reduce variable shape
"""
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("tuple_reduce")
def variable_shape(inputs):
    # type: (list) -> list
    """
    variable shape for tuple_reduce ops
    :param inputs: all inputs
    :return:
    """
    shapes = [tensor.get("shape") for tensor in inputs]
    shapes_transpose = list(map(list, zip(*shapes)))
    tuple_reduce_vars = []
    for j, col in enumerate(shapes_transpose):
        if -1 in col:
            tuple_reduce_vars.append(operation.var_inner("_dim_{}".format(j), (1, None)))
        else:
            tuple_reduce_vars.append(None)
    res = []
    for _shape in shapes:
        single_shape = [tuple_reduce_vars[j] if v == -1 else v for j, v in enumerate(_shape)]
        res.append(single_shape)
    return res
