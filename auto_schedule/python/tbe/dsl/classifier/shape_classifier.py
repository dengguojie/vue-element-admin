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
classifier of shape
"""
from typing import Any
from typing import Dict
from typing import Optional

from tbe.common.register import operation_func_mgr
from tbe.dsl.base import expr_compare
from tbe.dsl.base.classifier import conv2d_classifier
from tbe.dsl.base.classifier import softmax_norm_classifier

from . import concat_classifier
from . import elewise_classifier
from . import gather_classifier
from . import norm_classifier
from . import slice_classifier
from . import split_classifier
from . import transpose_classifier
from . import tuple_reduce_classifier
from .reduce import reduce_classifier
from .transdata import transdata_classifier

ELEWISE = "elewise"
BROADCAST = "broadcast"
NORM = "norm"
REDUCE = "reduce"
SOFTMAX_NORM = "softmax_norm"
GATHER = "gather"
GATHER_ND = "gather_nd"
SLICE = "slice"
TRANSPOSE = "transpose"
CONCAT = "concat"
TRANSDATA = "transdata"
SPLIT = "split"
TUPLE_REDUCE = "tuple_reduce"
CONV2D = "Convolution"


CLASSIFY_SAME_PATTERN_MAP = {
    "ElemWise": ELEWISE,
    "Broadcast": BROADCAST,
    "CommReduce": REDUCE,
    "SoftmaxNorm": SOFTMAX_NORM
}


@operation_func_mgr.register_classify_processor("Convolution", support_type="all")
def classify(ins: list, mode: str, extra_params: Optional[Dict[str, Any]] = None):
    """
    classify
    :param ins:
    :param mode:
    :param extra_params: must include keepdims when mode is reduce
    :return:
    """
    mode = CLASSIFY_SAME_PATTERN_MAP.get(mode, mode)
    if mode == ELEWISE:
        expr_compare.is_true(extra_params is None or "disable_optimization" not in extra_params,
                             {"errCode": "E90001",
                              "detailed_cause": "inputs of classify not support the dict extra_params with "
                                                "the key disable_optimization when mode is ELEWISE"
                             })
        return elewise_classifier.classify(ins, support_broadcast=False, extra_params=extra_params)
    if mode == BROADCAST:
        return elewise_classifier.classify(ins, support_broadcast=True, extra_params=extra_params)
    if mode == REDUCE:
        return reduce_classifier.classify(ins, extra_params)
    if mode == SOFTMAX_NORM:
        return softmax_norm_classifier.classify(ins, support_reduce=True, extra_params=extra_params)
    if mode == NORM:
        return norm_classifier.classify(ins, extra_params)
    if mode == GATHER:
        return gather_classifier.classify_gather(ins)
    if mode == GATHER_ND:
        return gather_classifier.classify_gather_nd(ins)
    if mode == SLICE:
        return slice_classifier.classify_slice(ins, extra_params)
    if mode == TRANSPOSE:
        return transpose_classifier.classify(ins, extra_params)
    if mode == CONCAT:
        return concat_classifier.classify(ins, extra_params)
    if mode == SPLIT:
        return split_classifier.classify(ins, extra_params)
    if mode == TRANSDATA:
        return transdata_classifier.classify(ins)
    if mode == TUPLE_REDUCE:
        return tuple_reduce_classifier.classify(ins, extra_params)
    if mode == CONV2D:
        return conv2d_classifier.classify(ins, extra_params)

    return [ins]
