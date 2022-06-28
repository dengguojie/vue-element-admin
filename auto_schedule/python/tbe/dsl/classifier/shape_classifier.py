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
import functools
from typing import Any
from typing import Dict
from typing import Optional

from tbe.common.register import operation_func_mgr
from tbe.dsl.base.classifier import conv2d_classifier
from tbe.dsl.base.classifier import softmax_norm_classifier

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

_classifiers = {}


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
    if mode == SOFTMAX_NORM:
        return softmax_norm_classifier.classify(ins, support_reduce=True, extra_params=extra_params)
    if mode == CONV2D:
        return conv2d_classifier.classify(ins, extra_params)

    classifier_func = _get_classifier(mode)
    if classifier_func is not None:
        return classifier_func(ins, extra_params)

    return [ins]


def _get_classifier(mode):
    # type: (str) -> callable
    return _classifiers.get(mode)


def register_classifier(mode):
    # type: (str) -> callable
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _classifiers[mode] = wrapper
        return wrapper

    return decorator
