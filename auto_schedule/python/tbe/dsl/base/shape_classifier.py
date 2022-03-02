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

from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.classifier import classify_elewise
from tbe.dsl.base.classifier import classify_norm
from tbe.dsl.base.classifier import classify_reduction
from tbe.dsl.base.classifier import classify_softmax_cross_entropy_with_logits
from tbe.dsl.base.classifier import classify_gather
from tbe.dsl.base.classifier import classify_gather_nd
from tbe.dsl.base.classifier import classify_slice
from tbe.dsl.base.classifier import classify_transpose
from tbe.dsl.base.classifier import classify_concat
from tbe.dsl.base.classifier import classify_split
from tbe.dsl.base.classifier import classify_transdata
from .expr_compare import is_true


ELEWISE = "elewise"
BROADCAST = "broadcast"
NORM = "norm"
REDUCE = "reduce"
SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_WITH_REDUCE = "softmax_cross_entropy_with_logits_with_reduce"
GATHER = "gather"
GATHER_ND = "gather_nd"
SLICE = "slice"
TRANSPOSE = "transpose"
CONCAT = "concat"
TRANSDATA = "transdata"
SPLIT = "split"


CLASSIFY_SAME_PATTERN_MAP = {
    "ElemWise": ELEWISE,
    "Broadcast": BROADCAST,
    "CommReduce": REDUCE,
    "SoftmaxCrossEntropyWithLogitsWithReduce": SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_WITH_REDUCE
}


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
        is_true(extra_params is None or "disable_optimization" not in extra_params,
                {"errCode": "E90001",
                 "detailed_cause": "inputs of classify not support the dict extra_params with "\
                                           "the key disable_optimization when mode is ELEWISE"})
        return classify_elewise(ins, support_broadcast=False, extra_params=extra_params)
    if mode == BROADCAST:
        return classify_elewise(ins, support_broadcast=True, extra_params=extra_params)
    if mode == REDUCE:
        return classify_reduction(ins, extra_params)
    if mode == SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_WITH_REDUCE:
        return classify_softmax_cross_entropy_with_logits(ins, support_reduce=True)
    if mode == NORM:
        return classify_norm(ins, extra_params)
    if mode == GATHER:
        return classify_gather(ins)
    if mode == GATHER_ND:
        return classify_gather_nd(ins)
    if mode == SLICE:
        return classify_slice(ins, extra_params)
    if mode == TRANSPOSE:
        return classify_transpose(ins, extra_params)
    if mode == CONCAT:
        return classify_concat(ins, extra_params)
    if mode == SPLIT:
        return classify_split(ins, extra_params)
    if mode == TRANSDATA:
        return classify_transdata(ins)

    return [ins]
