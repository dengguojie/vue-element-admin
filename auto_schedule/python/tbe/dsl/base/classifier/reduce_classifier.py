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
classifier of shape in reduce
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.operation import add_compile_info_inner

from .known_reduce_classifier import KnownReduceClassifier
from .unknown_reduce_classifier import UnknownReduceClassifier
from .mixed_reduce_classifier import MixedReduceClassifier

AXIS = "axis"


def _need_process(ins):
    _known_axis, _has_neg_two = None, False
    for _item in ins:
        if _item.get("rel_pos_to_reduce") == AXIS:
            _known_axis = _item.get("value")
            if isinstance(_known_axis, int):
                _known_axis = [_known_axis, ]
        else:
            _has_neg_two = True if -2 in _item.get("shape") else _has_neg_two

    return _known_axis, _has_neg_two


def classify(ins: list, keepdims: bool):
    """
    classify
    :param ins:
    :param keepdims:
    :return:
    """
    _known_axis, neg_two = _need_process(ins)
    if neg_two:
        if _known_axis is not None:
            add_compile_info_inner("_ori_axis", _known_axis)
        return MixedReduceClassifier(ins, keepdims, _known_axis).classify()

    _check_keepdims(keepdims)
    result = None
    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == AXIS:
            if single_input.get("value"):
                add_compile_info_inner("_ori_axis", single_input.get("value"))
                result = KnownReduceClassifier(ins, keepdims).classify()
            else:
                result = UnknownReduceClassifier(ins, keepdims).classify()
    result = [ins] if not result else result
    return result


def _check_keepdims(keepdims: bool):
    """
    check the type of keepdims
    :param keepdims:
    :return:
    """
    if not isinstance(keepdims, bool):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "keepdims in reduce_classifier must be the bool type"
        raise RuntimeError(dict_args, get_error_message(dict_args))
