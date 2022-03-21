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
from typing import Any
from typing import Dict
from typing import Optional

from tbe.common.utils.errormgr import get_error_message
from tbe.common.buildcfg import get_current_build_config
from tbe.dsl.base.operation import add_compile_info_inner

from .known_reduce_classifier import KnownReduceClassifier
from .unknown_reduce_classifier import UnknownReduceClassifier
from .mixed_reduce_classifier import MixedReduceClassifier
from ..expr_compare import is_true


class InputType:
    """
    type of shape
    """
    REDUCE_AXIS = "axis"
    BEFORE_REDUCE = "before"


class ReduceMode:
    """
    mode of reduce
    """
    ALL_REDUCE = "all"


def _need_process(ins):
    _known_axis, _has_neg_two = None, False
    for _item in ins:
        if _item.get("rel_pos_to_reduce") == InputType.REDUCE_AXIS:
            _known_axis = _item.get("value")
            if isinstance(_known_axis, int):
                _known_axis = [_known_axis, ]
        else:
            _has_neg_two = True if -2 in _item.get("shape") else _has_neg_two

    return _known_axis, _has_neg_two


def classify(ins: list, extra_params: Optional[Dict[str, Any]] = None):
    """
    classify
    :param ins: inputs
    :param extra_params: extend paras
    :return:
    """
    is_true(extra_params is not None and "keepdims" in extra_params,
            {"errCode": "E90001",
             "detailed_cause": "inputs of classify must include the dict extra_params with the key keepdims " \
                               "when mode is reduce"})

    keepdims = extra_params.get("keepdims")
    reduce_axes_type = extra_params.get("reduce_axes_type")

    _known_axis, neg_two = _need_process(ins)
    if neg_two:
        # if reduce mode is all , ignore axis value.
        if reduce_axes_type and reduce_axes_type == ReduceMode.ALL_REDUCE:
            add_compile_info_inner("_reduce_axes_type", 0)
            _known_axis = []
        elif _known_axis is not None:
            add_compile_info_inner("_ori_axis", _known_axis)

        ins_classify = MixedReduceClassifier(ins, keepdims, _known_axis).classify()
        return [ins_classify[0]] if get_current_build_config("enable_op_prebuild") else ins_classify

    _check_keepdims(keepdims)
    result = None

    # for reduce case not -2, if reduce mode is all,
    # we should change value of axis to all shape dimension to do all reduce
    def _fill_reduce_axis_for_all_reduce():
        if reduce_axes_type and reduce_axes_type == ReduceMode.ALL_REDUCE:
            axes = []
            for ins_single_input in ins:
                if ins_single_input.get("rel_pos_to_reduce") == InputType.BEFORE_REDUCE:
                    axes.extend(range(len(ins_single_input.get("shape"))))
                    break
            for ins_single_input in ins:
                if ins_single_input.get("rel_pos_to_reduce") == InputType.REDUCE_AXIS:
                    ins_single_input["value"] = axes

    _fill_reduce_axis_for_all_reduce()

    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == InputType.REDUCE_AXIS:
            if single_input.get("value"):
                add_compile_info_inner("_ori_axis", single_input.get("value"))
                result = KnownReduceClassifier(ins, keepdims).classify()
            else:
                result = UnknownReduceClassifier(ins, keepdims).classify()
    ins_classify = [ins] if not result else result

    return [ins_classify[0]] if get_current_build_config("enable_op_prebuild") else ins_classify



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
