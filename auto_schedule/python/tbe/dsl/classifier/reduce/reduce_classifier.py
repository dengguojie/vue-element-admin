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

from tbe.common.buildcfg import get_current_build_config
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import expr_compare
from tbe.dsl.base.operation import add_compile_info_inner

from .known_reduce_classifier import KnownReduceClassifier
from .mixed_reduce_classifier import MixedReduceClassifier
from .reduce_classifier_5hd import ReduceClassifier5HD
from .unknown_reduce_classifier import UnknownReduceClassifier

NC1HWC0_SHAPE_LENGTH = 5
NC1HWC0_ORI_SHAPE_LENGTH = 4
DISABLE_FUSE_AXES_5HD = [1, 4]


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


LAST_DIM_DTYPE_MAP_5HD = {
    "float32": (8, 16),
    "float16": 16,
    "int8": 32,
    "uint8": 32,
    "int32": (8, 16)
}


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
    expr_compare.is_true(extra_params is not None and "keepdims" in extra_params,
                         {"errCode": "E90001",
                          "detailed_cause": "inputs of classify must include the dict extra_params "
                                            "with the key keepdims when mode is reduce"
                         })

    keepdims = extra_params.get("keepdims")
    reduce_axes_type = extra_params.get("reduce_axes_type")

    _known_axis, neg_two = _need_process(ins)
    if neg_two:
        _check_binary_mode_not_support_fractal_format(ins)

        # if reduce mode is all , ignore axis value.
        if reduce_axes_type and reduce_axes_type == ReduceMode.ALL_REDUCE:
            add_compile_info_inner("_reduce_axes_type", 0)
            _known_axis = []
        elif _known_axis is not None:
            add_compile_info_inner("_ori_axis", _known_axis)

        ins_classify = MixedReduceClassifier(ins, keepdims, _known_axis).classify()
        return [ins_classify[0]] if get_current_build_config("enable_op_prebuild") else ins_classify

    _check_keepdims(keepdims)

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

    def _is_5hd_case(extra_params):
        """
        check if satisfy 5HD case.
        """
        # process fractal format business
        ignore_fractal_format = extra_params.get("ignore_fractal_format")
        if ignore_fractal_format is None:
            ignore_fractal_format = True
        else:
            _check_ignore_fractal_format(ignore_fractal_format)
        if ignore_fractal_format:
            return False, []

        disable_fuse_axes = []
        is_5hd = False
        # check if all input before reduce is 5HD format
        contains_5hd_format = False
        contains_nd_format = False
        for ins_single_input in ins:
            if ins_single_input.get("rel_pos_to_reduce") == InputType.BEFORE_REDUCE:
                if ins_single_input.get("format") == "NC1HWC0":
                    contains_5hd_format = True
                else:
                    contains_nd_format = True

        is_true = expr_compare.is_true
        if contains_5hd_format:
            is_true(not contains_nd_format,
                    {"errCode": "E90001",
                     "detailed_cause": "Format NC1HWC0 can not together with other formats at inputs before reduce."})
            is_5hd = True
            for ins_single_input in ins:
                if ins_single_input.get("rel_pos_to_reduce") == InputType.BEFORE_REDUCE:
                    shape = ins_single_input.get("shape")

                    # 1. check shape length
                    is_true(len(shape) == NC1HWC0_SHAPE_LENGTH,
                            {"errCode": "E90001",
                             "detailed_cause": "Input shape length of format NC1HWC0 should be 5."})

                    # 2. check shape last dim
                    last_dim = shape[-1]
                    dtype = ins_single_input.get("dtype")
                    is_true(isinstance(last_dim, int) and str(last_dim) in str(LAST_DIM_DTYPE_MAP_5HD.get(dtype)),
                            {"errCode": "E90001",
                             "detailed_cause": "Shape last dim of format NC1HWC0 is illegal."})

                    # 3. check ori_format
                    ori_format = ins_single_input.get("ori_format")
                    is_true(ori_format in ("NCHW", "NHWC"),
                            {"errCode": "E90001",
                             "detailed_cause": "Input ori_format of format NC1HWC0 should be NHWC or NCHW."})

                    # 4. check ori_shape
                    ori_shape = ins_single_input.get("ori_shape")
                    is_true(len(ori_shape) == NC1HWC0_ORI_SHAPE_LENGTH,
                            {"errCode": "E90001",
                             "detailed_cause": "Input ori_shape of format NC1HWC0 should be 4."})

            add_compile_info_inner("_disable_fuse_axes", DISABLE_FUSE_AXES_5HD)
            disable_fuse_axes.append(DISABLE_FUSE_AXES_5HD)
        return is_5hd, disable_fuse_axes

    is_5hd_case, disable_fuse_axes = _is_5hd_case(extra_params)

    # process reduce classify
    if is_5hd_case:
        result = ReduceClassifier5HD(ins, keepdims, disable_fuse_axes).classify()
    else:
        result = _classify_nd_format(ins, keepdims)
    ins_classify = [ins] if not result else result

    return [ins_classify[0]] if get_current_build_config("enable_op_prebuild") else ins_classify


def _classify_nd_format(ins, keepdims):
    """
    classify nd format
    """
    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == InputType.REDUCE_AXIS:
            if single_input.get("value"):
                add_compile_info_inner("_ori_axis", single_input.get("value"))
                result = KnownReduceClassifier(ins, keepdims).classify()
            else:
                result = UnknownReduceClassifier(ins, keepdims).classify()

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
        dict_args["detailed_cause"] = "Keepdims in reduce classifier must be the bool type."
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _check_ignore_fractal_format(ignore_fractal_format: bool):
    """
    check the type of ignore_fractal_format
    :param ignore_fractal_format: bool
    :return:
    """
    if not isinstance(ignore_fractal_format, bool):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "Ignore_fractal_format in reduce classifier must be the bool type."
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _check_binary_mode_not_support_fractal_format(ins):
    """
     neg two not support 5HD format
    """
    for ins_single_input in ins:
        if ins_single_input.get("rel_pos_to_reduce") == InputType.BEFORE_REDUCE:
            if ins_single_input.get("format") == "NC1HWC0":
                dict_args = {}
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "Binary mode not support format of NC1HWC0."
                raise RuntimeError(dict_args, get_error_message(dict_args))

