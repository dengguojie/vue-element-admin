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

from .known_reduce_classifier import KnownReduceClassifier
from .unknown_reduce_classifier import UnknownReduceClassifier

AXIS = "axis"


def classify(ins: list, keepdims: bool):
    """
    classify
    :param ins:
    :param keepdims:
    :return:
    """
    _check_all_unknown_shape(ins)
    _check_keepdims(keepdims)

    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == AXIS:
            if single_input.get("value"):
                return KnownReduceClassifier(ins, keepdims).classify()
            else:
                return UnknownReduceClassifier(ins, keepdims).classify()

    return [ins]


def _check_all_unknown_shape(ins: list):
    """
    check the case with shape -2
    :param ins:
    :return:
    """
    is_axis_not_negative_one = False
    is_known_classify = False
    has_all_unknown_shape = False

    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == AXIS:
            if single_input.get("value"):
                is_known_classify = True
            elif tuple(single_input.get("shape")) != (-1, ):
                is_axis_not_negative_one = True
        else:
            if tuple(single_input.get("shape")) == (-2, ):
                has_all_unknown_shape = True

    if has_all_unknown_shape and (is_known_classify or is_axis_not_negative_one):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "shape -2 is supported only when axis is input and its shape is -1"
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _check_keepdims(keepdims: bool):
    """
    check the type of keepdims
    :param keepdims:
    :return:
    """
    if not isinstance(keepdims, bool):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "keepdims in reduce_classifier must be the bool type"
        raise RuntimeError(dict_args, get_error_message(dict_args))
