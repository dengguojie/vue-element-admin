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
classifier of shape in norm
"""
import copy
from enum import Enum
from itertools import product

from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_context

from . import util

CONST = "const"
NORM = "Norm"

MAX_DIM_LEN = 8
REDUCE_PATTERN_KEY_WEIGHT = 1000


class ModeType(Enum):
    """
    mode type enum
    input after broadcast include:
    1. common
    2. no_fuse
    input before broadcast include:
    1. broadcast_axis_known
    2. no_broadcast, broadcast_reduce_equal, broadcast_reduce_opposite, all_broadcast,
       broadcast_unknown, single_broadcast_known_and_no_fuse
    """
    COMMON = "common"
    NO_FUSE = "no_fuse"
    BROADCAST_AXIS_KNOWN = "broadcast_axis_known"
    NO_BROADCAST = "no_broadcast"
    BROADCAST_REDUCE_EQUAL = "broadcast_reduce_equal"
    BROADCAST_REDUCE_OPPOSITE = "broadcast_reduce_opposite"
    ALL_BROADCAST = "all_broadcast"
    BROADCAST_UNKNOWN = "broadcast_unknown"
    SINGLE_BROADCAST_KNOWN_AND_NO_FUSE = "single_broadcast_known_and_no_fuse"


def _is_true(expr, dict_args):
    """
    check if true
    """
    if expr:
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _is_const(shape):
    """
    check const shape
    """
    return all(x > 0 for x in shape)


def _cal_norm_pattern(mode_str_list, broadcast_axis, reduce_axis, shape_len):
    """
    calculate norm pattern
    """
    def __get_single_pattern_key(_axis_list):
        _pattern = 0
        for _i in range(shape_len):
            if _i in _axis_list:
                _pattern += 2 ** (shape_len - _i - 1) * 2
            else:
                _pattern += 2 ** (shape_len - _i - 1)

        return _pattern

    input_mode_and_key_map = {
        ModeType.NO_BROADCAST: 0,
        ModeType.BROADCAST_REDUCE_EQUAL: 1,
        ModeType.BROADCAST_REDUCE_OPPOSITE: 2,
        ModeType.ALL_BROADCAST: 3,
        ModeType.BROADCAST_UNKNOWN: 4,
        ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE: 4
    }

    mode_enum_list = []
    for mode_str in mode_str_list:
        mode_enum_list.append(ModeType(mode_str))
    # all inputs broadcast axis are known and the same
    if broadcast_axis is not None:
        broadcast_pattern_key = __get_single_pattern_key(broadcast_axis)
    else:
        broadcast_pattern_key = 0
        for single_mode_enum in mode_enum_list:
            # input after broadcast is not involved in pattern key calculation
            if single_mode_enum in (ModeType.COMMON, ModeType.NO_FUSE):
                continue
            broadcast_pattern_key = 10 * broadcast_pattern_key + input_mode_and_key_map.get(single_mode_enum)

    reduce_pattern_key = __get_single_pattern_key(reduce_axis)

    return reduce_pattern_key * REDUCE_PATTERN_KEY_WEIGHT + broadcast_pattern_key


def _infer_negative_two_and_pre_process(ins, reduce_axis, broadcast_axis_list, is_fuse):
    """
    infer -2 in inputs and do some pre process
    """
    def __infer_negative_two():
        _dim_len = -1
        _exist_negative_two = False

        for _single_input in ins:
            _single_shape = _single_input.get("shape")
            if tuple(_single_shape) != (-2, ):
                _dim_len = max(_dim_len, len(_single_shape))

        if _dim_len == -1:
            _exist_negative_two = True
            _dim_len = MAX_DIM_LEN

        _ins = copy.deepcopy(ins)
        for _single_input in _ins:
            _shape_tuple = tuple(_single_input.get("shape"))
            _range_list = _single_input.get("range")
            if _exist_negative_two:
                _is_true(_shape_tuple != (-2, ),
                         {"errCode": "E90001",
                          "detailed_cause": "norm classifier do not support -2 and -1 mixed"})
            if _shape_tuple == (-2, ):
                _single_input["shape"] = [-1] * _dim_len
                _single_input["range"] = [(1, None) for _ in range(_dim_len)]
            elif len(_shape_tuple) < _dim_len:
                # pad 1 from high dimension
                _single_input["shape"] = [1] * (_dim_len - len(_shape_tuple)) + list(_shape_tuple)
                _single_input["range"] = [(1, 1)] * _dim_len if _range_list is None else \
                                         [(1, 1)] * (_dim_len - len(_range_list)) + list(_range_list)
        ins_list.append(_ins)

        return _dim_len, _exist_negative_two

    ins_list = []
    dim_len, exist_negative_two = __infer_negative_two()
    normalize_reduce_axis = list(set(x + dim_len if x < 0 else x for x in reduce_axis))
    min_dim_len = max(normalize_reduce_axis) + 1

    for broadcast_axis in broadcast_axis_list:
        if broadcast_axis:
            current_min_dim_len = max(x + dim_len if x < 0 else x for x in broadcast_axis) + 1
            if current_min_dim_len > min_dim_len:
                min_dim_len = current_min_dim_len

    _is_true(min_dim_len > dim_len,
             {"errCode": "E90001",
              "detailed_cause": "max of reduce axis in norm classifier should be less than dim len"})

    max_dim_len = min_dim_len if is_fuse else dim_len
    # input are all -2
    # include last reduce and nlast reduce
    if exist_negative_two:
        for opt_dim_len in range(min_dim_len, max_dim_len + 1):
            local_ins = copy.deepcopy(ins)
            for single_input in local_ins:
                if tuple(single_input.get("shape")) == (-2, ):
                    single_input["shape"] = [-1] * opt_dim_len
                    single_input["range"] = [(1, None) for _ in range(opt_dim_len)]
            ins_list.append(local_ins)

    return ins_list, normalize_reduce_axis


def _add_remove_last_one(ins, input_type_list, reduce_axis, broadcast_axis, is_fuse_axis):
    """
    add remove last several 1 case
    """
    # the following is the case that can remove one axis
    # 1. must have after broadcast input
    # 2. is_fuse_axis is true
    # 3. broadcast axis is known when has before broadcast input
    # if the reduce axis is empty after removing one axis, a R axis should be made up on the front
    # shape: (A, ) -> (1, A)
    # reduce_axis: () -> (0, )
    # return: is append remove last one case and is desert origin case
    def __judge_same_state(_first_idx, _second_idx, _axis_list):
        return (_first_idx in _axis_list) == (_second_idx in _axis_list)

    def __possible_remove_axis(_last_index):
        _remove_axis_list = [_last_index]
        for _idx in range(_last_index - 1, -1, -1):
            if not __judge_same_state(_idx, _idx + 1, reduce_axis):
                return _remove_axis_list
            if broadcast_axis is not None:
                if not __judge_same_state(_idx, _idx + 1, broadcast_axis):
                    return _remove_axis_list
            _remove_axis_list.append(_idx)

        return _remove_axis_list

    is_no_after_broadcast_input = 0 not in input_type_list
    is_no_fuse_axis = not is_fuse_axis
    is_unify_broadcast_axis_unknown = broadcast_axis is None and input_type_list != [0] * len(input_type_list)
    is_cannot_remove_one = is_no_after_broadcast_input or is_no_fuse_axis or is_unify_broadcast_axis_unknown
    if is_cannot_remove_one:
        return False, False

    input_after_bro_index = input_type_list.index(0)
    shape_after_bro = ins[input_after_bro_index].get("shape")
    remove_axis_list = __possible_remove_axis(len(shape_after_bro) - 1)

    # origin case should be deserted when dims of remove axes are all 1
    is_compile_known_all_one = True
    for remove_axis_idx in remove_axis_list:
        for single_input in ins:
            single_shape = list(single_input["shape"])
            is_compile_known_all_one = is_compile_known_all_one and single_shape[remove_axis_idx] == 1

    for remove_axis_idx in remove_axis_list:
        if remove_axis_idx in reduce_axis:
            reduce_axis.remove(remove_axis_idx)
        # A -> (1, A) and reduce axis is [0]
        is_all_common_axis = len(reduce_axis) == 0
        for single_input in ins:
            single_shape = list(single_input["shape"])
            if single_shape[remove_axis_idx] > 1:
                return False, False
            single_range = list(single_input["range"])
            single_shape.pop(remove_axis_idx)
            single_range.pop(remove_axis_idx)
            if is_all_common_axis:
                single_shape = [1] + single_shape
                single_range = [(1, 1)] + single_range
            single_input["shape"] = single_shape
            single_input["range"] = single_range
        if is_all_common_axis:
            reduce_axis.append(0)
        if broadcast_axis is not None:
            if remove_axis_idx in broadcast_axis:
                broadcast_axis.remove(remove_axis_idx)
            if is_all_common_axis:
                for idx, axis in enumerate(broadcast_axis):
                    broadcast_axis[idx] = axis + 1

    return True, is_compile_known_all_one


def _process_extra_params(input_num, extra_params):
    """
    process extra params
    """
    ori_input_type_list = [0] * input_num
    ori_compile_broadcast_axes = {}
    ori_runtime_broadcast_axes = {}
    broadcast_axis_list = [None] * input_num

    if extra_params is None:
        return ori_input_type_list, broadcast_axis_list

    ori_same_input_shape_group_list = [list(range(input_num))]
    input_type_list = [0] * input_num

    for key, value in extra_params.items():
        if key == "input_shape_type":
            _is_true(not isinstance(value, (list, tuple)) or len(value) != input_num,
                     {"errCode": "E90001",
                      "detailed_cause": "input_shape_type in norm classifier must be a list and"
                                        "its len must be equal to input num"})
            for single_value in value:
                _is_true(single_value not in (0, 1),
                         {"errCode": "E90001",
                          "detailed_cause": "value in input_shape_type in norm classifier must be 0 or 1"})
            ori_input_type_list = value
        if key == "same_input_shape_group" in extra_params:
            _is_true(not isinstance(value, (list, tuple)) or len(value) == 0 or
                     not isinstance(value[0], (list, tuple)),
                     {"errCode": "E90001",
                      "detailed_cause": "same_input_shape_group in norm classifier must be a dual list"})
            ori_same_input_shape_group_list = value
        if key == "compile_broadcast_axes" in extra_params:
            _is_true(not isinstance(value, dict),
                     {"errCode": "E90001",
                      "detailed_cause": "compile_broadcast_axes in norm classifier must be a dict"})
            ori_compile_broadcast_axes = value
        if key == "runtime_broadcast_axes" in extra_params:
            _is_true(not isinstance(value, dict),
                     {"errCode": "E90001",
                      "detailed_cause": "runtime_broadcast_axes in norm classifier must be a dict"})
            ori_runtime_broadcast_axes = value

    for input_index in range(input_num):
        is_broadcast_axis_known = \
            input_index in ori_compile_broadcast_axes and input_index not in ori_runtime_broadcast_axes
        if is_broadcast_axis_known:
            broadcast_axis_list[input_index] = ori_compile_broadcast_axes[input_index]

    input_type_flag = 0
    complete_shape_index = [i for i in range(input_num) if ori_input_type_list[i] == 0]
    for single_same_input_shape_group in ori_same_input_shape_group_list:
        if set(single_same_input_shape_group) & set(complete_shape_index):
            continue
        input_type_flag += 1
        for single_index in single_same_input_shape_group:
            _is_true(input_type_list[single_index] != 0,
                     {"errCode": "E90001",
                      "detailed_cause": "norm classifier do not support same_input_shape_group with intersection"})
            input_type_list[single_index] = input_type_flag

    input_type_flag = max(input_type_flag, 1)
    for index in range(input_num):
        remain_partial_shape = ori_input_type_list[index] == 1 and input_type_list[index] == 0
        if remain_partial_shape:
            input_type_list[index] = input_type_flag
            input_type_flag += 1

    _is_true(len(set(input_type_list)) > 4,
             {"errCode": "E90001",
              "detailed_cause": "Currently, a maximum of four input types are supported in norm schedule"})

    return input_type_list, broadcast_axis_list


def classify(ins: list, extra_params: dict):
    """
    classify
    :param ins: inputs list and last element must be reduce axis
    :param extra_params: a dict with the following keys:
        "input_shape_type": list, the length is the same as input num. It represented the type of input shape and
                            only support 0 or 1. 0 means complete shape, 1 means partial shape, default is all 0.
        "same_input_shape_group": list[list], the value in sub_list represents index of input and
                                  all inputs in sub_list are same.
                                  Example: [[1, 2]] means input 1 and input 2 are equal.
                                  But intersection is not supported, such as [[1, 2], [2, 3]] is illegal.
                                  Default is all inputs are same.
        "compile_broadcast_axes": dict, ksy is input index, value is compile broadcast axis(list).
                                  Defalut is common axis.
        "runtime_broadcast_axes": dict, ksy is input index, value is runtime broadcast axis(list).
                                  Defalut is common axis.
    :return:
    """
    _is_true(extra_params is not None and not isinstance(extra_params, dict),
             {"errCode": "E90001",
              "detailed_cause": "extra_params must be a dict or None when mode is norm"})
    _is_true(len(ins) < 1 or not isinstance(ins[-1], (list, tuple)),
             {"errCode": "E90001",
              "detailed_cause": "last element of inputs in norm classify must be a list and represent reduce axis"})
    get_context().set_pattern(NORM)
    ins = copy.deepcopy(ins)
    # fuse axis flag
    is_fuse_axis = True
    if extra_params is not None and "disable_optimization" in extra_params:
        is_fuse_axis = not extra_params.get("disable_optimization")
    add_compile_info_inner("_fuse_axis", is_fuse_axis)
    # ins last element is reduce axis
    reduce_axis = list(set(ins[-1]))
    ins.pop()
    add_compile_info_inner("_ori_reduce_axis", reduce_axis[:])
    input_type_list, broadcast_axis_list = _process_extra_params(len(ins), extra_params)
    add_compile_info_inner("_input_type", input_type_list)
    ins_list, ins_reduce_axis = \
        _infer_negative_two_and_pre_process(ins, reduce_axis, broadcast_axis_list, is_fuse_axis)

    norm_classify_out = []
    norm_pattern_non_duplicate_list = []
    for single_ins in ins_list:
        norm_classifier = NormClassifier(single_ins, ins_reduce_axis, is_fuse_axis,
                                         input_type_list, broadcast_axis_list)
        # When all inputs have the same determinable broadcast axis, the broadcast axis is unify_broadcast_axis;
        # In other cases, unify_broadcast_axis is None
        unify_broadcast_axis = norm_classifier.get_unify_broadcast_axis()
        # all inputs have the same determinable broadcast axis
        if unify_broadcast_axis is not None:
            unify_broadcast_axis = list(unify_broadcast_axis)
            add_compile_info_inner("_ori_broadcast_axis", unify_broadcast_axis[:])
        is_append_remove_one, is_desert_origin = \
            _add_remove_last_one(single_ins, input_type_list, ins_reduce_axis, unify_broadcast_axis, is_fuse_axis)
        # before remove last 1
        if not is_desert_origin:
            for single_classify_out in norm_classifier.classify():
                # if the same pattern exists in norm classify output, this single output should be deserted
                norm_pattern = single_classify_out[0].get("norm_pattern")
                if norm_pattern not in norm_pattern_non_duplicate_list:
                    norm_classify_out.append(single_classify_out)
                    norm_pattern_non_duplicate_list.append(norm_pattern)
        # after remove last 1
        if is_append_remove_one:
            for single_classify_out in NormClassifier(single_ins, ins_reduce_axis, is_fuse_axis,
                                                      input_type_list, broadcast_axis_list).classify():
                # if the same pattern exists in norm classify output, this single output should be deserted
                norm_pattern = single_classify_out[0].get("norm_pattern")
                if norm_pattern not in norm_pattern_non_duplicate_list:
                    norm_classify_out.append(single_classify_out)
                    norm_pattern_non_duplicate_list.append(norm_pattern)

    # const process can not be performed when dynamic and const are mixed in norm classify output
    is_const_set = set()
    for single_classify_out in norm_classify_out:
        is_const = True
        for single_input in single_classify_out:
            if isinstance(single_input, dict):
                is_const = is_const and _is_const(single_input.get("shape"))
        is_const_set.add(is_const)
    is_const_and_dynamic_mixed = len(is_const_set) == 2
    get_context().add("_const_and_dynamic_mixed", is_const_and_dynamic_mixed)

    return norm_classify_out


class NormClassifier:
    """
    norm classifier
    """
    def __init__(self, ins: list, reduce_axis: list, is_fuse_axis: bool, input_type_list: list,
                 broadcast_axis_list: list):
        """
        norm classifier init
        """
        self.ins = ins

        self.max_shape_len = max(len(x["shape"]) for x in ins)
        self.reduce_axis = reduce_axis
        self.unreduce_axis = sorted(set(range(self.max_shape_len)) - set(self.reduce_axis))

        self.is_fuse_axis = is_fuse_axis
        self.input_type_list = input_type_list
        self.broadcast_axis_list = broadcast_axis_list

        self.inputs_after_bro, self.inputs_before_bro, self.inputs_bro_axis = self._inputs_classify()
        self.shape_before_fuse, self.range_before_fuse, self.inputs_before_bro_info = self._infer_and_extract_info()
        self.unify_broadcast_axis = self.get_unify_broadcast_axis()
        self.shape_after_fuse, self.range_after_fuse, self.reduce_axis_after_fuse, self.broadcast_axis_after_fuse = \
            self._simplify()

    def _inputs_classify(self):
        """
        classify inputs
        return:
        1. inputs after broadcast list
        2. input type and before broadcast inputs map {input_type: input before broadcast}
        3. input type and broadcast_axis of before broadcast inputs map {input_type: broadcast_axis}
        """
        inputs_after_bro = []
        inputs_before_bro = {}
        inputs_bro_axis = {}
        for index, input_type in enumerate(self.input_type_list):
            if input_type == 0:
                inputs_after_bro.append(self.ins[index])
            else:
                if input_type in inputs_before_bro:
                    inputs_before_bro[input_type].append(self.ins[index])
                    inputs_bro_axis[input_type].append(self.broadcast_axis_list[index])
                else:
                    inputs_before_bro[input_type] = [self.ins[index]]
                    inputs_bro_axis[input_type] = [self.broadcast_axis_list[index]]

        return inputs_after_bro, inputs_before_bro, inputs_bro_axis

    def _infer_and_extract_info(self):
        """
        infer and extract info
        return:
        1. shape to be fuse
        2. range to be fuse
        3. input type and info of before broadcast input map:
        # {
            input_type:
                {
                    "shape": ,
                    "range": ,
                    "all broadcast axis": ,
                    "partial_broadcast_axis": ,
                    "partial_unbroadcast_axis":
                }
        # }
        """
        def __get_shape_and_range_list(_inputs):
            _shape_list = [x["shape"] for x in _inputs]
            _range_list = []
            for _single_input in _inputs:
                _single_range = _single_input.get("range")
                if _single_range:
                    _range_list.append(_single_range)
                else:
                    _range_list.append([(1, None) for _ in range(len(_shape_list[0]))])

            return _shape_list, _range_list

        def __infer_shape_and_range(_shape_list, _range_list):
            def ___get_dim(__i):
                return max(s[__i] for s in _shape_list)

            def ___select_min_upper_bound(__upper_bound_list):
                __min_ele = util.VAR_BOUND_LIMIT + 1
                for __ele in __upper_bound_list:
                    if __ele is None:
                        continue
                    if __ele < __min_ele:
                        __min_ele = __ele

                return __min_ele if __min_ele != util.VAR_BOUND_LIMIT + 1 else None

            def ___get_range(__i):
                if _shape_out[__i] != -1:
                    return _shape_out[__i], _shape_out[__i]

                return max(r[__i][0] for r in _range_list), \
                    ___select_min_upper_bound([r[__i][1] for r in _range_list])

            _shape_out = [___get_dim(i) for i in range(len(_shape_list[0]))]
            _range_out = [___get_range(i) for i in range(len(_range_list[0]))]

            for _index in range(len(_shape_out)):
                if _range_out[_index][0] == _range_out[_index][1]:
                    _shape_out[_index] = _range_out[_index][0]

            return _shape_out, _range_out

        def __infer_broadcast_axis(_shape, _range, _current_broadcast_list):
            # normalize broadcast axis
            _is_all_unknown = True
            _broadcast_axis = set()
            for _single_broadcast_axis in _current_broadcast_list:
                if _single_broadcast_axis is not None:
                    _is_all_unknown = False
                    for _axis in _single_broadcast_axis:
                        if _axis < 0:
                            _axis = _axis + self.max_shape_len
                        _broadcast_axis.add(_axis)
            # original input broadcast axis
            _input_broadcast_axis = None if _is_all_unknown else sorted(_broadcast_axis)
            # infer broadcast axis and unbroadcast axis according to shape and range
            _infer_broadcast_axis = []
            _infer_unbroadcast_axis = []
            for _index, _value in enumerate(_shape):
                if _value == 1:
                    _infer_broadcast_axis.append(_index)
                elif _value != -1 or _range[_index][0] > 1:
                    _infer_unbroadcast_axis.append(_index)

            if len(_infer_broadcast_axis) + len(_infer_unbroadcast_axis) == len(_shape):
                return _infer_broadcast_axis, _infer_broadcast_axis, _infer_unbroadcast_axis

            return _input_broadcast_axis, _infer_broadcast_axis, _infer_unbroadcast_axis

        shape_to_fuse = []
        range_to_fuse = []
        inputs_before_bro_info = {}
        # if exist after broadcast input, shape_to_fuse and range_to_fuse is the shape and range of after bro input
        if self.inputs_after_bro:
            local_shape_list, local_range_list = __get_shape_and_range_list(self.inputs_after_bro)
            shape_to_fuse, range_to_fuse = __infer_shape_and_range(local_shape_list, local_range_list)

        for input_type in self.inputs_before_bro:
            local_shape_list, local_range_list = __get_shape_and_range_list(self.inputs_before_bro[input_type])
            local_shape, local_range = __infer_shape_and_range(local_shape_list, local_range_list)
            # all_broadcast_axis means all broadcast axes of input are known
            # partial_broadcast_axis means partial broadcast axes can be inferred
            # partial_unbroadcast_axis means partial unbroadcast axes can be inferred
            all_broadcast_axis, partial_broadcast_axis, partial_unbroadcast_axis = \
                __infer_broadcast_axis(local_shape, local_range, self.inputs_bro_axis[input_type])
            # the value of dim corresponding to the broadcast axis set 1
            if all_broadcast_axis is not None:
                for idx in all_broadcast_axis:
                    local_shape[idx] = 1
                    local_range[idx] = (1, 1)
            inputs_before_bro_info[input_type] = {
                "shape": local_shape,
                "range": local_range,
                "all_broadcast_axis": all_broadcast_axis,
                "partial_broadcast_axis": partial_broadcast_axis,
                "partial_unbroadcast_axis": partial_unbroadcast_axis
            }

        # if there is no inputs_after_bro
        # we suppose the shape_to_fuse is [-1] * max_shape_len
        # except that dims inputs_before_bro are const and shape_to_fuse can be inferred
        if not shape_to_fuse:
            before_bro_shape_list = [inputs_before_bro_info[x]["shape"] for x in inputs_before_bro_info]
            shape_to_fuse = [-1] * self.max_shape_len
            range_to_fuse = [(1, None) for _ in range(self.max_shape_len)]
            for idx in range(self.max_shape_len):
                input_dim = [x[idx] for x in before_bro_shape_list]
                max_dim = max(input_dim)
                min_dim = min(input_dim)
                if max_dim > 1 or (max_dim == min_dim == 1):
                    shape_to_fuse[idx] = max_dim
                    range_to_fuse[idx] = (max_dim, max_dim)

        return shape_to_fuse, range_to_fuse, inputs_before_bro_info

    def get_unify_broadcast_axis(self):
        """
        obtain the broadcast axis when all broadcast axes of inputs are the same
        """
        broadcast_axis_set = set()
        for input_type in self.inputs_before_bro_info:
            single_broadcast_axis = self.inputs_before_bro_info[input_type].get("all_broadcast_axis")
            if single_broadcast_axis is None:
                return None
            broadcast_axis_set.add(tuple(sorted(single_broadcast_axis)))
        if len(broadcast_axis_set) == 1:
            return list(broadcast_axis_set)[0]

        return None

    def _simplify(self):
        """
        simplify shape, range, reduce axis, broadcast axis
        fuse continuous reduce axis or broadcast aixs or common axis or reduce and broadcast axis.
        """
        def __obtain_state():
            if is_reduce_axis and not is_broadcast_axis:
                return "reduce"
            if is_broadcast_axis and not is_reduce_axis:
                return "broadcast"
            if is_broadcast_axis and is_reduce_axis:
                return "reduce_and_broadcast"

            return "common"

        if not self.is_fuse_axis:
            return self.shape_before_fuse, self.range_before_fuse, self.reduce_axis, self.unify_broadcast_axis

        f_shape, f_ranges, f_reduce_axis, f_broadcast_axis = [], [], [], []
        state = "init"
        for i, (d, r) in enumerate(zip(self.shape_before_fuse, self.range_before_fuse)):
            is_reduce_axis = i in self.reduce_axis
            is_broadcast_axis = i in self.unify_broadcast_axis if self.unify_broadcast_axis is not None else False
            state_i = __obtain_state()

            if state == state_i:
                f_shape[-1] = util.combine_dim([f_shape[-1], d])
                f_ranges[-1] = util.combine_range([f_ranges[-1], r])
            else:
                f_shape.append(d)
                f_ranges.append(r)

            if is_reduce_axis:
                reduce_axis = len(f_shape) - 1
                if not f_reduce_axis or f_reduce_axis[-1] != reduce_axis:
                    f_reduce_axis.append(reduce_axis)

            if is_broadcast_axis:
                broadcast_axis = len(f_shape) - 1
                if not f_broadcast_axis or f_broadcast_axis[-1] != broadcast_axis:
                    f_broadcast_axis.append(broadcast_axis)

            state = state_i

        return f_shape, f_ranges, f_reduce_axis, f_broadcast_axis

    def _gen_classify_out(self):
        """
        generate output of norm classifier
        """
        def __infer_before_broadcast_mode(all_broadcast_axis, partial_broadcast_axis, partial_unbroadcast_axis):
            """
            return:
            possible mode list
            flag of whether the unknown_broadcast case should be generated
            flag of whether the no fuse case should be generated
            """
            def ___increase_and_decrease_enum():
                __out_mode_enum = set()
                __partial_broadcast_axis_set = set(partial_broadcast_axis)
                __partial_unbroadcast_axis_set = set(partial_unbroadcast_axis)
                __reduce_axis_set = set(self.reduce_axis)
                __unreduce_axis_set = set(self.unreduce_axis)
                if not __partial_broadcast_axis_set:
                    __out_mode_enum.add(ModeType.NO_BROADCAST)
                if __partial_broadcast_axis_set & __reduce_axis_set == __partial_broadcast_axis_set:
                    __out_mode_enum.add(ModeType.ALL_BROADCAST)
                    __out_mode_enum.add(ModeType.BROADCAST_REDUCE_EQUAL)
                if __partial_broadcast_axis_set & __unreduce_axis_set == __partial_broadcast_axis_set:
                    __out_mode_enum.add(ModeType.ALL_BROADCAST)
                    __out_mode_enum.add(ModeType.BROADCAST_REDUCE_OPPOSITE)
                if __partial_broadcast_axis_set & __reduce_axis_set and\
                        __partial_broadcast_axis_set & __unreduce_axis_set:
                    __out_mode_enum.add(ModeType.ALL_BROADCAST)
                    __out_mode_enum.discard(ModeType.BROADCAST_REDUCE_EQUAL)
                    __out_mode_enum.discard(ModeType.BROADCAST_REDUCE_OPPOSITE)
                if __partial_unbroadcast_axis_set:
                    __out_mode_enum.discard(ModeType.ALL_BROADCAST)
                if __partial_unbroadcast_axis_set & __reduce_axis_set == __partial_unbroadcast_axis_set and\
                        __partial_unbroadcast_axis_set:
                    __out_mode_enum.discard(ModeType.ALL_BROADCAST)
                    __out_mode_enum.discard(ModeType.BROADCAST_REDUCE_EQUAL)
                if __partial_unbroadcast_axis_set & __unreduce_axis_set == __partial_unbroadcast_axis_set and\
                        __partial_unbroadcast_axis_set:
                    __out_mode_enum.discard(ModeType.ALL_BROADCAST)
                    __out_mode_enum.discard(ModeType.BROADCAST_REDUCE_OPPOSITE)
                if __partial_unbroadcast_axis_set & __reduce_axis_set and\
                        __partial_unbroadcast_axis_set & __unreduce_axis_set:
                    __out_mode_enum.discard(ModeType.ALL_BROADCAST)
                    __out_mode_enum.discard(ModeType.BROADCAST_REDUCE_EQUAL)
                    __out_mode_enum.discard(ModeType.BROADCAST_REDUCE_OPPOSITE)

                return __out_mode_enum

            # all broadcast axes are known and the same
            if self.unify_broadcast_axis is not None:
                return [ModeType.BROADCAST_AXIS_KNOWN], False, False
            # single broadcast axis is known and mode inferring is no need
            if all_broadcast_axis is not None:
                # broadcast axis is []
                if not all_broadcast_axis:
                    return [ModeType.NO_BROADCAST], False, False
                elif all_broadcast_axis == self.reduce_axis:
                    return [ModeType.BROADCAST_REDUCE_EQUAL], False, False
                elif all_broadcast_axis == self.unreduce_axis:
                    return [ModeType.BROADCAST_REDUCE_OPPOSITE], False, False
                elif all_broadcast_axis == list(range(self.max_shape_len)):
                    return [ModeType.ALL_BROADCAST], False, False
                else:
                    return [ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE], False, True
            # infer mode
            _enum_broadcast_unknown = False if (len(self.reduce_axis) == len(self.unreduce_axis) == 1) else True

            return ___increase_and_decrease_enum(), _enum_broadcast_unknown, False

        def __gen_single_mode_out(_mode, _input_type, _ori_before_bro_shape=None, _ori_before_bro_range=None,
                                  _single_broadcast_axis=None):
            """
            generate classify out of single input mode
            input after broadcast include:
            1. common
            2. no_fuse
            input before broadcast include:
            1. broadcast_axis_known
            2. no_broadcast, broadcast_reduce_equal, broadcast_reduce_opposite, all_broadcast,
            broadcast_unknown, single_broadcast_known_and_no_fuse
            """
            def ___gen_single_out_shape_or_range(__ori_value_list, __axis_list, __set_value,
                                                 __in_axis_set_value=True):
                __out_list = []
                for __index in range(len(__ori_value_list)):
                    if __in_axis_set_value:
                        if __index in __axis_list:
                            __out_list.append(__set_value)
                        else:
                            __out_list.append(__ori_value_list[__index])
                    else:
                        if __index not in __axis_list:
                            __out_list.append(__set_value)
                        else:
                            __out_list.append(__ori_value_list[__index])

                return __out_list

            _out_shape = []
            _out_range = []
            _out_broadcast_axis = None
            if _mode == ModeType.ALL_BROADCAST:
                _out_shape = [1] * len(self.shape_after_fuse)
                _out_range = [(1, 1)] * len(self.range_after_fuse)
                _out_broadcast_axis = list(range(len(self.shape_after_fuse)))
            elif _mode == ModeType.BROADCAST_REDUCE_EQUAL:
                _out_shape = ___gen_single_out_shape_or_range(self.shape_after_fuse, self.reduce_axis_after_fuse, 1)
                _out_range = ___gen_single_out_shape_or_range(self.range_after_fuse, self.reduce_axis_after_fuse,
                                                              [(1, 1)])
                _out_broadcast_axis = self.reduce_axis_after_fuse[:]
            elif _mode == ModeType.BROADCAST_REDUCE_OPPOSITE:
                _out_shape = ___gen_single_out_shape_or_range(self.shape_after_fuse, self.reduce_axis_after_fuse, 1,
                                                              False)
                _out_range = ___gen_single_out_shape_or_range(self.range_after_fuse, self.reduce_axis_after_fuse,
                                                              [(1, 1)], False)
                _out_broadcast_axis = sorted(set(range(len(self.shape_after_fuse))) -
                                             set(self.reduce_axis_after_fuse))
            elif _mode == ModeType.BROADCAST_UNKNOWN:
                _out_shape = _ori_before_bro_shape[:]
                _out_range = _ori_before_bro_range[:]
            elif _mode == ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE:
                _out_shape = ___gen_single_out_shape_or_range(self.shape_before_fuse, _single_broadcast_axis, 1)
                _out_range = ___gen_single_out_shape_or_range(self.range_before_fuse, _single_broadcast_axis,
                                                              [(1, 1)])
                _out_broadcast_axis = _single_broadcast_axis[:]
            elif _mode == ModeType.BROADCAST_AXIS_KNOWN:
                _out_shape = ___gen_single_out_shape_or_range(self.shape_after_fuse, self.broadcast_axis_after_fuse,
                                                              1)
                _out_range = ___gen_single_out_shape_or_range(self.range_after_fuse, self.broadcast_axis_after_fuse,
                                                              [(1, 1)])
                _out_broadcast_axis = self.broadcast_axis_after_fuse[:]
            elif _mode == ModeType.NO_BROADCAST:
                _out_shape = self.shape_after_fuse[:]
                _out_range = self.range_after_fuse[:]
                _out_broadcast_axis = []
            elif _mode == ModeType.NO_FUSE:
                _out_shape = self.shape_before_fuse[:]
                _out_range = self.range_before_fuse[:]
            # ModeType.COMMON
            else:
                _out_shape = self.shape_after_fuse[:]
                _out_range = self.range_after_fuse[:]

            return {
                "shape": _out_shape,
                "range": _out_range,
                "mode": _mode.value,
                "input_type": _input_type,
                "broadcast_axis": _out_broadcast_axis
            }

        def __gen_reduce_axis_out(_is_fuse):
            """
            generate reduce axis output
            """
            return self.reduce_axis_after_fuse if _is_fuse else self.reduce_axis

        def __pruning(_input_dict):
            """
            pruning when:
            1. some inputs have fused axes while some inputs have unfused axes
            2. there is no after broadcast input and before broadcast inputs have duplicated broadcast axis
            """
            _input_mode_list = [ModeType(x.get("mode")) for x in _input_dict]
            if set(_input_mode_list) & {ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE, ModeType.BROADCAST_UNKNOWN}\
                    and set(_input_mode_list) & {ModeType.ALL_BROADCAST, ModeType.BROADCAST_REDUCE_EQUAL,
                                                 ModeType.BROADCAST_REDUCE_OPPOSITE, ModeType.NO_BROADCAST}:
                return True

            _input_type_list = [x.get("input_type") for x in _input_dict]
            # exist input after broadcast
            if 0 in _input_type_list:
                return False
            # only one input type
            if len(set(_input_type_list)) < 2:
                return False
            _broadcast_axis_set = set(_input_dict[0].get("broadcast_axis"))
            for _idx in range(1, len(_input_dict)):
                _broadcast_axis_set = _broadcast_axis_set & set(_input_dict[_idx].get("broadcast_axis"))
            # have duplicated broadcast axis
            if _broadcast_axis_set:
                for _idx in _broadcast_axis_set:
                    _input_dim = [x["shape"][_idx] for x in _input_dict]
                    # all broadcast dims are not equal to 1
                    if not (min(_input_dim) == max(_input_dim) == 1):
                        return True

            return False

        exist_unknown_broadcast_case = False
        exist_no_fuse_case = False
        # {input type: single input classify out}
        before_broadcast_enum_out = {}
        # index list
        input_type_order_list = []

        for input_type in self.inputs_before_bro_info:
            input_type_order_list.append(input_type)
            mode_list, flag_of_unknown_broadcast, flag_of_no_fuse = \
                __infer_before_broadcast_mode(self.inputs_before_bro_info[input_type]["all_broadcast_axis"],
                                              self.inputs_before_bro_info[input_type]["partial_broadcast_axis"],
                                              self.inputs_before_bro_info[input_type]["partial_unbroadcast_axis"])
            exist_unknown_broadcast_case = exist_unknown_broadcast_case or flag_of_unknown_broadcast
            exist_no_fuse_case = exist_no_fuse_case or flag_of_no_fuse
            single_input_out = []
            for mode in mode_list:
                single_input_out.append(
                    __gen_single_mode_out(mode,
                                          input_type,
                                          self.inputs_before_bro_info[input_type]["shape"],
                                          self.inputs_before_bro_info[input_type]["range"],
                                          self.inputs_before_bro_info[input_type]["all_broadcast_axis"])
                )
            before_broadcast_enum_out[input_type] = single_input_out

        after_broadcast_enum_out = [__gen_single_mode_out(ModeType.COMMON, 0)]

        product_in = [before_broadcast_enum_out[index] for index in before_broadcast_enum_out]
        product_in.append(after_broadcast_enum_out)
        input_type_order_list.append(0)
        # combine all possibilities for each type of input
        product_out = list(product(*product_in))

        total_classify_out = []
        for product_item in product_out:
            local_product_item = copy.deepcopy(product_item)
            single_classify_out = []
            for input_type in self.input_type_list:
                idx = input_type_order_list.index(input_type)
                single_classify_out.append(list(local_product_item)[idx])
            if __pruning(single_classify_out):
                continue
            single_classify_out.append(__gen_reduce_axis_out(True))
            total_classify_out.append(single_classify_out)

        # add no fuse case
        if exist_unknown_broadcast_case or exist_no_fuse_case:
            no_fuse_case = []
            for _, input_type in enumerate(self.input_type_list):
                if input_type == 0:
                    no_fuse_case.append(__gen_single_mode_out(ModeType.NO_FUSE, input_type))
                else:
                    # no fuse case but broadcast axis of single input is known
                    if self.inputs_before_bro_info[input_type]["all_broadcast_axis"] is not None:
                        no_fuse_case.append(
                            __gen_single_mode_out(ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE,
                                                  input_type,
                                                  self.inputs_before_bro_info[input_type]["shape"],
                                                  self.inputs_before_bro_info[input_type]["range"],
                                                  self.inputs_before_bro_info[input_type]["all_broadcast_axis"])
                        )
                    # unknown broadcast
                    else:
                        no_fuse_case.append(
                            __gen_single_mode_out(ModeType.BROADCAST_UNKNOWN,
                                                  input_type,
                                                  self.inputs_before_bro_info[input_type]["shape"],
                                                  self.inputs_before_bro_info[input_type]["range"])
                        )
                        dict_args = {
                            "errCode": "E90001",
                            "detailed_cause": "norm schedule do not support unknown broadcast now"
                        }
                        raise RuntimeError(dict_args, get_error_message(dict_args))
            no_fuse_case.append(__gen_reduce_axis_out(False))
            total_classify_out.append(no_fuse_case)

        return total_classify_out

    def classify(self):
        """
        generate norm classifier out and add norm pattern
        """
        classify_out = self._gen_classify_out()
        # add norm pattern
        for single_out in classify_out:
            mode_non_duplicate_list = []
            input_type_non_duplicate_list = []
            reduce_axis = None
            broadcast_axis = None
            shape_len = None
            for single_input in single_out:
                if isinstance(single_input, dict):
                    shape_len = len(single_input.get("shape"))
                    mode_str = single_input.get("mode")
                    input_type = single_input.get("input_type")
                    if mode_str == ModeType.BROADCAST_AXIS_KNOWN.value:
                        broadcast_axis = single_input.get("broadcast_axis")
                    if input_type not in input_type_non_duplicate_list:
                        input_type_non_duplicate_list.append(input_type)
                        mode_non_duplicate_list.append(mode_str)
                else:
                    reduce_axis = single_input
            norm_pattern = _cal_norm_pattern(mode_non_duplicate_list, broadcast_axis, reduce_axis, shape_len)
            for single_input in single_out:
                if isinstance(single_input, dict):
                    single_input["norm_pattern"] = norm_pattern

        return classify_out
