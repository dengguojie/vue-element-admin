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
UNKNOWN = "unknown"

MAX_DIM_LEN = 8
REDUCE_PATTERN_KEY_WEIGHT = 1000


def _is_true(expr, dict_args):
    """
    check if true
    """
    if not expr:
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _is_const(shape):
    """
    check const shape
    """
    return all(x > 0 for x in shape)


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


class ReduceAxisType(Enum):
    """
    reduce axis type enum
    """
    ASSIGNED = "assigned"
    ANY = "any"
    SINGLE = "single"
    AFTER = "after"
    BEFORE = "before"


class BroadcastAxisType(Enum):
    """
    reduce axis type enum
    """
    ASSIGNED = "assigned"
    ANY = "any"
    SINGLE = "single"
    AFTER = "after"
    BEFORE = "before"
    SAME_REDUCE = "same_reduce"
    OPPOSITE_REDUCE = "opposite_reduce"


class NormReduceAxis:
    """
    norm reduce axis
    """
    def __init__(self, axis_value, axis_type):
        """
        norm reduce axis init
        """
        self._check(axis_value, axis_type)
        self.value = self._init_value(axis_value)
        self.type = axis_type
        self.is_variable = self.value == UNKNOWN

    @staticmethod
    def _check(axis_value, axis_type):
        """
        check axis value and type
        """
        if axis_type == ReduceAxisType.ASSIGNED:
            _is_true(isinstance(axis_value, (list, tuple)),
                     {"errCode": "E90001",
                      "detailed_cause": f"reduce axis must be a list/tuple when type is {axis_type.value}."})
        elif axis_type in (ReduceAxisType.AFTER, ReduceAxisType.BEFORE):
            _is_true(axis_value == UNKNOWN or isinstance(axis_value, int),
                     {"errCode": "E90001",
                      "detailed_cause": f"reduce axis must be a int or {UNKNOWN} when type is {axis_type.value}."})
        else:
            _is_true(axis_value == UNKNOWN,
                     {"errCode": "E90001",
                      "detailed_cause": f"reduce axis must be {UNKNOWN} when type is {axis_type.value}."})

    @staticmethod
    def _init_value(axis_value):
        """
        value init
        """
        if isinstance(axis_value, (list, tuple)):
            value = list(set(axis_value))
        elif isinstance(axis_value, int):
            value = [axis_value]
        else:
            value = axis_value

        return value

    def convert_value_to_positive(self, max_dim_len):
        """
        convert negative value to positive
        -1 -> -1 + max_dim_len
        """
        if isinstance(self.value, list):
            local_reduce_axis = self.value[:]
            self.value = [x + max_dim_len if x < 0 else x for x in local_reduce_axis]

    def get_max_reduce_axis(self):
        """
        get max reduce axis
        """
        max_reduce_axis = -1
        if isinstance(self.value, list) and len(self.value) >= 1:
            max_reduce_axis = max(self.value)

        return max_reduce_axis


class NormBroadcastAxis:
    """
    norm broadcast axis
    """
    def __init__(self, axis_compile_value, axis_runtime_value, axis_type):
        """
        norm broadcast axis init
        """
        self._check(axis_compile_value, axis_runtime_value, axis_type)
        self.type = axis_type
        self.compile_value = self._init_value(axis_compile_value, mode="compile")
        self.runtime_value = self._init_value(axis_runtime_value, mode="runtime")
        self.is_variable = self.compile_value == UNKNOWN
        self.is_compile_broadcast = not self.runtime_value

    @staticmethod
    def _check(axis_compile_value, axis_runtime_value, axis_type):
        """
        check axis value and type
        """
        if axis_type == BroadcastAxisType.ASSIGNED:
            is_legal_case = \
                isinstance(axis_compile_value, (list, tuple)) and isinstance(axis_runtime_value, (list, tuple))
            _is_true(is_legal_case,
                     {"errCode": "E90001",
                      "detailed_cause": f"broadcast axis must be a list/tuple when type is {axis_type.value}."})
        elif axis_type in (BroadcastAxisType.AFTER, BroadcastAxisType.BEFORE):
            _is_true(axis_compile_value == UNKNOWN or isinstance(axis_compile_value, int),
                     {"errCode": "E90001",
                      "detailed_cause": f"broadcast axis must be a int or {UNKNOWN} when type is {axis_type.value}."})
        else:
            _is_true(axis_compile_value == UNKNOWN,
                     {"errCode": "E90001",
                      "detailed_cause": f"broadcast axis must be {UNKNOWN} when type is {axis_type.value}."})

    def _init_value(self, axis_value, mode):
        """
        value init
        """
        if mode == "runtime" and self.type != BroadcastAxisType.ASSIGNED:
            return []

        if isinstance(axis_value, (list, tuple)):
            value = list(set(axis_value))
        elif isinstance(axis_value, int):
            value = [axis_value]
        else:
            value = axis_value

        return value

    def convert_value_to_positive(self, max_dim_len):
        """
        convert negative value to positive
        -1 -> -1 + max_dim_len
        """
        if isinstance(self.compile_value, list):
            local_compile_axis = self.compile_value[:]
            self.compile_value = [x + max_dim_len if x < 0 else x for x in local_compile_axis]
        if isinstance(self.runtime_value, list):
            local_runtime_axis = self.runtime_value[:]
            self.runtime_value = [x + max_dim_len if x < 0 else x for x in local_runtime_axis]

    def get_max_broadcast_axis(self):
        """
        get max broadcast axis
        """
        max_broadcast_axis = -1
        if isinstance(self.compile_value, list) and len(self.compile_value) >= 1:
            max_broadcast_axis = max(self.compile_value)
        if isinstance(self.runtime_value, list) and len(self.runtime_value) >= 1:
            max_broadcast_axis = max(max(self.runtime_value), max_broadcast_axis)

        return max_broadcast_axis

    def value_is_start_or_end_index(self):
        """
        judge value is start index or end index
        """
        return self.type in (BroadcastAxisType.AFTER, BroadcastAxisType.BEFORE) and not self.is_variable

    def get_compile_broadcast_axis(self):
        """
        get compile broadcast axis
        """
        return self.compile_value if (self.is_compile_broadcast and not self.is_variable) else None


class NormBroadcastAxisList:
    """
    norm broadcast axis list
    """
    def __init__(self, norm_broadcast_axis_list):
        """
        norm broadcast axis list init
        """
        self._check(norm_broadcast_axis_list)
        self.broadcast_axis_list = norm_broadcast_axis_list

    @staticmethod
    def _check(norm_broadcast_axis_list):
        """
        check
        """
        _is_true(isinstance(norm_broadcast_axis_list, list),
                 {"errCode": "E90001",
                  "detailed_cause": "norm_broadcast_axis_list must be a list."})
        for broadcast_axis_obj in norm_broadcast_axis_list:
            _is_true(isinstance(broadcast_axis_obj, NormBroadcastAxis),
                     {"errCode": "E90001",
                      "detailed_cause": "element in norm_broadcast_axis_list must be NormBroadcastAxis object."})

    def convert_value_to_positive(self, max_dim_len):
        """
        convert negative value of element in broadcast axis list to positive
        """
        for broadcast_axis_obj in self.broadcast_axis_list:
            broadcast_axis_obj.convert_value_to_positive(max_dim_len)

    def get_max_broadcast_axis(self):
        """
        get max broadcast axis of all element in broadcast axis list
        """
        max_broadcast_axis = -1
        for broadcast_axis_obj in self.broadcast_axis_list:
            max_broadcast_axis = max(broadcast_axis_obj.get_max_broadcast_axis(), max_broadcast_axis)

        return max_broadcast_axis

    def get_broadcast_type(self):
        """
        get broadcast type of element in broadcast axis list
        """
        return [x.type for x in self.broadcast_axis_list]

    def exist_variable_broadcast_axis(self):
        """
        exist variable broadcast axis or not
        """
        if len(self.broadcast_axis_list) < 1:
            return False

        return any(x.is_variable for x in self.broadcast_axis_list)


class NormClassifyInfo:
    """
    norm classify info
    """
    def __init__(self, input_num, reduce_axis, extra_params):
        """
        norm classify info init
        """
        self._check_extra_params(extra_params)
        self.input_num = input_num
        self.extra_params = extra_params
        self.is_fuse_axis = self._parse_is_fuse_axis()
        self.disable_fuse_axes = self._parse_disable_fuse_axes()
        self.reduce_axis_obj = self._parse_reduce_axis(reduce_axis)
        self.input_type_list = self._parse_input_type()
        self.broadcast_axis_list_obj = self._parse_broadcast_axis_list()
        self._check_current_axis_types_is_supported()
        self._add_input_type_list_to_compile_info()
        self._add_reduce_axis_to_compile_info()
        self._add_broadcast_axis_to_compile_info()

    @staticmethod
    def _check_extra_params(extra_params):
        """
        check extra_params
        """
        _is_true(extra_params is None or isinstance(extra_params, dict),
                 {"errCode": "E90001",
                  "detailed_cause": "extra_params must be a dict or None when mode is norm."})

    @staticmethod
    def _check_disable_fuse_axes(disable_fuse_axes):
        """
        check disable_fuse_axes
        """
        _is_true(all(x >= 0 for x in disable_fuse_axes),
                 {"errCode": "E90001",
                  "detailed_cause": "element in disable_fuse_axes must be positive."})

    def _check_input_type(self, input_type_list):
        """
        check input_type
        """
        _is_true(isinstance(input_type_list, (list, tuple)) and len(input_type_list) == self.input_num,
                {"errCode": "E90001",
                 "detailed_cause": "input_shape_type in norm classifier must be a list/tuple and"
                                   "its len must be equal to input num."})
        for single_value in input_type_list:
            _is_true(single_value in (0, 1),
                     {"errCode": "E90001",
                      "detailed_cause": "value in input_shape_type in norm classifier must be 0 or 1."})

    @staticmethod
    def _check_same_input_shape_group(same_input_shape_group):
        """
        check same_input_shape_group
        """
        _is_true(isinstance(same_input_shape_group, (list, tuple)) and len(same_input_shape_group) != 0,
                 {"errCode": "E90001",
                  "detailed_cause": "same_input_shape_group in norm classifier must be a dual list."})
        for single_group in same_input_shape_group:
            _is_true(isinstance(single_group, (list, tuple)),
                     {"errCode": "E90001",
                      "detailed_cause": "element in same_input_shape_group must be a list."})

    @staticmethod
    def _check_broadcast(key, value):
        """
        check broadcast
        """
        _is_true(isinstance(value, dict),
                 {"errCode": "E90001",
                  "detailed_cause": f"{key} in norm classifier must be a dict."})

    def _parse_is_fuse_axis(self):
        """
        parse is_fuse_axis
        """
        is_fuse_axis = True
        if self.extra_params is not None:
            if "disable_optimization" in self.extra_params:
                is_fuse_axis = not self.extra_params.get("disable_optimization")
        add_compile_info_inner("_fuse_axis", is_fuse_axis)

        return is_fuse_axis

    def _parse_disable_fuse_axes(self):
        """
        parse disable_fuse_axes
        """
        if self.extra_params is None:
            return []

        disable_fuse_axes = []
        if "disable_fuse_axes" in self.extra_params:
            disable_fuse_axes = self.extra_params.get("disable_fuse_axes")
            self._check_disable_fuse_axes(disable_fuse_axes)
            add_compile_info_inner("_disable_fuse_axes", disable_fuse_axes)

        return disable_fuse_axes

    def _parse_reduce_axis(self, reduce_axis_value):
        """
        parse reduce_axis
        """
        reduce_axis_type = ReduceAxisType.ASSIGNED
        if self.extra_params is not None:
            if "reduce_axes_type" in self.extra_params:
                reduce_axis_type = ReduceAxisType(self.extra_params.get("reduce_axes_type"))

        return NormReduceAxis(reduce_axis_value, reduce_axis_type)

    def _parse_input_type(self):
        """
        parse input_type
        """
        ori_input_type_list = [0] * self.input_num

        if self.extra_params is None:
            return ori_input_type_list

        ori_same_input_shape_group_list = [list(range(self.input_num))]
        if "input_shape_type" in self.extra_params:
            value = self.extra_params.get("input_shape_type")
            self._check_input_type(value)
            ori_input_type_list = value
        if "same_input_shape_group" in self.extra_params:
            value = self.extra_params.get("same_input_shape_group")
            self._check_same_input_shape_group(value)
            ori_same_input_shape_group_list = value

        input_type_list = [0] * self.input_num
        input_type_flag = 0
        complete_shape_index = [i for i in range(self.input_num) if ori_input_type_list[i] == 0]
        for single_same_input_shape_group in ori_same_input_shape_group_list:
            if not single_same_input_shape_group:
                continue
            # input in same_input_shape_group is complete shape
            if set(single_same_input_shape_group) & set(complete_shape_index):
                continue
            input_type_flag += 1
            for single_index in single_same_input_shape_group:
                _is_true(input_type_list[single_index] == 0,
                         {"errCode": "E90001",
                          "detailed_cause": "norm classifier do not support "
                                            "same_input_shape_group with intersection."})
                input_type_list[single_index] = input_type_flag

        input_type_flag += 1
        for index in range(self.input_num):
            remain_partial_shape = ori_input_type_list[index] == 1 and input_type_list[index] == 0
            if remain_partial_shape:
                input_type_list[index] = input_type_flag
                input_type_flag += 1

        _is_true(len(set(input_type_list)) <= 4,
                 {"errCode": "E90001",
                  "detailed_cause": "Currently, a maximum of four input types are supported in norm schedule."})

        return input_type_list

    def _parse_broadcast_axis_list(self):
        """
        parse broadcast_axis_list
        """
        if self.extra_params is None:
            return NormBroadcastAxisList([])

        ori_compile_broadcast_axes = {}
        ori_runtime_broadcast_axes = {}
        ori_broadcast_axes_type = {}

        if "compile_broadcast_axes" in self.extra_params:
            value = self.extra_params.get("compile_broadcast_axes")
            self._check_broadcast("compile_broadcast_axes", value)
            ori_compile_broadcast_axes = value
        if "runtime_broadcast_axes" in self.extra_params:
            value = self.extra_params.get("runtime_broadcast_axes")
            self._check_broadcast("runtime_broadcast_axes", value)
            ori_runtime_broadcast_axes = value
        if "broadcast_axes_type" in self.extra_params:
            value = self.extra_params.get("broadcast_axes_type")
            self._check_broadcast("broadcast_axes_type", value)
            ori_broadcast_axes_type = value

        broadcast_axis_list = []
        for input_index in range(self.input_num):
            if self.input_type_list[input_index] == 0:
                 continue
            single_compile_broadcast_axis = ori_compile_broadcast_axes.get(input_index, [])
            single_runtime_broadcast_axis = ori_runtime_broadcast_axes.get(input_index, [])
            single_broadcast_axis_type = BroadcastAxisType(ori_broadcast_axes_type.get(input_index, "assigned"))
            broadcast_axis_obj = NormBroadcastAxis(single_compile_broadcast_axis, single_runtime_broadcast_axis,
                                                   single_broadcast_axis_type)
            broadcast_axis_list.append(broadcast_axis_obj)

        return NormBroadcastAxisList(broadcast_axis_list)

    def _check_current_axis_types_is_supported(self):
        """
        check axis types combination is supported
        """
        support_types = [
            {ReduceAxisType.SINGLE, BroadcastAxisType.OPPOSITE_REDUCE, BroadcastAxisType.SAME_REDUCE},
            {ReduceAxisType.AFTER, BroadcastAxisType.OPPOSITE_REDUCE, BroadcastAxisType.SAME_REDUCE},
            {ReduceAxisType.ASSIGNED, BroadcastAxisType.ASSIGNED}
        ]
        current_type_set = set(self.broadcast_axis_list_obj.get_broadcast_type())
        current_type_set.add(self.reduce_axis_obj.type)
        is_supported = False
        for single_support_type in support_types:
            if single_support_type.issuperset(current_type_set):
                is_supported = True
                break

        _is_true(is_supported,
                 {"errCode": "E90001",
                  "detailed_cause": "current types of reduce and broadcast axis "
                                    "are not supported in norm classifier now."})
        _is_true(not self.disable_fuse_axes or
                 {ReduceAxisType.ASSIGNED, BroadcastAxisType.ASSIGNED}.issuperset(current_type_set),
                 {"errCode": "E90001",
                  "detailed_cause": "current types of reduce and broadcast axis with disable_fuse_axes"
                                    "are not supported in norm classifier now."})

    def _add_input_type_list_to_compile_info(self):
        """
        add input_type_list to compile_info
        """
        add_compile_info_inner("_input_type", self.input_type_list)

    def _add_reduce_axis_to_compile_info(self):
        """
        add reduce_axis to compile_info
        """
        if self.reduce_axis_obj.type == ReduceAxisType.ASSIGNED:
            add_compile_info_inner("_ori_reduce_axis", self.reduce_axis_obj.value[:])
        elif self.reduce_axis_obj.type == ReduceAxisType.AFTER:
            if self.reduce_axis_obj.value != UNKNOWN:
                add_compile_info_inner("_ori_reduce_axis", self.reduce_axis_obj.value[:])
            add_compile_info_inner("_reduce_axis_type", 3)
        elif self.reduce_axis_obj.type == ReduceAxisType.BEFORE:
            if self.reduce_axis_obj.value != UNKNOWN:
                add_compile_info_inner("_ori_reduce_axis", self.reduce_axis_obj.value[:])
            add_compile_info_inner("_reduce_axis_type", 4)
        else:
            add_compile_info_inner("_reduce_axis_type", 9)

    def _add_broadcast_axis_to_compile_info(self):
        """
        add broadcast_axis to compile_info
        """
        input_type_non_duplicate_list = []
        broadcast_axis_type_list = []
        count = 0
        for single_input_type in self.input_type_list:
            if single_input_type in input_type_non_duplicate_list:
                count += 1
                continue
            if single_input_type == 0:
                continue
            broadcast_axis_obj = self.broadcast_axis_list_obj.broadcast_axis_list[count]
            input_type_non_duplicate_list.append(single_input_type)
            count += 1
            if broadcast_axis_obj.type == BroadcastAxisType.ASSIGNED:
                continue
            elif broadcast_axis_obj.type == BroadcastAxisType.SAME_REDUCE:
                broadcast_axis_type_list.append(1)
            elif broadcast_axis_obj.type == BroadcastAxisType.OPPOSITE_REDUCE:
                broadcast_axis_type_list.append(2)
            elif broadcast_axis_obj.type == BroadcastAxisType.AFTER:
                broadcast_axis_type_list.append(3)
            elif broadcast_axis_obj.type == BroadcastAxisType.BEFORE:
                broadcast_axis_type_list.append(4)
            else:
                broadcast_axis_type_list.append(9)

        if broadcast_axis_type_list:
            add_compile_info_inner("_broadcast_axis_type_list", broadcast_axis_type_list)


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


def _infer_negative_two_and_pre_process(ins: dict, classify_info: NormClassifyInfo):
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
                _is_true(_shape_tuple == (-2, ),
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
    classify_info.reduce_axis_obj.convert_value_to_positive(dim_len)
    classify_info.broadcast_axis_list_obj.convert_value_to_positive(dim_len)

    max_reduce_axis = classify_info.reduce_axis_obj.get_max_reduce_axis()
    max_broadcast_max_axis = classify_info.broadcast_axis_list_obj.get_max_broadcast_axis()
    min_dim_len = max(max_reduce_axis, max_broadcast_max_axis) + 1

    _is_true(min_dim_len <= dim_len,
             {"errCode": "E90001",
              "detailed_cause": "max of reduce axis in norm classifier should be less than dim len."})

    if not exist_negative_two:
        return ins_list

    if classify_info.reduce_axis_obj.is_variable or \
            classify_info.broadcast_axis_list_obj.exist_variable_broadcast_axis():
        return ins_list

    _is_true(not classify_info.disable_fuse_axes,
             {"errCode": "E90001",
              "detailed_cause": "norm classifier with disable_fuse_axes don't support -2."})

    max_dim_len = min_dim_len if classify_info.is_fuse_axis else dim_len
    # input are all -2
    # include last reduce and nlast reduce
    for opt_dim_len in range(min_dim_len, max_dim_len + 1):
        if opt_dim_len == dim_len:
            continue
        local_ins = copy.deepcopy(ins)
        for single_input in local_ins:
            if tuple(single_input.get("shape")) == (-2, ):
                single_input["shape"] = [-1] * opt_dim_len
                single_input["range"] = [(1, None) for _ in range(opt_dim_len)]
        ins_list.append(local_ins)

    return ins_list


def _judge_is_const_and_dynamic_mixed(norm_classify_out):
    # const process can not be performed when dynamic and const are mixed in norm classify output
    is_const_set = set()
    for single_classify_out in norm_classify_out:
        is_const = True
        for single_input in single_classify_out:
            if isinstance(single_input, dict):
                is_const = is_const and _is_const(single_input.get("shape"))
        is_const_set.add(is_const)
    is_const_and_dynamic_mixed = len(is_const_set) == 2

    return is_const_and_dynamic_mixed


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
        "compile_broadcast_axes": dict, key is input index, value is compile broadcast axis(list).
                                  Default is common axis.
        "runtime_broadcast_axes": dict, key is input index, value is runtime broadcast axis(list).
                                  Default is common axis.
        "disable_fuse_axes": [input: list, output: list[list]].
                             The input list consists of the indices of the non-fusible axes.
                             Each list in the output list is the new index of the non-fusible axes
                             after the fused axes in the corresponding scene.
        "reduce_axes_type": str, the type of reduce axis, support "single"(single reduce axis) and
                            "after"(after this axis). At the same time, reduce_axis must be "unknown".
                            Operators can add reduce axis attr name and dtype to compile info. Example:
                            add_compile_info("reduce_axis_attr_name", "axis")
                            add_compile_info("reduce_axis_attr_dtype", "ListInt")
        "broadcast_axes_type":str, the type of broadcast axis, support "opposite_reduce" and
                              "same_reduce", At the same time, compile_broadcast_axes must be "unknown".
    :return:
    """
    get_context().set_pattern(NORM)

    _is_true(len(ins) >= 1,
             {"errCode": "E90001",
              "detailed_cause": "length of inputs in norm classifier must be no less than 1."})
    inputs = ins[:-1]
    ops_reduce_axis = ins[-1]

    classify_info = NormClassifyInfo(len(inputs), ops_reduce_axis, extra_params)
    ins_list = _infer_negative_two_and_pre_process(inputs, classify_info)

    disable_fuse_axes_after_classify = []
    norm_classify_out = []
    norm_pattern_non_duplicate_list = []

    for single_ins in ins_list:
        norm_classifier = NormClassifier(single_ins, classify_info)
        classify_out = norm_classifier.classify()
        # When all inputs have the same determinable broadcast axis, the broadcast axis is unify_broadcast_axis;
        # In other cases, unify_broadcast_axis is None
        unify_broadcast_axis = norm_classifier.unify_broadcast_axis
        # all inputs have the same determinable broadcast axis
        if unify_broadcast_axis is not None:
            unify_broadcast_axis = list(unify_broadcast_axis)
            add_compile_info_inner("_ori_broadcast_axis", unify_broadcast_axis[:])

        disable_fuse_axes_list = norm_classifier.get_disable_fuse_axes_list()
        for idx, single_classify_out in enumerate(classify_out):
            # if the same pattern exists in norm classify output, this single output should be deserted
            norm_pattern = single_classify_out[0].get("norm_pattern")
            if norm_pattern not in norm_pattern_non_duplicate_list:
                norm_classify_out.append(single_classify_out)
                norm_pattern_non_duplicate_list.append(norm_pattern)
                if classify_info.disable_fuse_axes:
                    disable_fuse_axes_after_classify.append(disable_fuse_axes_list[idx])

    is_const_and_dynamic_mixed = _judge_is_const_and_dynamic_mixed(norm_classify_out)
    get_context().add("_const_and_dynamic_mixed", is_const_and_dynamic_mixed)
    if classify_info.disable_fuse_axes:
        extra_params.update({"disable_fuse_axes": disable_fuse_axes_after_classify})

    return norm_classify_out


class NormClassifier:
    """
    norm classifier
    """
    def __init__(self, ins: list, classify_info: NormClassifyInfo):
        """
        norm classifier init
        """
        self.ins = ins
        self.classify_info = classify_info
        self.pad_axis_index = 0
        self.max_shape_len = max(len(x.get("shape")) for x in ins)

        self.reduce_axis = classify_info.reduce_axis_obj.value
        self.unreduce_axis = None
        self.broadcast_axis_list = classify_info.broadcast_axis_list_obj.broadcast_axis_list

        self.disable_fuse_axes_after_fuse = []
        self.disable_fuse_axes_list = []

        self.inputs_after_bro, self.inputs_before_bro, self.inputs_bro_axis = self._inputs_classify()
        self.shape_before_fuse, self.range_before_fuse, self.inputs_before_bro_info = self._infer_and_extract_info()
        self.reduce_and_broadcast_combination = self._handle_variable_axis()
        self.unify_broadcast_axis = None
        self.shape_after_fuse = None
        self.range_after_fuse = None
        self.reduce_axis_after_fuse = None
        self.broadcast_axis_after_fuse = None

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
        count_before_broadcast = 0
        for index, input_type in enumerate(self.classify_info.input_type_list):
            if input_type == 0:
                inputs_after_bro.append(self.ins[index])
            else:
                if input_type in inputs_before_bro:
                    inputs_before_bro.get(input_type).append(self.ins[index])
                    inputs_bro_axis.get(input_type).append(self.broadcast_axis_list[count_before_broadcast])
                else:
                    inputs_before_bro[input_type] = [self.ins[index]]
                    inputs_bro_axis[input_type] = [self.broadcast_axis_list[count_before_broadcast]]
                count_before_broadcast += 1

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
            _shape_list = [x.get("shape") for x in _inputs]
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

            for _index, _ in enumerate(_shape_out):
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
            local_shape_list, local_range_list = __get_shape_and_range_list(self.inputs_before_bro.get(input_type))
            local_shape, local_range = __infer_shape_and_range(local_shape_list, local_range_list)
            # all_broadcast_axis means all broadcast axes of input are known
            # partial_broadcast_axis means partial broadcast axes can be inferred
            # partial_unbroadcast_axis means partial unbroadcast axes can be inferred
            all_broadcast_axis, partial_broadcast_axis, partial_unbroadcast_axis = \
                __infer_broadcast_axis(local_shape, local_range,
                                       [x.get_compile_broadcast_axis() for x in self.inputs_bro_axis.get(input_type)])
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
            before_bro_shape_list = [inputs_before_bro_info.get(x).get("shape") for x in inputs_before_bro_info]
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

    def _handle_variable_axis(self):
        """
        handle variable reduce axis or broadcast axis
        """
        def __gen_opposite_axis(_ori_axis, _total_dim_len):
            return sorted(set(range(_total_dim_len)) - set(_ori_axis))

        def __gen_broadcast_enum_values(_reduce_axis, _broadcast_type):
            broadcast_axis_type_and_enum_values_map = {
                BroadcastAxisType.SAME_REDUCE: [_reduce_axis],
                BroadcastAxisType.OPPOSITE_REDUCE: [__gen_opposite_axis(_reduce_axis, max_dim + 1)],
                BroadcastAxisType.SINGLE: [_reduce_axis] + [[max_dim // 2 - 1], [max_dim // 2 + 1]],
                BroadcastAxisType.AFTER: [_reduce_axis] +
                                         [list(range(mid_dim - 1, max_dim + 1)), list(range(mid_dim, max_dim + 1)),
                                          list(range(mid_dim + 1, max_dim + 1))],
                BroadcastAxisType.BEFORE: [_reduce_axis] +
                                          [list(range(mid_dim - 1)), list(range(mid_dim)), list(range(mid_dim + 1))]
            }

            return broadcast_axis_type_and_enum_values_map.get(_broadcast_type)

        max_dim = self.max_shape_len - 1
        mid_dim = self.max_shape_len // 2
        reduce_axis_type_and_enum_values_map = {
            ReduceAxisType.SINGLE: [[0], [mid_dim], [max_dim]],
            ReduceAxisType.AFTER: [list(range(max_dim + 1)), list(range(mid_dim, max_dim + 1)), [max_dim]],
            ReduceAxisType.BEFORE: [[0], list(range(mid_dim)), list(range(max_dim + 1))]
        }
        reduce_axis_type = self.classify_info.reduce_axis_obj.type
        if not self.classify_info.reduce_axis_obj.is_variable:
            reduce_axis_enum_values = [self.reduce_axis]
        else:
            reduce_axis_enum_values = reduce_axis_type_and_enum_values_map.get(reduce_axis_type)

        output_axis_list = []
        for single_reduce_axis_value in reduce_axis_enum_values:
            output_axis = [[single_reduce_axis_value]]
            for input_type in self.inputs_bro_axis:
                broadcast_axis_obj = self.inputs_bro_axis.get(input_type)[0]
                if broadcast_axis_obj.is_variable:
                    broadcast_axis_enum_values = __gen_broadcast_enum_values(single_reduce_axis_value,
                                                                             broadcast_axis_obj.type)
                elif broadcast_axis_obj.value_is_start_or_end_index():
                    broadcast_axis_enum_values = [broadcast_axis_obj.compile_value]
                else:
                    broadcast_axis_enum_values = [self.inputs_before_bro_info.get(input_type).get("all_broadcast_axis")]
                output_axis.append(broadcast_axis_enum_values)

            output_axis_list.extend(list(product(*output_axis)))

        return output_axis_list

    def _get_unify_broadcast_axis(self):
        """
        obtain the broadcast axis when all broadcast axes of inputs are the same
        """
        broadcast_axis_set = set()
        broadcast_axis_list = None
        for input_type in self.inputs_before_bro_info:
            single_broadcast_axis = self.inputs_before_bro_info.get(input_type).get("all_broadcast_axis")
            if single_broadcast_axis is None:
                return broadcast_axis_list
            broadcast_axis_set.add(tuple(sorted(single_broadcast_axis)))
        if len(broadcast_axis_set) == 1:
            broadcast_axis_list = list(broadcast_axis_set)[0]

        return broadcast_axis_list

    def _simplify(self):
        """
        simplify shape, range, reduce axis, broadcast axis
        fuse continuous reduce axis or broadcast aixs or common axis or reduce and broadcast axis.
        """
        def __obtain_state():
            if is_pad_axis:
                self.pad_axis_index += 1
                return "pad_" + str(self.pad_axis_index)
            if is_reduce_axis and not is_broadcast_axis:
                return "reduce"
            if is_broadcast_axis and not is_reduce_axis:
                return "broadcast"
            if is_broadcast_axis and is_reduce_axis:
                return "reduce_and_broadcast"

            return "common"

        if not self.classify_info.is_fuse_axis:
            return self.shape_before_fuse, self.range_before_fuse, self.reduce_axis, self.unify_broadcast_axis

        f_shape, f_ranges, f_reduce_axis, f_broadcast_axis, f_pad_axis = [], [], [], [], []
        state = "init"
        for i, (d, r) in enumerate(zip(self.shape_before_fuse, self.range_before_fuse)):
            is_pad_axis = i in self.classify_info.disable_fuse_axes
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

            if is_pad_axis:
                f_pad_axis.append(len(f_shape) - 1)

            state = state_i

        self.disable_fuse_axes_after_fuse.append(f_pad_axis)

        return f_shape, f_ranges, f_reduce_axis, f_broadcast_axis

    def _judge_remove_last_one(self):
        """
        judge whether to remove last one
        """
        # the following is the case that can remove one axis
        # 1. must have after broadcast input
        # 2. is_fuse_axis is true
        # 3. broadcast axis is known when has before broadcast input
        # 4. dont have disable_fuse_axes
        # return: is append remove last one case and is desert origin case
        is_no_after_broadcast_input = 0 not in self.classify_info.input_type_list
        is_no_fuse_axis = not self.classify_info.is_fuse_axis
        is_disable_fuse_axes = len(self.classify_info.disable_fuse_axes) > 0
        is_unify_broadcast_axis_unknown = \
            self.unify_broadcast_axis is None and \
            self.classify_info.input_type_list != [0] * len(self.classify_info.input_type_list)

        is_cannot_remove_one = is_no_after_broadcast_input or is_no_fuse_axis or \
            is_unify_broadcast_axis_unknown or is_disable_fuse_axes
        if is_cannot_remove_one:
            return False, False
        if len(self.shape_after_fuse) <= 1:
            return False, False

        last_dim = self.shape_after_fuse[-1]
        last_range = self.range_after_fuse[-1]
        # append remove last one case and desert origin case
        if last_dim == 1:
            return True, True
        # don't append remove last one case
        elif last_dim > 1 or last_range[0] > 1:
            return False, False
        # append remove last one case
        return True, False

    def _gen_remove_last_one(self):
        """
        generate remove last one case
        """
        last_index = len(self.shape_after_fuse) - 1

        shape_after_fuse = self.shape_after_fuse[:]
        range_after_fuse = self.range_after_fuse[:]
        reduce_axis_after_fuse = self.reduce_axis_after_fuse[:]
        broadcast_axis_after_fuse = self.broadcast_axis_after_fuse[:]

        shape_after_fuse.pop()
        range_after_fuse.pop()
        ori_broadcast_axis_is_empty = not broadcast_axis_after_fuse
        if last_index in reduce_axis_after_fuse:
            reduce_axis_after_fuse.remove(last_index)
        if last_index in broadcast_axis_after_fuse:
            broadcast_axis_after_fuse.remove(last_index)
        # if the reduce axis is empty after removing one axis, a R axis should be made up on the front
        # shape: (A, ) -> (1, A)
        # reduce_axis: () -> (0, )
        # broadcast_axis: (m, ) -> (0, m + 1)
        if not reduce_axis_after_fuse:
            shape_after_fuse.insert(0, 1)
            range_after_fuse.insert(0, (1, 1))
            reduce_axis_after_fuse.insert(0, 0)
            for index, axis in enumerate(broadcast_axis_after_fuse):
                broadcast_axis_after_fuse[index] = axis + 1
            if not ori_broadcast_axis_is_empty:
                broadcast_axis_after_fuse.insert(0, 0)

        self.shape_after_fuse = shape_after_fuse
        self.range_after_fuse = range_after_fuse
        self.reduce_axis_after_fuse = reduce_axis_after_fuse
        self.broadcast_axis_after_fuse = broadcast_axis_after_fuse

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

                return [ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE], False, True
            # infer mode
            _enum_broadcast_unknown = not len(self.reduce_axis) == len(self.unreduce_axis) == 1

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
                for __index, _ in enumerate(__ori_value_list):
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
                                                              (1, 1))
                _out_broadcast_axis = self.reduce_axis_after_fuse[:]
            elif _mode == ModeType.BROADCAST_REDUCE_OPPOSITE:
                _out_shape = ___gen_single_out_shape_or_range(self.shape_after_fuse, self.reduce_axis_after_fuse, 1,
                                                              False)
                _out_range = ___gen_single_out_shape_or_range(self.range_after_fuse, self.reduce_axis_after_fuse,
                                                              (1, 1), False)
                _out_broadcast_axis = sorted(set(range(len(self.shape_after_fuse))) -
                                             set(self.reduce_axis_after_fuse))
            elif _mode == ModeType.BROADCAST_UNKNOWN:
                _out_shape = _ori_before_bro_shape[:]
                _out_range = _ori_before_bro_range[:]
            elif _mode == ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE:
                _out_shape = ___gen_single_out_shape_or_range(self.shape_before_fuse, _single_broadcast_axis, 1)
                _out_range = ___gen_single_out_shape_or_range(self.range_before_fuse, _single_broadcast_axis,
                                                              (1, 1))
                _out_broadcast_axis = _single_broadcast_axis[:]
            elif _mode == ModeType.BROADCAST_AXIS_KNOWN:
                _out_shape = ___gen_single_out_shape_or_range(self.shape_after_fuse, self.broadcast_axis_after_fuse,
                                                              1)
                _out_range = ___gen_single_out_shape_or_range(self.range_after_fuse, self.broadcast_axis_after_fuse,
                                                              (1, 1))
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

            if set(_input_mode_list) & {ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE, ModeType.BROADCAST_UNKNOWN}\
                    and ModeType.COMMON in _input_mode_list:
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
                    _input_dim = [x.get("shape")[_idx] for x in _input_dict]
                    # all broadcast dims are not equal to 1
                    if not min(_input_dim) == max(_input_dim) == 1:
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
                __infer_before_broadcast_mode(
                    self.inputs_before_bro_info.get(input_type).get("all_broadcast_axis"),
                    self.inputs_before_bro_info.get(input_type).get("partial_broadcast_axis"),
                    self.inputs_before_bro_info.get(input_type).get("partial_unbroadcast_axis"))
            exist_unknown_broadcast_case = exist_unknown_broadcast_case or flag_of_unknown_broadcast
            exist_no_fuse_case = exist_no_fuse_case or flag_of_no_fuse
            single_input_out = []
            for mode in mode_list:
                single_input_out.append(
                    __gen_single_mode_out(
                        mode,
                        input_type,
                        self.inputs_before_bro_info.get(input_type).get("shape"),
                        self.inputs_before_bro_info.get(input_type).get("range"),
                        self.inputs_before_bro_info.get(input_type).get("all_broadcast_axis"))
                )
            before_broadcast_enum_out[input_type] = single_input_out

        after_broadcast_enum_out = [__gen_single_mode_out(ModeType.COMMON, 0)]

        product_in = [before_broadcast_enum_out.get(index) for index in before_broadcast_enum_out]
        product_in.append(after_broadcast_enum_out)
        input_type_order_list.append(0)
        # combine all possibilities for each type of input
        product_out = list(product(*product_in))

        total_classify_out = []
        for product_item in product_out:
            local_product_item = copy.deepcopy(product_item)
            single_classify_out = []
            for input_type in self.classify_info.input_type_list:
                idx = input_type_order_list.index(input_type)
                single_classify_out.append(list(local_product_item)[idx])
            if __pruning(single_classify_out):
                continue
            single_classify_out.append(__gen_reduce_axis_out(True))
            total_classify_out.append(single_classify_out)

        # add no fuse case
        if exist_unknown_broadcast_case or exist_no_fuse_case:
            no_fuse_case = []
            for _, input_type in enumerate(self.classify_info.input_type_list):
                if input_type == 0:
                    no_fuse_case.append(__gen_single_mode_out(ModeType.NO_FUSE, input_type))
                else:
                    # no fuse case but broadcast axis of single input is known
                    if self.inputs_before_bro_info.get(input_type).get("all_broadcast_axis") is not None:
                        no_fuse_case.append(
                            __gen_single_mode_out(
                                ModeType.SINGLE_BROADCAST_KNOWN_AND_NO_FUSE,
                                input_type,
                                self.inputs_before_bro_info.get(input_type).get("shape"),
                                self.inputs_before_bro_info.get(input_type).get("range"),
                                self.inputs_before_bro_info.get(input_type).get("all_broadcast_axis"))
                        )
                    # unknown broadcast
                    else:
                        no_fuse_case.append(
                            __gen_single_mode_out(ModeType.BROADCAST_UNKNOWN,
                                                  input_type,
                                                  self.inputs_before_bro_info.get(input_type).get("shape"),
                                                  self.inputs_before_bro_info.get(input_type).get("range"))
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
        def _add_pattern_to_output(_classify_out):
            for single_out in _classify_out:
                self.disable_fuse_axes_list.extend(self.disable_fuse_axes_after_fuse)
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

        total_classify_out = []
        for single_axis_combination in self.reduce_and_broadcast_combination:
            self.reduce_axis = single_axis_combination[0][:]
            self.unreduce_axis = sorted(set(range(self.max_shape_len)) - set(self.reduce_axis))
            broadcast_axis_index = 1
            for single_input_type in self.inputs_before_bro_info:
                single_broadcast_axis = None if single_axis_combination[broadcast_axis_index] is None\
                    else single_axis_combination[broadcast_axis_index][:]
                self.inputs_before_bro_info.get(single_input_type).update(
                    {"all_broadcast_axis": single_broadcast_axis})
                broadcast_axis_index += 1
            self.unify_broadcast_axis = self._get_unify_broadcast_axis()
            self.shape_after_fuse, self.range_after_fuse,\
                self.reduce_axis_after_fuse, self.broadcast_axis_after_fuse = self._simplify()
            is_append_remove_one, is_desert_origin = self._judge_remove_last_one()
            if not is_desert_origin:
                classify_out = self._gen_classify_out()
                _add_pattern_to_output(classify_out)
                total_classify_out.extend(classify_out)
            if is_append_remove_one:
                self._gen_remove_last_one()
                classify_out = self._gen_classify_out()
                _add_pattern_to_output(classify_out)
                total_classify_out.extend(classify_out)

        return total_classify_out

    def get_disable_fuse_axes_list(self):
        """
        get disable_fuse_axes_list
        """
        return self.disable_fuse_axes_list
