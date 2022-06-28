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
classifier of shape in slice
"""
from itertools import chain

from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

from . import shape_classifier

MAX_DIM_LEN = 8
INPUT_LEN = 3
INPUT_X_IDX = 0
INPUT_BEGIN_IDX = 1
INPUT_END_IDX = 2
END_MODE = "end_mode"
MODE_SIZE = "size"
MODE_END = "end"
END_MODE_MAP = {
    MODE_SIZE: 0,
    MODE_END: 1
}


@shape_classifier.register_classifier(shape_classifier.SLICE)
def classify_slice(ins: list, extra_params: dict):
    """
    Slice classifier
    :param ins: A list constains tensor, begin, size or end.
    :param extra_params: A dict contains end_mode. End_mode can be size or end.
    :return:
    """
    return SliceClassifier(ins, extra_params).classify()


class SliceClassifier:
    def __init__(self, ins: list, extra_params: dict):
        """
        :param ins: input list
        """
        self.mode = extra_params.get(END_MODE, MODE_END)
        operation.get_context().add("_end_mode", END_MODE_MAP.get(self.mode))

        self.check_input(ins)

        # check dynamic or static
        self.is_static = operation.get_op_mode() == "static"

        tensor_nums = self.get_tensor_inputs_idx(ins)

        self.is_const = operation.get_op_mode() == "dynamic" and \
                                 len(tensor_nums) == 1 and \
                                 -1 not in self.x_shape and \
                                 -2 not in self.x_shape

        operation.get_context().add("_is_const", self.is_const)

        if self.is_static or self.is_const:
            if "const_value" in ins[INPUT_BEGIN_IDX]:
                self.is_begin_list = True
                self.begin_list = list(ins[INPUT_BEGIN_IDX].get("const_value"))
                self.check_list_len(self.begin_list, len(self.x_shape))

            if "const_value" in ins[INPUT_END_IDX]:
                self.is_end_list = True
                if self.mode == MODE_END:
                    self.end_list = list(ins[INPUT_END_IDX].get("const_value"))
                else:
                    self.size_list = list(ins[INPUT_END_IDX].get("const_value"))
                    self.end_list = self._cal_end_by_size(self.begin_list, self.size_list)
                    self.check_list_len(self.end_list, len(self.x_shape))
        else:
            self.begin_shape = list(ins[INPUT_BEGIN_IDX].get("shape"))
            self.begin_dtype = ins[INPUT_BEGIN_IDX].get("dtype")
            self.begin_range = ins[INPUT_BEGIN_IDX].get("range")
            if INPUT_BEGIN_IDX not in tensor_nums:
                operation.get_context().add("_const_begins", list(ins[INPUT_BEGIN_IDX].get("const_value")))

            self.end_shape = list(ins[INPUT_END_IDX].get("shape"))
            self.end_dtype = ins[INPUT_END_IDX].get("dtype")
            self.end_range = ins[INPUT_END_IDX].get("range")
            if INPUT_END_IDX not in tensor_nums:
                third_input_type = "_const_ends" if self.mode == MODE_END else "_const_sizes"
                operation.get_context().add(third_input_type, list(ins[INPUT_END_IDX].get("const_value")))

    @staticmethod
    def get_tensor_inputs_idx(ins):
        tmp_tensor_nums = [INPUT_X_IDX]
        if isinstance(ins[INPUT_BEGIN_IDX], dict) and "const_value" not in ins[INPUT_BEGIN_IDX]:
            tmp_tensor_nums.append(INPUT_BEGIN_IDX)
        if isinstance(ins[INPUT_END_IDX], dict) and "const_value" not in ins[INPUT_END_IDX]:
            tmp_tensor_nums.append(INPUT_END_IDX)
        return tmp_tensor_nums

    @staticmethod
    def check_list_len(input_list, list_len):
        if len(input_list) != list_len:
            dict_args = {}
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "the slice begin or end list must be same as x shape length"
            raise RuntimeError(dict_args, get_error_message(dict_args))

    @staticmethod
    def construct_static_input_dict(input_shape, input_dtype, input_range):
        input_dict = {}
        input_dict["shape"] = input_shape
        input_dict["dtype"] = input_dtype
        input_dict["range"] = input_range
        return input_dict

    @staticmethod
    def zeros_condition(input_dtype):
        x_dict = {}
        x_dict["shape"] = [0, ]
        x_dict["dtype"] = input_dtype
        x_dict["range"] = [[0, 0]]

        return [x_dict, [0], [0]]

    @staticmethod
    def _cal_size_by_end(begin_list, end_list):
        tmp_size_list = []
        for _begin, _end in zip(begin_list, end_list):
            tmp_size_list.append(_end - _begin)
        return tmp_size_list

    @staticmethod
    def _cal_end_by_size(begin, size):
        tmp_end_list = []
        for _begin, _size in zip(begin, size):
            tmp_end_list.append(_begin + _size)
        return tmp_end_list

    def check_input(self, ins):
        if len(ins) != INPUT_LEN:
            dict_args = {}
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "the slice input must be 3"
            raise RuntimeError(dict_args, get_error_message(dict_args))

        self.x_shape = list(ins[INPUT_X_IDX].get("shape"))
        self.x_dtype = ins[INPUT_X_IDX].get("dtype")
        self.x_range = ins[INPUT_X_IDX].get("range")

    def classify(self):
        if self.is_static or self.is_const:
            return self.static_classify()
        return self.dynamic_classify()

    def static_classify(self):

        if self.mode == MODE_END:
            self.size_list = self._cal_size_by_end(self.begin_list, self.end_list)

        # fix static value
        self._fix_static_value()

        if 0 in chain(self.size_list + self.x_shape):
            zero_ins = self.zeros_condition(self.x_dtype)
            operation.get_context().add("_const_begins", zero_ins[1])
            return [zero_ins]

        group_shape, group_begin, group_size, group_len = [], [], [], 0
        # fuse all normal axis
        for _idx, _shape in enumerate(self.x_shape):
            _begin, _size = self.begin_list[_idx], self.size_list[_idx]
            if _size - _begin == _shape:
                if _idx == 0:
                    group_shape.append(_shape)
                    group_begin.append(_begin)
                    group_size.append(_size)
                    group_len += 1
                else:
                    group_shape[group_len - 1] *= _shape
                    group_begin[group_len - 1] *= _shape
                    group_size[group_len - 1] *= _shape
            else:
                group_shape.append(_shape)
                group_begin.append(_begin)
                group_size.append(_size)
                group_len += 1

        fused_shape, fused_begin, fused_size, fused_idx = [], [], [], 0
        # fuse all slice 1 axis
        for i, _ in enumerate(group_shape):
            if i == 0:
                fused_shape.append(group_shape[i])
                fused_begin.append(group_begin[i])
                fused_size.append(group_size[i])
            else:
                if fused_size[fused_idx] == 1:
                    fused_shape[fused_idx] = group_shape[i] * fused_shape[fused_idx]
                    fused_begin[fused_idx] = group_shape[i] * fused_begin[fused_idx] + group_begin[i]
                    fused_size[fused_idx] = group_size[i]
                else:
                    fused_shape.append(group_shape[i])
                    fused_begin.append(group_begin[i])
                    fused_size.append(group_size[i])
                    fused_idx += 1

        # x dict
        fused_range = [(_range, _range) for _range in fused_shape]
        fused_dict = self.construct_static_input_dict(fused_shape, self.x_dtype, fused_range)

        # fuse sn condition
        fused_end = self._cal_end_by_size(fused_begin, fused_size)

        operation.get_context().add("_const_begins", fused_begin)
        operation.get_context().add("_const_ends", fused_end)

        return [[fused_dict, fused_begin, fused_end]]

    def dynamic_classify(self):
        ins = []
        x_shape_len = len(self.x_shape) if -2 not in self.x_shape else MAX_DIM_LEN
        for dim_len in range(1, x_shape_len + 1):
            # x
            x_dict = {}
            x_dict["shape"] = [-1] * dim_len
            x_dict["dtype"] = self.x_dtype
            x_dict["range"] = [[1, None]] * dim_len

            # begin
            begin_dict = {}
            begin_dict["shape"] = [dim_len]
            begin_dict["dtype"] = self.begin_dtype
            begin_dict["range"] = [[dim_len, dim_len]]

            # end
            end_dict = {}
            end_dict["shape"] = [dim_len]
            end_dict["dtype"] = self.end_dtype
            end_dict["range"] = [[dim_len, dim_len]]

            ins.append([x_dict, begin_dict, end_dict])

        ins.append(self.zeros_condition(self.x_dtype))
        if x_shape_len >= 2:
            ins.append(self.lr_depad_condition())
        return ins

    def lr_depad_condition(self):
        x_dict = {}
        x_dict["shape"] = [-1, -1]
        x_dict["dtype"] = self.x_dtype
        x_dict["range"] = [[1, None], [1, None]]

        # begin
        begin_dict = {}
        begin_dict["shape"] = [2]
        begin_dict["dtype"] = self.begin_dtype
        begin_dict["range"] = [[2, 2]]
        begin_dict["lr_depad"] = True

        # end
        end_dict = {}
        end_dict["shape"] = [2]
        end_dict["dtype"] = self.end_dtype
        end_dict["range"] = [[2, 2]]

        return [x_dict, begin_dict, end_dict]

    def _fix_static_value(self):
        for idx, val in enumerate(self.begin_list):
            if val < 0:
                self.begin_list[idx] = self.begin_list[idx] + self.x_shape[idx]

        for idx, val in enumerate(self.end_list):
            if val < 0:
                self.end_list[idx] = self.end_list[idx] + self.x_shape[idx]

        for idx, val in enumerate(self.size_list):
            if val == -1:
                self.size_list[idx] = self.x_shape[idx] - self.begin_list[idx]
