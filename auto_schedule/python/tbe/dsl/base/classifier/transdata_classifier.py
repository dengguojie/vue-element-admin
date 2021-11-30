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
import copy
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

GENERAL_FORWARD = "general.forward"
GENERAL_BACKWARD = "general.backward"


class CombineInfo:
    def __init__(self, dst_idx, src_idx, dim):
        self.dst_idx = dst_idx
        self.src_idx = src_idx
        self.dim = dim


class TransdataClassify:
    def __init__(self, ins):
        self.ins = copy.deepcopy(ins)
        self.src_shape = None
        self.dst_shape = None
        self.axes_map = None

        self.src_list = []
        self.dst_list = []
        self.pad_list = []
        self.is_forward = True

        self.combine_info = []
        self.pad_mapping = None
        self.shape_before_transpose = None

        self._collect()
        self._swaps()
        self._eliminate_one()
        self._support_check()

    def _collect(self):
        self.src_shape = self.ins[0].get("shape")
        self.dst_shape = self.ins[1]
        self.axes_map = self.ins[2]
        self._convert_map_to_list()

    def _convert_map_to_list(self):
        for key, value in self.axes_map.items():
            if isinstance(value, (tuple, list)) and isinstance(key, int):
                self.src_list.extend([key] * len(value))
                self.dst_list.extend(list(value))
                self.pad_list.append(key)
            elif isinstance(value, int) and isinstance(key, (tuple, list)):
                self.src_list.extend(list(key))
                self.dst_list.extend([value] * len(key))
                self.pad_list.append(value)
            elif isinstance(value, int) and isinstance(key, int):
                self.src_list.append(key)
                self.dst_list.append(value)

        def normalize(input):
            back = list(set(input))
            length = len(back)
            result = []
            for i in input:
                result.append(i if i >= 0 else length + i)
            return result

        def do_duplicate(input):
            base = []
            result = []
            for i in input:
                if i not in base:
                    result.append(i)
                    base.append(i)
                else:
                    while i in base:
                        i += 0.1
                    result.append(i)
                    base.append(i)
            return result

        self.src_list = normalize(self.src_list)
        self.dst_list = normalize(self.dst_list)

        self.src_list = do_duplicate(self.src_list)
        self.dst_list = do_duplicate(self.dst_list)

    def _swaps(self):
        def do_order(x, y):
            back = []
            for x0, y0 in zip(x, y):
                back.append([x0, y0])
            back.sort(key=lambda x: x[0])
            x, y = [], []
            for value in back:
                x.append(value[0])
                y.append(value[1])
            return x, y

        length_src = len(set([int(x) for x in self.src_list]))
        length_dst = len(set([int(x) for x in self.dst_list]))
        if length_src > length_dst:
            self.is_forward = False
            self.src_list, self.dst_list = self.dst_list, self.src_list
            self.src_shape, self.dst_shape = self.dst_shape, self.src_shape
            self.src_list, self.dst_list = do_order(self.src_list, self.dst_list)

    def _support_check(self):
        for key, value in self.axes_map.items():
            length_key = len(key) if isinstance(key, (list, tuple)) else 1
            length_value = len(value) if isinstance(value, (list, tuple)) else 1

            if self.is_forward and isinstance(key, (list, tuple)):
                dict_args = {"errCode": "E90001", "detailed_cause":
                    "Op[Pad] and Op[DePad] can not existed together, "
                    f"current Op is Pad, but key of axes_map is {key}"}
                raise RuntimeError(dict_args, get_error_message(dict_args))

            if not self.is_forward and isinstance(value, (list, tuple)):
                dict_args = {"errCode": "E90001", "detailed_cause":
                    "Op[Pad] and Op[DePad] can not existed together, "
                    f"current Op is DePad, but value of axes_map is {value}"}
                raise RuntimeError(dict_args, get_error_message(dict_args))

            if length_key > 2:
                dict_args = {"errCode": "E90001", "detailed_cause":
                    "Only support fused [DimX0, DimX1] to DimX,"
                    f"but key of axes_map is {key}"}
                raise RuntimeError(dict_args, get_error_message(dict_args))

            if length_value > 2:
                dict_args = {"errCode": "E90001", "detailed_cause":
                    "Only support split DimX to [DimX0, DimX1],"
                    f"but value of axes_map is {value}"}
                raise RuntimeError(dict_args, get_error_message(dict_args))

    def classify(self):
        """
        Regulation: dst_shape <dst_list> shape_before_transpose <src_list> src_shape
        DST_SHAPE    DST_LIST    SHAPE_BEFORE_TRANSPOSE    SRC_LIST    PAD_LIST
           N            0                N                     0
           C1           2                H                     1
           H            3                W                     2
           W            1                C1                    3           1
           C0           4                C0                    3.1         4
        """
        # combine info to fused
        self.shape_before_transpose = [self.dst_shape[i] for i in self.dst_list]
        info = zip(self.dst_list, self.src_list, self.shape_before_transpose)
        for dst_idx, src_idx, dim in info:
            self.combine_info.append(CombineInfo(dst_idx, src_idx, dim))
        result = self._fuse_combine_info(self.combine_info)

        # update base info
        dst_list, src_list = [], []
        shape_before_transpose = []
        for item in result:
            dst_list.append(item.dst_idx)
            src_list.append(item.src_idx)
            shape_before_transpose.append(item.dim)

        # -------------------------
        # update dst_list/dst_shape
        # -------------------------
        n_dst_list, n_dst_shape = [], [0] * len(dst_list)
        help_dst_list = dst_list.copy()
        help_dst_list.sort()

        for value in dst_list:
            n_dst_list.append(help_dst_list.index(value))
        for index, value in zip(n_dst_list, shape_before_transpose):
            n_dst_shape[index] = value

        # -------------------------
        # update src_list/src_shape
        # -------------------------
        help_src_list = list(set(int(x) for x in src_list))
        n_src_list = list(range(len(help_src_list)))
        n_src_shape, n_pad_list = [], []

        for value in help_src_list:
            if value in self.pad_list:
                n_src_shape.append(self.src_shape[value])
                n_pad_list.append(len(n_src_shape) - 1)
            else:
                n_src_shape.append(shape_before_transpose[src_list.index(value)])

        # ------------------
        # update pad_mapping
        # ------------------
        pad_mapping = {v: [] for v in self.pad_list}

        for src_idx, dst_idx in zip(src_list, dst_list):
            if int(src_idx) in self.pad_list:
                pad_mapping[int(src_idx)].append(dst_idx)

        new_pad_mapping = {}
        for ori_key, ori_value in pad_mapping.items():
            update_value = []
            for i in ori_value:
                update_value.append(help_dst_list.index(i))
            update_key = help_src_list.index(ori_key)
            new_pad_mapping[update_key] = update_value

        # -----------------------------
        # result(if backward, reversed)
        # -----------------------------
        self.src_list = n_src_list
        self.dst_list = n_dst_list
        self.src_shape = n_src_shape
        self.dst_shape = n_dst_shape
        self.pad_mapping = new_pad_mapping
        self._update_axes_map()
        self._add_compile_info(src_list)
        self._reversed()
        self.ins[0]["shape"] = self.src_shape
        self.ins[0]["range"] = [[1, None] if x == -1 else [x, x] for x in self.src_shape]
        self.ins[0]["is_forward"] = self.is_forward

        if self.is_forward:
            self.ins[0]["transdata_category"] = GENERAL_FORWARD
        else:
            self.ins[0]["transdata_category"] = GENERAL_BACKWARD

        self.ins[1] = self.dst_shape
        self.ins[2] = self.axes_map
        result = [self.ins]

        return result

    def _add_compile_info(self, ori_src_list):
        """
        ori_src_list: axis which not fused
        _src_pad has three model: 0 is not pad, 1 is do pad, 2 is do split and pad
        """
        _src_pad = []
        for key, value in self.axes_map.items():
            if isinstance(value, int):
                _src_pad.append(0)
            elif isinstance(value, (list, tuple)) and len(value) == 1:
                _src_pad.append(1)
                pad_factor = self.dst_shape[value[0]]
                operation.add_compile_info_inner("_pad_factor", pad_factor)
            else:
                _src_pad.append(2)
                pad_factor = self.dst_shape[value[-1]]
                operation.add_compile_info_inner("_pad_factor", pad_factor)

        operation.add_compile_info_inner("_src_pad", _src_pad)
        operation.add_compile_info_inner("_permute", self.dst_list)
        operation.add_compile_info_inner("_src_fuse", list(set([int(x) for x in ori_src_list])))

    def _reversed(self):
        if self.is_forward:
            return

        def mapping_reversed(input):
            result = {}
            for key, value in input.items():
                value = tuple(value) if isinstance(value, list) else value
                result[value] = key
            return result

        self.src_shape, self.dst_shape = self.dst_shape, self.src_shape
        self.pad_mapping = mapping_reversed(self.pad_mapping)
        self.axes_map = mapping_reversed(self.axes_map)

    def _update_axes_map(self):
        current_idx = 0
        self.axes_map = {}
        self.pad_list = [x for x in self.pad_mapping.keys()]
        for key in self.src_list:
            if key in self.pad_list:
                target = self.pad_mapping.get(key)
                self.axes_map[key] = target
                current_idx += len(target)
            else:
                self.axes_map[key] = self.dst_list[current_idx]
                current_idx += 1

    def _fuse_combine_info(self, input):
        _result = []
        root_ptr = 0
        while root_ptr <= len(input) - 1:
            if root_ptr == len(input) - 1:
                _result.append(input[root_ptr])
                break

            slide_ptr = root_ptr + 1
            dim = input[root_ptr].dim
            while slide_ptr <= len(input) - 1:
                is_continuous = input[slide_ptr].dst_idx == input[slide_ptr - 1].dst_idx + 1
                right_axis_type = int(input[slide_ptr].src_idx) in self.pad_list
                left_axis_type = int(input[slide_ptr - 1].src_idx) in self.pad_list
                is_same_type = right_axis_type == left_axis_type

                if not is_continuous or not is_same_type:
                    break

                next_dim = input[slide_ptr].dim
                dim = -1 if dim == -1 or next_dim == -1 else dim * next_dim
                slide_ptr += 1

            input[root_ptr].dim = dim
            _result.append(input[root_ptr])
            root_ptr = slide_ptr
        return _result

    def _eliminate_one(self, ):
        # 1. Only work in Const (Don't do infer_shape)
        # 2. Pad axis can not do eliminate
        if not _is_const(self.src_shape) or not _is_const(self.dst_shape):
            self.src_shape = [-1] * len(self.src_shape)
            return
        src_shape = []
        src_list = []
        pad_list = []
        for k, v in enumerate(self.src_shape):
            if v == 1 and k not in self.pad_list:
                continue

            if k in self.pad_list:
                pad_list.append(len(src_shape))
            src_shape.append(v)

            for i in self.src_list:
                if int(i) == k:
                    src_list.append(i)

        dst_list = [self.dst_list[self.src_list.index(x)] for x in src_list]
        dst_shape = [self.dst_shape[x] for x in sorted(dst_list)]

        self.src_shape = src_shape
        self.dst_shape = dst_shape
        self.pad_list = pad_list
        self.dst_list = [sorted(dst_list).index(x) for x in dst_list]
        num = 0
        self.src_list = []
        for i in src_list:
            if not isinstance(i, int):
                self.src_list.append(num - 1 + i - int(i))
            else:
                self.src_list.append(num)
                num += 1


def classify(ins: list):
    """
    classify: fused axis, assure pad_axis
    """
    from tbe.common.buildcfg import get_current_build_config
    if get_current_build_config("enable_op_prebuild"):
        return [ins]
    return TransdataClassify(ins).classify()


def _is_const(shape):
    return -1 not in shape
