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
classifier of shape in pure elewise
"""
from typing import Any
from typing import Dict
from typing import Optional

from functools import reduce as shape_product
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

from . import util

COMMON = "common"
SPECIAL = "special"
CONST = "const"
EMPTY = "empty"
NOT_ALL_FUSE = "not_all_fuse"
UNKNOWN_RANK = -2
FIVE_HD_SHAPE_LEN = 5
FIVE_HD_ORI_SHAPE_LEN = 4
NHWC_PAD_C_AXIS = 3
NCHW_PAD_C_AXIS = 1
DIFF_SHAPE_NUM = 2
INT32_MAX = 2147483647
NC1HWC0_C1_INDEX = 1
NC1HWC0_C0_INDEX = 4
CONST_5HD_FORMATS = ['N', 'C1', 'H', 'W', 'C0']

C0_MAPPING = {
    "int64": 4,
    "uint64": 4,
    "float32": 16,
    "uint32": 16,
    "int32": 16,
    "float16": 16,
    "uint16": 16,
    "int16": 16,
    "int8": 32,
    "uint8": 32,
    "bool": 32,
    "uint1": 256,
}


def element_multiply(inputs):
    """
    get product res
    """
    if -1 in inputs:
        return -1
    if None in inputs:
        return None
    return shape_product(lambda x, y: x * y, inputs)


def parse_pad_axes(inputs):
    """
    get origin C axis index
    """
    if len(inputs) < 1:
        dict_args = {"errCode": "E90001",
                     "detailed_cause": "input tensor num must be more than 0."}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    ori_formats = set(x.get("ori_format") for x in inputs)
    if len(ori_formats) > 1 or len(ori_formats) < 0 or ori_formats == {None}:
        dict_args = {"errCode": "E90001",
                     "detailed_cause": "classify inputs must be all same ori_format."}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    pad_axes = {}
    if ori_formats == {'NHWC'}:
        pad_axes['C'] = NHWC_PAD_C_AXIS
    elif ori_formats == {'NCHW'}:
        pad_axes['C'] = NCHW_PAD_C_AXIS
    else:
        dict_args = {"errCode": "E90001",
                     "detailed_cause": "5HD ori format not support other format expect NCHW and NHWC."}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    operation.add_compile_info_inner("_elewise_pad_axis", pad_axes.get('C'))
    return pad_axes


def parse_np_mapping(in_format):
    """
    get mapping relation
    """
    if in_format != 'NC1HWC0':
        dict_args = {"errCode": "E90001",
                     "detailed_cause": "np mapping not support other format except 5HD."}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return {'C1': 'C', 'C0': 'C'}


class PureElewiseClassifier:
    """
    Pure elewise classifier
    """

    def __init__(self, ins: list, extra_params: Optional[Dict[str, Any]]):
        """
        init
        :param ins:
        """
        # original inputs info record
        self.ins = ins
        extra_params = {} if extra_params is None else extra_params
        # inputs may contain -2, so use max len as the dim_length
        self.dim_length = max(len(_ins.get('shape')) for _ins in self.ins)
        # deal with -2 shape if it exists
        self.is_unknown_rank = self._check_update_unknown_rank()
        self.shapes = [list(x.get('shape')) for x in self.ins]

        # judge if fractal_format sensitive and record corresponding infos
        self.ignore_fractal_format = extra_params.get('ignore_fractal_format', True)
        if not self.ignore_fractal_format:
            self.ori_shapes = [list(x.get('ori_shape', [])) for x in self.ins]
            self.formats = [x.get('format', 'ND') for x in self.ins]
            self.disable_fuse_axes = self._get_disable_fuse_axes()

        self.is_5hd_sensitive = self._check_5hd_sensitive()

        # check and update elewise inputs to simplify elewise classifier
        self._check_elewise_inputs_valid()
        self._update_elewise_inputs()

        operation.get_context().add("_unknown_rank", self.is_unknown_rank)
        operation.get_context().add("_ignore_fractal_format", self.ignore_fractal_format)

    def classify(self):
        """
        classify
        :return: classify res
        """
        return self._classify_const() if self._is_const() else self._classify_var()

    def _check_update_unknown_rank(self):
        """
        check if inputs contains unknown rank shapes
        :return: classify res
        """
        is_unknown_rank = False
        for _in in self.ins:
            shapes = list(_in.get('shape'))
            if UNKNOWN_RANK in shapes:
                if len(shapes) != 1:
                    dict_args = {"errCode": "E90001",
                                 "detailed_cause": "if the shape contains -2, it must be [-2] or (-2,)."}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                _in["shape"] = [-1] * self.dim_length
                _in["range"] = [(1, None)] * self.dim_length
                is_unknown_rank = True
        return is_unknown_rank

    def _get_disable_fuse_axes(self):
        """
        get disable fuse axes
        :return: disable fuse axes
        """
        if not self.ignore_fractal_format:
            if 'NC1HWC0' in self.formats:
                return [NC1HWC0_C1_INDEX, NC1HWC0_C0_INDEX]
        return []

    def _check_5hd_sensitive(self):
        """
        check if inputs format sensitive
        :return:
        """
        # binary inputs not support 5HD
        if self.is_unknown_rank and not self.ignore_fractal_format:
            dict_args = {"errCode": "E90001",
                         "detailed_cause": "5HD not support binary scene."}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if self.ignore_fractal_format:
            return False

        for index, _in in enumerate(self.ins):
            in_shape = self.shapes[index]
            in_ori_shape = self.ori_shapes[index]
            in_dtype = _in.get('dtype').lower()
            in_format = self.formats[index]
            ori_format = _in.get('ori_format').upper()

            # shape len must be 5, ori_shape len must be 4
            if len(in_shape) != FIVE_HD_SHAPE_LEN or len(in_ori_shape) != FIVE_HD_ORI_SHAPE_LEN:
                return False

            # format must be 5HD, ori_format must be NCHW or NHWC
            if in_format != 'NC1HWC0' or ori_format not in ('NCHW', 'NHWC'):
                return False

            # last axis must be align shape
            if in_shape[-1] != C0_MAPPING.get(in_dtype):
                return False

        return True

    def _check_elewise_inputs_valid(self):
        """
        check if inputs satisify elewise requirement
        """
        # check shape valid
        for dim_index in range(self.dim_length):
            diff_shape = set([each_shape[dim_index] for each_shape in self.shapes])
            if len(diff_shape) == DIFF_SHAPE_NUM:
                if -1 not in diff_shape:
                    dict_args = {"errCode": "E90001",
                                 "detailed_cause": "elewise not support two diff shape without -1."}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
            if len(diff_shape) > DIFF_SHAPE_NUM:
                dict_args = {"errCode": "E90001",
                             "detailed_cause": "elewise not support different known shape at one dim."}
                raise RuntimeError(dict_args, get_error_message(dict_args))

        # check ori_shape valid
        if self.is_5hd_sensitive:
            for ori_index in range(len(self.ori_shapes[0])):
                # update ori_shape
                diff_ori_shape = set([each_ori_shape[ori_index] for each_ori_shape in self.ori_shapes])
                if len(diff_ori_shape) == DIFF_SHAPE_NUM:
                    if -1 not in diff_ori_shape:
                        dict_args = {"errCode": "E90001",
                                     "detailed_cause": "elewise 5hd not support two diff ori shape without -1."}
                        raise RuntimeError(dict_args, get_error_message(dict_args))
                if len(diff_ori_shape) > DIFF_SHAPE_NUM:
                    dict_args = {"errCode": "E90001",
                                 "detailed_cause": "elewise 5hd not support different known ori shape at one dim."}
                    raise RuntimeError(dict_args, get_error_message(dict_args))

    def _update_elewise_shape_and_range(self):
        """
        update elewise shape and range to simplify elewise classify
        """
        # elewise classify only support same dim_len
        for dim_index in range(self.dim_length):
            diff_shape = set([each_shape[dim_index] for each_shape in self.shapes])
            left_range = set([_in.get('range')[dim_index][0] for _in in self.ins])
            right_range = set([_in.get('range')[dim_index][1] for _in in self.ins])
            if None in right_range:
                right_range.remove(None)
                right_range.add(INT32_MAX)
            # all dim must be intersecting
            max_left_range = max(left_range)
            min_right_range = min(right_range)

            # update shape and range
            if len(diff_shape) == 1:
                # all shape at the index is same, update the range to be (max_left_range, min_right_range)
                for input_index in range(len(self.shapes)):
                    self.shapes[input_index][dim_index] = \
                        max_left_range if max_left_range == min_right_range else self.shapes[input_index][dim_index]
                    self.ins[input_index]['range'][dim_index] = \
                        (max_left_range, min_right_range) if min_right_range < INT32_MAX else (max_left_range, None)

            if len(diff_shape) == DIFF_SHAPE_NUM:
                # the shape at index not support shape_combine without -1
                diff_shape.remove(-1)
                known_shape = diff_shape.pop()
                for input_index in range(len(self.shapes)):
                    self.ins[input_index]['range'][dim_index] = (known_shape, known_shape)
                    self.shapes[input_index][dim_index] = known_shape

    def _update_elewise_ori_shape(self):
        """
        update ori shape while fractal format sensitive
        """
        for ori_index in range(len(self.ori_shapes[0])):
            # update ori_shape
            diff_ori_shape = set([each_ori_shape[ori_index] for each_ori_shape in self.ori_shapes])
            if len(diff_ori_shape) == DIFF_SHAPE_NUM:
                # the ori shape at index not support shape_combine without -1
                diff_ori_shape.remove(-1)
                known_ori_shape = diff_ori_shape.pop()
                for input_index in range(len(self.ori_shapes)):
                    self.ori_shapes[input_index][ori_index] = known_ori_shape

    def _update_elewise_inputs(self):
        """
        update inputs to simplify elewise classify
        """
        # static scene no need update any info
        if operation.get_context().get_mode() == "static":
            return
        # convert tuple range to list for replace some elements
        for _in in self.ins:
            _in['range'] = list(_in.get('range'))

        # update shape and range
        self._update_elewise_shape_and_range()

        # update ori_shape if satisfy 5hd scene
        if self.is_5hd_sensitive:
            self._update_elewise_ori_shape()

    def _is_const(self):
        for i in range(self.dim_length):
            if min(s[i] for s in self.shapes) == -1:
                return False
        return True

    def _get_elewise_fused_index(self):
        able_fuse_axis = list(range(self.dim_length))
        axis = list(filter(lambda x: x not in self.disable_fuse_axes, able_fuse_axis))
        left = axis[0]
        right = axis[0]

        fused_index = []
        for _index in axis[1:]:
            if _index == right + 1:
                right = _index
            else:
                fused_index.append([left] if left == right else [left, right])
                left = _index
                right = _index
        fused_index.append([left] if left == right else [left, right])
        for i in self.disable_fuse_axes:
            fused_index.append([i])
            fused_index.sort(key=lambda x: x[0])
        return fused_index

    def _partial_fuse_shapes(self):
        fusion_axis = self._get_elewise_fused_index()
        in_shape = self.shapes[0]
        fused_shape = []
        fused_range = []
        for axis_list in fusion_axis:
            if len(axis_list) == 1:
                fused_shape.append(in_shape[axis_list[0]])
            else:
                need_fused_shape = in_shape[axis_list[0]:axis_list[1] + 1]
                product_shape = element_multiply(need_fused_shape)
                fused_shape.append(product_shape)

        if operation.get_context().get_mode() == "static":
            fused_range = [(known_shape, known_shape) for known_shape in fused_shape]
        else:
            in_range = self.ins[0].get('range')
            for axis_list in fusion_axis:
                if len(axis_list) == 1:
                    fused_range.append(tuple(in_range[axis_list[0]]))
                else:
                    range_l = [in_range[i][0] for i in range(axis_list[0], axis_list[1] + 1)]
                    fuse_range_left = element_multiply(range_l)
                    fuse_range_left = \
                        None if fuse_range_left is not None and fuse_range_left >= INT32_MAX else fuse_range_left

                    range_r = [in_range[i][1] for i in range(axis_list[0], axis_list[1] + 1)]
                    fuse_range_right = element_multiply(range_r)
                    fuse_range_right = \
                        None if fuse_range_right is not None and fuse_range_right >= INT32_MAX else fuse_range_right

                    fused_range.append((fuse_range_left, fuse_range_right))

        fused_shapes = [fused_shape] * len(self.shapes)
        fused_ranges = [fused_range] * len(self.shapes)
        operation.add_compile_info_inner("_elewise_fused_index", fusion_axis)
        return fused_shapes, fused_ranges, fusion_axis

    def _is_const_not_fuse(self):
        if self.ignore_fractal_format:
            return False

        all_ori_c_align = True
        if self.is_5hd_sensitive:
            # check if ori_c is all align
            ori_c_map = parse_pad_axes(self.ins)
            ori_c_index = ori_c_map.get('C')
            for _, _in in enumerate(self.ins):
                c_values = _in.get('ori_shape')[ori_c_index]
                c0_align = C0_MAPPING.get(_in.get('dtype'))
                if c_values % c0_align == 0:
                    continue
                all_ori_c_align = False
                break
        return not all_ori_c_align

    def _classify_const(self):
        const_shape = self.shapes[0]
        # only format sensitive and ori_c exist not aligned generate not all fuse res
        if self._is_const_not_fuse():
            fused_shapes, _, fused_index = self._partial_fuse_shapes()
            s_format = []
            for f_index in fused_index:
                s_format.append([CONST_5HD_FORMATS[index] for index in range(f_index[0], f_index[-1] + 1)])
            inputs = [ConstMode.gen_in(fused_shapes[i]) for i, _ in enumerate(self.ins)]
            for i, input_x in enumerate(inputs):
                input_x['const_shape'] = const_shape
                input_x['ori_shape'] = self.ori_shapes[i]
                input_x['format'] = self.formats[i]
                input_x['s_format'] = s_format
                input_x['pad_axes'] = parse_pad_axes(self.ins)
                input_x['np_mapping'] = parse_np_mapping(self.formats[i])
                input_x["mode_5hd"] = True
            return [inputs]

        shape = [element_multiply(const_shape)]
        inputs = [ConstMode.gen_in(shape) for _ in self.ins]
        for input_x in inputs:
            input_x["const_shape"] = const_shape

        return [inputs]

    def _check_dynamic_not_fuse(self):
        # get maybe not fuse and only not fuse
        if self.ignore_fractal_format:
            return False, False

        if self.is_5hd_sensitive:
            ori_c_map = parse_pad_axes(self.ins)
            ori_c_index = ori_c_map.get('C')
            ori_c_shape = self.ori_shapes[0][ori_c_index]
            max_c0_value = max(C0_MAPPING.get(_in.get('dtype')) for _in in self.ins)
            if ori_c_shape < 0:
                return True, False
            if ori_c_shape > 0 and ori_c_shape % max_c0_value != 0:
                return True, True
        return False, False

    def _classify_var(self):
        ret = []
        maybe_not_fuse, only_not_fuse = self._check_dynamic_not_fuse()
        # maybe not fuse will generate not all fuse res and all fuse res
        # only not fuse will only generate not all fuse res
        if maybe_not_fuse:
            fused_shapes, fused_ranges, fused_index = self._partial_fuse_shapes()
            s_format = []
            for f_index in fused_index:
                s_format.append([CONST_5HD_FORMATS[index] for index in range(f_index[0], f_index[-1] + 1)])
            inputs = \
                [NotAllFuseMode.gen_in(fused_shapes[i], fused_ranges[i]) for i, _ in enumerate(self.ins)]
            for i, input_x in enumerate(inputs):
                input_x['ori_shape'] = self.ori_shapes[i]
                input_x['format'] = self.formats[i]
                input_x['s_format'] = s_format
                input_x['pad_axes'] = parse_pad_axes(self.ins)
                input_x['np_mapping'] = parse_np_mapping(self.formats[i])
                input_x["mode_5hd"] = True

            ret.append(inputs)
            if only_not_fuse:
                return ret

        maybe_empty_tensor = False
        must_empty_tensor = False
        ins = []
        # format insensitive or format sensitive but ori_c all align will only generate ND res
        for index, _shape in enumerate(self.shapes):
            in_x = SpecialMode.gen_in([-1])
            before_fuse_range = self.ins[index].get('range')
            in_x["range"] = [util.combine_range(before_fuse_range)]
            maybe_empty_tensor = maybe_empty_tensor or 0 in in_x.get('range')[0]
            if 0 in _shape or (0, 0) in before_fuse_range or [0, 0] in before_fuse_range:
                must_empty_tensor = True
                break
            ins.append(in_x)

        if not must_empty_tensor:
            ret.append(ins)
        if maybe_empty_tensor or self.is_unknown_rank:
            input_length = len(self.ins)
            ins = [EmptyMode.gen_in()] * input_length
            ret.append(ins)

        return ret


class EmptyMode:
    """
    Empty Mode
    """

    @classmethod
    def gen_in(cls):
        """
        generate input
        :return:
        """
        return {"shape": (0,),
                "range": [(0, 0)],
                "support_broadcast": True,
                "mode": EMPTY,
                }


class ConstMode:
    """
    ConstMode
    """

    @classmethod
    def gen_in(cls, shape):
        """
        gen_in
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "mode": CONST,
                "support_broadcast": False,
                }


class SpecialMode:
    """
    SpecialMode
    """

    @classmethod
    def gen_in(cls, shape):
        """
        gen_in
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "mode": SPECIAL,
                "support_broadcast": False,
                "pattern": (COMMON,)
                }


class NotAllFuseMode:
    """
    NotAllFuseMode
    """

    @classmethod
    def gen_in(cls, shape, shape_range):
        """
        gen_in
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": shape_range,
                "mode": SPECIAL,
                "support_broadcast": False,
                "pattern": ('not_all_fuse',)
                }
