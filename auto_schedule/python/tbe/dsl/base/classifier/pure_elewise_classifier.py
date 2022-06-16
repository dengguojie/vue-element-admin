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
        # record origin inputs info
        self.ins = ins
        extra_params = {} if extra_params is None else extra_params
        self.ignore_fractal_format = extra_params.get('ignore_fractal_format', True)
        self.disable_fuse_axes = self._get_disable_fuse_axes()
        self.dim_length = max(len(_ins.get('shape')) for _ins in self.ins)
        # record processed inputs info
        self.is_unknown_rank = self._check_update_unknown_rank()
        self.shapes = [list(x.get('shape')) for x in self.ins]
        self.ori_shapes = [list(x.get('ori_shape', [])) for x in self.ins]
        self.ranges = [list(
            list(dim_range) for dim_range in x.get('range', [[i, i] if i != -1 else [1, None] for i in x.get('shape')]))
                       for x in self.ins]
        self.formats = [x.get('format', 'ND') for x in self.ins]
        self.is_5hd_sensitive = self._check_5hd_sensitive()
        operation.get_context().add("_ignore_fractal_format", self.ignore_fractal_format)
        operation.get_context().add("_unknown_rank", self.is_unknown_rank)

    def check_and_update_inputs(self):
        """
        check inputs valid and update inputs info
        :return: updated inputs
        """
        # elewise classify only support same dim_len
        for dim_index in range(self.dim_length):
            shape_at_index = \
                set(self.shapes[input_index][dim_index] for input_index in range(len(self.shapes)))
            range_left_at_index = \
                set(self.ranges[input_index][dim_index][0] for input_index in range(len(self.ranges)))
            range_right_at_index = \
                set(self.ranges[input_index][dim_index][1] for input_index in range(len(self.ranges)))
            if None in range_right_at_index:
                range_right_at_index.remove(None)
                range_right_at_index.add(INT32_MAX)
            # all dim range must be intersecting
            max_range_left = max(range_left_at_index)
            min_range_right = min(range_right_at_index)

            # update shape and range
            if len(shape_at_index) == 1:
                for input_index in range(len(self.shapes)):
                    self.ranges[input_index][dim_index][0] = max_range_left
                    self.ranges[input_index][dim_index][1] = min_range_right
                    # if shape is -1 and range_left == range_right, update shape to known value
                    if self.shapes[input_index][dim_index] == -1 and max_range_left == min_range_right:
                        self.shapes[input_index][dim_index] = max_range_left
            elif len(shape_at_index) == DIFF_SHAPE_NUM:
                if -1 not in shape_at_index:
                    dict_args = {"errCode": "E90001",
                                 "detailed_cause": "elewise shape is illegal."}
                    raise RuntimeError(dict_args, get_error_message(dict_args))

                shape_at_index.remove(-1)
                known_shape = shape_at_index.pop()
                for input_index in range(len(self.shapes)):
                    self.shapes[input_index][dim_index] = known_shape
                    self.ranges[input_index][dim_index][0] = known_shape
                    self.ranges[input_index][dim_index][1] = known_shape
            else:
                dict_args = {"errCode": "E90001",
                             "detailed_cause": "elewise shape is illegal because of different shape."}
                raise RuntimeError(dict_args, get_error_message(dict_args))

        for dim_index in range(len(self.ori_shapes[0])):
            # update ori_shape
            ori_shape_at_index = \
                set(self.ori_shapes[input_index][dim_index] for input_index in range(len(self.ori_shapes)))
            if len(ori_shape_at_index) == DIFF_SHAPE_NUM:
                if -1 not in ori_shape_at_index:
                    dict_args = {"errCode": "E90001",
                                 "detailed_cause": "elewise ori_shape is illegal."}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                ori_shape_at_index.remove(-1)
                known_ori_shape = ori_shape_at_index.pop()
                for input_index in range(len(self.ori_shapes)):
                    self.ori_shapes[input_index][dim_index] = known_ori_shape

    def classify(self):
        """
        classify
        :return: classify res
        """
        return self._classify_const() if self._is_const() else self._classify_var()

    def _get_disable_fuse_axes(self):
        """
        get disable fuse axes
        :return: disable fuse axes
        """
        if not self.ignore_fractal_format:
            if 'NC1HWC0' in [x.get('format', 'ND') for x in self.ins]:
                return [NC1HWC0_C1_INDEX, NC1HWC0_C0_INDEX]
        return []

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

        for _in in self.ins:
            in_shape = _in.get('shape')
            in_ori_shape = _in.get('ori_shape')
            in_dtype = _in.get('dtype').lower()
            in_format = _in.get('format').upper()
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
        in_range = self.ranges[0]

        fused_shape = []
        fused_range = []
        for axis_list in fusion_axis:
            if len(axis_list) == 1:
                fused_shape.append(in_shape[axis_list[0]])
                fused_range.append(tuple(in_range[axis_list[0]]))
            else:
                need_fused_shape = in_shape[axis_list[0]:axis_list[1] + 1]
                product_shape = element_multiply(need_fused_shape)

                range_l = [in_range[i][0] for i in range(axis_list[0], axis_list[1] + 1)]
                fuse_range_left = element_multiply(range_l)
                fuse_range_left = fuse_range_left if fuse_range_left <= INT32_MAX else None

                range_r = [in_range[i][1] for i in range(axis_list[0], axis_list[1] + 1)]
                fuse_range_right = element_multiply(range_r)
                fuse_range_right = fuse_range_right if fuse_range_right <= INT32_MAX else None

                fused_shape.append(product_shape)
                fused_range.append((fuse_range_left, fuse_range_right))

        fused_shapes = [fused_shape] * len(self.shapes)
        fused_ranges = [fused_range] * len(self.ranges)
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
        for _, (_shape, _range) in enumerate(zip(self.shapes, self.ranges)):
            in_x = SpecialMode.gen_in([-1])
            in_x["range"] = [util.combine_range(_range)]
            maybe_empty_tensor = maybe_empty_tensor or 0 in in_x.get('range')[0]
            if 0 in _shape or (0, 0) in _range or [0, 0] in _range:
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
