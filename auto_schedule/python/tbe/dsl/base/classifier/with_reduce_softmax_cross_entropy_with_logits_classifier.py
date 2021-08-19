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
classifier of shape in broadcast elewise
"""
import re
from enum import Enum, auto
from tbe.common.utils.errormgr import get_error_message

from . import util

ONE = "one"
COMMON = "common"
BROADCAST = "broadcast"
COMMON_REDUCE = "common_reduce"
BROADCAST_REDUCE = "broadcast_reduce"
SPECIAL = "special"
CONST = "const"
ORIGINAL = "original"
MAX_INT32_VALUE = 2147483647


class WithReduceSoftmaxCrossEntropyWithLogitsClassifier:
    """
    Elewise with broadcast classifier
    """

    def __init__(self, ins: list):
        """
        init
        :param ins:
        """
        if len(ins) != 2:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The size of input parameter [%s] must be 2, " \
                                          "when support broadcast." % len(ins)
            raise RuntimeError(dict_args, get_error_message(dict_args))

        self.ins = ins
        self.format = self.ins[0]["format"]
        shapes = [x["shape"] for x in self.ins]
        self.dim_length = max([len(s) for s in shapes])

        self.normalized_ins = self._normalize()
        self.normalized_shapes = [x["shape"] for x in self.normalized_ins]
        f_shape1, f_range1, f_shape2, f_range2, = _simplify_shape(*self.normalized_ins)
        self._reduce_axis_index = _get_reduce_axis_index(self.normalized_shapes, self.format, f_shape1, f_shape2)
        self.f_shapes = [f_shape1, f_shape2]
        self.f_ranges = [f_range1, f_range2]
        self.f_strict_pattern = _get_strict_pattern(f_shape1, f_shape2, self.format, self._reduce_axis_index)
        self.f_patterns = _combinations(self.f_strict_pattern)
        if self.format == "ND" and f_shape1[0] == f_shape2[0] == 1:
            self.f_patterns = _trans_one_to_common(self.f_patterns)

    def classify(self):
        """
        classify
        :return:
        """
        [[(r00_l, r00_r), (r01_l, r01_r)], [(r10_l, r10_r), (r11_l, r11_r)]] = self.f_ranges
        tail_not_broadcast = (r01_l > 1 and r11_l > 1) or (r01_l == r01_r == 1 and r11_l == r11_r == 1)
        tail_may_broadcast = not tail_not_broadcast
        batch_not_broadcast = (r00_l > 1 and r10_l > 1) or (r00_l == r00_r == 1 and r10_l == r10_r == 1)
        batch_may_broadcast = not batch_not_broadcast

        def gen_template(shape, range, mode, key):
            return {"shape": shape,
                    "range": range,
                    "mode": mode,
                    "key": key
                    }

        def is_legal_range(dim_range):
            dim_range_l, dim_range_r = dim_range
            return dim_range_l <= dim_range_r

        def _process_range_vs_2(dim_range):
            if dim_range[1] >= 2:
                return max(dim_range[0], 2), dim_range[1]
            else:
                return dim_range

        def _range_to_int(range_val):
            return MAX_INT32_VALUE if range_val is None else int(range_val)

        def _process_range(range0, range1):
            dim00_range = range0[0]
            dim01_range = range0[1]
            dim10_range = range1[0]
            dim11_range = range1[1]
            if _range_to_int(dim00_range[0]) > 1 and _range_to_int(dim10_range[0]) > 1:
                intersection_dim00_dim10_range = (max(_range_to_int(dim00_range[0]), _range_to_int(dim10_range[0])),
                                                  min(_range_to_int(dim00_range[1]), _range_to_int(dim10_range[1])))
                dim00_range = intersection_dim00_dim10_range
                dim10_range = intersection_dim00_dim10_range
            else:
                dim00_range = (_range_to_int(dim00_range[0]), _range_to_int(dim00_range[1]))
                dim10_range = (_range_to_int(dim10_range[0]), _range_to_int(dim10_range[1]))

            if _range_to_int(dim01_range[0]) > 1 and _range_to_int(dim11_range[0]) > 1:
                intersection_dim01_dim11_range = (max(_range_to_int(dim01_range[0]), _range_to_int(dim11_range[0])),
                                                  min(_range_to_int(dim01_range[1]), _range_to_int(dim11_range[1])))
                dim01_range = intersection_dim01_dim11_range
                dim11_range = intersection_dim01_dim11_range
            else:
                dim01_range = (_range_to_int(dim01_range[0]), _range_to_int(dim01_range[1]))
                dim11_range = (_range_to_int(dim11_range[0]), _range_to_int(dim11_range[1]))

            range0 = [dim00_range, dim01_range]
            range1 = [dim10_range, dim11_range]
            return range0, range1

        if tail_not_broadcast:
            res = []
            res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL, 0),
                        gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL, 0)])
            return res

        if tail_may_broadcast and batch_not_broadcast:
            res = []
            res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL, 0),
                        gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL, 0)])

            special_shape0 = [self.f_shapes[0][0], 1]
            special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
            special_shape1 = self.f_shapes[1]
            special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
            special_range0, special_range1 = _process_range(special_range0, special_range1)
            if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                    and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                res.append([gen_template(special_shape0, special_range0, "vec2", 2),
                            gen_template(special_shape1, special_range1, "vec2", 2)])

            special_shape0 = self.f_shapes[0]
            special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
            special_shape1 = [self.f_shapes[1][0], 1]
            special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
            special_range0, special_range1 = _process_range(special_range0, special_range1)
            if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                    and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                res.append([gen_template(special_shape0, special_range0, "vec8", 8),
                            gen_template(special_shape1, special_range1, "vec8", 8)])
            return res

        if tail_may_broadcast and batch_may_broadcast:
            res = []
            res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL, 0),
                        gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL, 0)])

            special_shape0 = [self.f_shapes[0][0], 1]
            special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
            special_shape1 = self.f_shapes[1]
            special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
            special_range0, special_range1 = _process_range(special_range0, special_range1)
            if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                    and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                res.append([gen_template(special_shape0, special_range0, "vec2", 2),
                            gen_template(special_shape1, special_range1, "vec2", 2)])

            special_shape0 = self.f_shapes[0]
            special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
            special_shape1 = [self.f_shapes[1][0], 1]
            special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
            special_range0, special_range1 = _process_range(special_range0, special_range1)
            if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                    and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                res.append([gen_template(special_shape0, special_range0, "vec8", 8),
                            gen_template(special_shape1, special_range1, "vec8", 8)])

            special_shape0 = [1, self.f_shapes[0][1]]
            special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
            special_shape1 = [self.f_shapes[1][0], 1]
            special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
            special_range0, special_range1 = _process_range(special_range0, special_range1)
            if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                    and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                res.append([gen_template(special_shape0, special_range0, "vec9", 9),
                            gen_template(special_shape1, special_range1, "vec9", 9)])

            special_shape0 = [self.f_shapes[0][0], 1]
            special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
            special_shape1 = [1, self.f_shapes[1][1]]
            special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
            special_range0, special_range1 = _process_range(special_range0, special_range1)
            if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                    and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                res.append([gen_template(special_shape0, special_range0, "vec6", 6),
                            gen_template(special_shape1, special_range1, "vec6", 6)])
            return res

    def _normalize(self):
        def clone_complete(_in):
            _shape, _range = list(_in["shape"]), _in.get("range")
            d_v = self.dim_length - len(_shape)

            in_x = _in.copy()
            in_x["shape"] = [1] * d_v + _shape
            in_x["range"] = util.generate_range(_shape) if _range is None else \
                [(1, 1)] * d_v + list(_range)
            return in_x

        return [clone_complete(x) for x in self.ins]

    def _is_const(self):
        for i in range(self.dim_length):
            dims_i = [s[i] for s in self.normalized_shapes]
            min_dim_v, max_dim_v = min(dims_i), max(dims_i)
            if min_dim_v == -1 and max_dim_v in (-1, 1):
                return False

        return True

    def _classify_const(self):
        def add_original():

            ins = []
            for x in self.ins:
                in_x = OriginalMode.gen_in(x["shape"])
                if "range" in x:
                    in_x["range"] = x["range"]
                ins.append(in_x)
            return [ins]

        ret = []
        ret.extend(add_original())
        return ret

    def _classify_var(self):
        def add_special():
            ins_list = []
            if self.format in ["ND"]:
                for rs, p, sp in zip(SpecialMode2D.REGS, SpecialMode2D.PATTERS,
                                     SpecialMode2D.STRICT_PATTERNS):
                    matched_list = SpecialMode2D.match(rs, self.f_patterns)
                    if not any(matched_list):
                        continue
                    ins_list.append(SpecialMode2D.gen_ins(matched_list, p, sp, self.format, self.normalized_shapes))
                return ins_list

        def add_original():
            ins = []
            for x in self.ins:
                in_x = OriginalMode.gen_in(x["shape"])
                if "range" in x:
                    in_x["range"] = x["range"]
                ins.append(in_x)

            return [ins]

        ret = []
        ret.extend(add_original())
        return ret


def _trans_one_to_common(f_patterns):
    if len(f_patterns) > 1:
        res = []
        for str in f_patterns:
            str_i = ''
            if str[0] == 'A':
                str_i = '0' + str[1]
            res.append(str_i)
        return res


def _simplify_shape(d_1, d_2):
    shape1, shape2 = d_1["shape"], d_2["shape"]
    len1, len2 = len(shape1), len(shape2)
    diff12, diff21 = len1 - len2, len2 - len1

    shape1 = diff21 * [1] + list(shape1)
    shape2 = diff12 * [1] + list(shape2)
    f_shape1, f_shape2 = [1], [1]

    range1 = diff21 * [(1, 1)] + list(d_1["range"])
    range2 = diff12 * [(1, 1)] + list(d_2["range"])
    f_range1, f_range2 = [(1, 1)], [(1, 1)]

    state = ShapeSimplifier.State.ONE
    for i, (s_1, s_2, r_1, r_2) in enumerate(zip(shape1, shape2, range1, range2)):
        is_reduce_axis = 1
        state_i = ShapeSimplifier.get_state(s_1, s_2, is_reduce_axis)
        operator = ShapeSimplifier.get_operator(state, state_i)

        if operator == ShapeSimplifier.Operator.FUSED:
            f_shape1[-1] = ShapeSimplifier.combine_dim(f_shape1[-1], s_1)
            f_shape2[-1] = ShapeSimplifier.combine_dim(f_shape2[-1], s_2)
            f_range1[-1] = ShapeSimplifier.combine_range(f_range1[-1], r_1)
            f_range2[-1] = ShapeSimplifier.combine_range(f_range2[-1], r_2)
        else:
            f_shape1.append(s_1)
            f_shape2.append(s_2)
            f_range1.append(r_1)
            f_range2.append(r_2)
        if state_i != ShapeSimplifier.State.ONE:
            state = state_i

    return shape1, range1, shape2, range2


def _get_reduce_axis_index(nor_shape: list, format: str, simple_shape1, simple_shape2):
    if format == 'ND':
        reduce_axis_index = len(simple_shape1) - 1
    else:
        if nor_shape[0][0] == nor_shape[1][0] and nor_shape[0][0] == 1:
            reduce_axis_index = 0
        else:
            reduce_axis_index = 1
    return reduce_axis_index


def _get_strict_pattern(shape1, shape2, format, reduce_index):
    strict_pattern = []
    for i, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if i == reduce_index:
            if shape1[-1] > 1 and shape2[-1] > 1:
                strict_pattern.extend([[AxisType.REDUCE_COMMON]])
            elif shape1[-1] == -1 and shape2[-1] > 1:
                strict_pattern.extend([[AxisType.REDUCE_COMMON, AxisType.REDUCE_ATOB]])
            elif shape1[-1] > 1 and shape2[-1] == -1:
                strict_pattern.extend([[AxisType.REDUCE_COMMON, AxisType.REDUCE_BTOA]])
            else:
                strict_pattern.extend([[AxisType.REDUCE_COMMON, AxisType.REDUCE_BTOA, AxisType.REDUCE_ATOB]])
        else:
            strict_pattern.extend([StrictPattern.get_pattern(dim1, dim2)])
    return strict_pattern


def _combinations(dim_patterns):
    def divide(i, c_shape_patterns):
        if i == len(dim_patterns):
            return c_shape_patterns
        shape_patterns = []
        for sp in c_shape_patterns:
            for dp in dim_patterns[i]:
                spc = sp.copy()
                spc.append(dp)
                shape_patterns.append(spc)
        return divide(i + 1, shape_patterns)

    def convert_to_str(p_list):
        q = p_list
        return "".join(q) if any(q) else "0"

    return [convert_to_str(q) for q in divide(0, [[]])]


def _match(reg, patterns):
    for p in patterns:
        if re.match(reg, p):
            return True
    return False


def _get_broadcast_axis_size(pattern):
    broadcast_axis_size = 0
    for p_i in pattern:
        if AxisType.ATOB in p_i or AxisType.BTOA in p_i or AxisType.REDUCE_BTOA in p_i or AxisType.REDUCE_ATOB in p_i:
            broadcast_axis_size += 1
    return broadcast_axis_size


class AxisType:
    """
    there are senven axis type
    """
    ONE = "A"
    COMMON = "0"
    ATOB = "1"
    BTOA = "2"
    REDUCE_COMMON = "3"
    REDUCE_ATOB = "4"
    REDUCE_BTOA = "5"


class SpecialMode2D:
    """
    SpecialMode2D const
    """
    PATTERS = [
        (COMMON, COMMON_REDUCE),  # AR
        (COMMON, BROADCAST_REDUCE),  # A(BR)
        (BROADCAST, COMMON_REDUCE),  # BR
    ]

    # what's 012 means, @see AxisType
    STRICT_PATTERNS = [
        ["03"],
        ["04", "05"],
        ["13", "23"],
    ]

    REGS = [
        ["^03$"],
        ["^04$", "^05$"],
        ["^13$", "^23$"],
    ]

    @classmethod
    def match(cls, regs, patterns):
        """
        match regex
        """
        return [_match(r, patterns) for r in regs]

    @classmethod
    def gen_ins(cls, matched_list, pattern, strict_pattern, format, normalized_shape):
        """
        generate inputs
        """

        def gen_in(shape):
            return {"shape": shape,
                    "range": util.generate_range(shape),
                    "support_reduce": True,
                    "mode": SPECIAL,
                    "pattern": pattern,
                    "format": format
                    }

        if all(matched_list):
            [shape1, shape2] = [[-1] * len(pattern)] * 2
            normal_shape1, normal_shape2 = normalized_shape
            for i in range(len(shape1)):
                if normal_shape1[i] != -1 and normal_shape2[i] != -1:
                    shape1[i] = normal_shape1[i]
                    shape2[i] = normal_shape2[i]
            return [gen_in(shape1), gen_in(shape2)]

        shape1, shape2 = [], []
        out = strict_pattern[matched_list.index(True)]
        for i, sp_i in strict_pattern[matched_list.index(True)]:
            if sp_i == '0':
                shape1.append(-1)
                shape2.append(-1)
            elif sp_i == '1':
                shape1.append(1)
                shape2.append(-1)
            elif sp_i == '2':
                shape1.append(-1)
                shape2.append(1)
            if normal_shape1[i] != -1 and normal_shape2[i] != -1:
                shape1[-1] = normal_shape1[i]
                shape2[-1] = normal_shape2[i]
        return [gen_in(shape1), gen_in(shape2)]


class OriginalMode:
    """
    Original Mode
    """

    @classmethod
    def gen_in(cls, shape):
        """
        generate input
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "support_reduce": True,
                "mode": ORIGINAL,
                }


class ConstMode:
    """
    Const Mode
    """

    @classmethod
    def gen_in(cls, shape):
        """
        generate input
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "mode": CONST,
                "support_reduce": True,
                }


class StrictPattern:
    """
    StrictPattern
    """
    KNOWN, UNKNOWN, ONE = 1, -1, 0

    CATEGORY = {
        (ONE, ONE): [AxisType.ONE],
        (ONE, KNOWN): [AxisType.ATOB],
        (ONE, UNKNOWN): [AxisType.ONE, AxisType.ATOB],
        (KNOWN, ONE): [AxisType.BTOA],
        (KNOWN, KNOWN): [AxisType.COMMON],
        (KNOWN, UNKNOWN): [AxisType.COMMON, AxisType.BTOA],
        (UNKNOWN, ONE): [AxisType.ONE, AxisType.BTOA],
        (UNKNOWN, KNOWN): [AxisType.COMMON, AxisType.ATOB],
        (UNKNOWN, UNKNOWN): [AxisType.COMMON, AxisType.ATOB, AxisType.BTOA, AxisType.ONE],
    }

    @classmethod
    def get_pattern(cls, dim1, dim2):
        def get_axis_type(_dim):
            return cls.UNKNOWN if _dim < 0 else cls.KNOWN if _dim > 1 else cls.ONE

        return cls.CATEGORY[(get_axis_type(dim1), get_axis_type(dim2))]


class ShapeSimplifier:
    """
    ShapeSimplifier
    """

    class State(Enum):
        """
        the axis type in same location of two shapes
        """
        ONE = auto()
        CONST = auto()
        ATOB = auto()
        BTOA = auto()
        UNKNOWN_BROADCAST = auto()
        REDUCE = auto()

    class Operator(Enum):
        """
        the fusion behavior of two contiguous axis
        """
        # can fuse axis
        FUSED = auto()
        # can not fuse axis
        ALONE = auto()

    @classmethod
    def get_state(cls, dim1, dim2, is_reduce_axis):
        """
        get_state
        :param dim1:
        :param dim2:
        :return:
        """
        if is_reduce_axis:
            return cls.State.REDUCE
        if dim1 == 1 and dim2 == 1:
            return cls.State.ONE
        if dim1 > 1 and dim2 > 1:
            return cls.State.CONST
        if dim1 == 1 and dim2 != 1:
            return cls.State.ATOB
        if dim1 != 1 and dim2 == 1:
            return cls.State.BTOA
        return cls.State.UNKNOWN_BROADCAST

    @classmethod
    def get_operator(cls, state1, state2):
        """
        get_operator
        :param state1:
        :param state2:
        :return:
        """
        if state1 == cls.State.ONE or state2 == cls.State.ONE:
            return cls.Operator.FUSED
        if state1 == cls.State.UNKNOWN_BROADCAST or state2 == cls.State.UNKNOWN_BROADCAST:
            return cls.Operator.ALONE
        if state1 == cls.State.REDUCE or state2 == cls.State.REDUCE:
            return cls.Operator.ALONE
        if state1 == state2:
            return cls.Operator.FUSED

        return cls.Operator.ALONE

    @classmethod
    def combine_dim(cls, dim1, dim2):
        """
        combine_dim
        :param dim1:
        :param dim2:
        :return:
        """
        return -1 if -1 in (dim1, dim2) else dim1 * dim2

    @classmethod
    def combine_range(cls, range1, range2):
        """
        combine_range
        :param range1:
        :param range2:
        :return:
        """
        return util.combine_range([range1, range2])
