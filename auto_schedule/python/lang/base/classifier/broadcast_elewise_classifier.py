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
from te.utils.error_manager.error_manager_util import get_error_message

from . import util

COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
ORIGINAL = "original"
VAR_BOUND_LIMIT = 2147483647


class BroadcastElewiseClassifier:
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
        shapes = [x["shape"] for x in self.ins]
        self.dim_length = max([len(s) for s in shapes])

        self.normalized_ins = self._normalize()
        self.normalized_shapes = [x["shape"] for x in self.normalized_ins]

        f_shape1, f_range1, f_shape2, f_range2 = _simplify_shape(*self.normalized_ins)
        self.f_shapes = [f_shape1, f_shape2]
        self.f_ranges = [f_range1, f_range2]

        self.f_strict_pattern = _get_strict_pattern(f_shape1, f_shape2)
        self.f_patterns = _combinations(self.f_strict_pattern)

    def classify(self):
        """
        classify
        :return:
        """
        return self._classify_const() if self._is_const() else self._classify_var()

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
        def divide(i, _shapes):
            if i == self.dim_length:
                return [_shapes]

            dims_i = [s[i] for s in self.normalized_shapes]
            min_dim_v, max_dim_v = min(dims_i), max(dims_i)

            # 1. don't need divide, all const value in the current axis
            if min_dim_v != -1:
                append_i(_shapes, i)
                return divide(i + 1, _shapes)

            # 2. -1 divide to 1 and the dim value, need cover broadcast
            ret_shapes = []

            # 2.1. -1 -> 1, x -> x
            _shapes_copy = copy(_shapes)
            append_b(_shapes_copy, i, 1)
            ret_shapes.extend(divide(i + 1, _shapes_copy))

            # 2.2. -1 -> x, x -> x
            _shapes_copy = copy(_shapes)
            append_b(_shapes_copy, i, max_dim_v)
            ret_shapes.extend(divide(i + 1, _shapes_copy))

            return ret_shapes

        def append_i(_shapes, dim_i):
            for i, _shape in enumerate(_shapes):
                _shape.append(self.normalized_shapes[i][dim_i])

        def append_b(_shapes, dim_i, dim_v):
            for i, _shape in enumerate(_shapes):
                _shape.append(max(self.normalized_shapes[i][dim_i], dim_v))

        def copy(_shapes):
            return [_shape.copy() for _shape in _shapes]

        ret = []
        for shapes in divide(0, [[] for _ in self.normalized_ins]):
            ret.append([ConstMode.gen_in(shapes[0]), ConstMode.gen_in(shapes[1])])

        return ret

    def _classify_var(self):
        def add_special():
            ins_list = []
            for rs, p, sp in zip(SpecialMode.REGS, SpecialMode.PATTERS,
                                 SpecialMode.STRICT_PATTERNS):
                matched_list = SpecialMode.match(rs, self.f_patterns)
                if not any(matched_list):
                    continue
                ins_list.append(SpecialMode.gen_ins(matched_list, p, sp))

            return ins_list

        def add_special_scalar():
            ins_list = []
            for r, p, ss in zip(SpecialScalarMode.REGEX, SpecialScalarMode.PATTERNS,
                                SpecialScalarMode.SHAPES_LIST):
                if _match(r, self.f_patterns):
                    x_0 = SpecialScalarMode.gen_in(ss[0], p)
                    x_1 = SpecialScalarMode.gen_in(ss[1], p)
                    ins_list.append([x_0, x_1])

            return ins_list

        def add_original():
            if _get_broadcast_axis_size(self.f_strict_pattern) <= 1:
                return []

            ins = []
            for x in self.ins:
                in_x = OriginalMode.gen_in(x["shape"])
                if "range" in x:
                    in_x["range"] = x["range"]
                ins.append(in_x)

            return [ins]

        ret = []
        ret.extend(add_special())
        ret.extend(add_special_scalar())
        ret.extend(add_original())
        return ret


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
    for s_1, s_2, r_1, r_2 in zip(shape1, shape2, range1, range2):
        state_i = ShapeSimplifier.get_state(s_1, s_2)
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

    return f_shape1, f_range1, f_shape2, f_range2


def _get_strict_pattern(shape1, shape2):
    return [StrictPattern.get_pattern(a, b) for a, b in zip(shape1, shape2)]


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
        p = list(map(lambda x: "" if x == "A" else x, p_list))
        return "".join(p) if any(p) else "0"

    return [convert_to_str(p) for p in divide(0, [[]])]


def _match(reg, patterns):
    for p in patterns:
        if re.match(reg, p):
            return True
    return False


def _get_broadcast_axis_size(pattern):
    broadcast_axis_size = 0
    for p_i in pattern:
        if AxisType.ATOB in p_i or AxisType.BTOA in p_i:
            broadcast_axis_size += 1
    return broadcast_axis_size


class AxisType:
    """
    there are three axis type, one axis may contains some of them
    """
    # 'All equals 1, for example: (a, b) = (1, 1)
    ONE = "A"
    # common axis, not need broadcast, and not equals 1
    COMMON = "0"
    # a broadcast to b
    ATOB = "1"
    # b broadcast to a
    BTOA = "2"


class SpecialMode:
    """
    SpecialMode const
    """
    # The (BROADCAST,) pattern covered by special scalar mode
    PATTERS = [
        (COMMON,),
        (COMMON, BROADCAST),
        (COMMON, BROADCAST, COMMON),
        (BROADCAST, COMMON)
    ]

    # what's 012 means, @see AxisType
    STRICT_PATTERNS = [
        ["0"],
        ["01", "02"],
        ["010", "020"],
        ["10", "20"],
    ]

    REGS = [
        ["^0+$"],
        ["^0+1+$", "^0+2+$"],
        ["^0+1+0+$", "^0+2+0+$"],
        ["^1+0+$", "^2+0+$"],
    ]

    @classmethod
    def match(cls, regs, patterns):
        """
        match regex
        :param regs:
        :param patterns:
        :return:
        """
        return [_match(r, patterns) for r in regs]

    @classmethod
    def gen_ins(cls, matched_list, pattern, strict_pattern):
        """
        generate inputs
        :param matched_list:
        :param pattern:
        :param strict_pattern:
        :return:
        """

        def gen_in(shape):
            return {"shape": shape,
                    "range": util.generate_range(shape),
                    "support_broadcast": True,
                    "mode": SPECIAL,
                    "pattern": pattern
                    }

        if all(matched_list):
            [shape1, shape2] = [[-1] * len(pattern)] * 2
            return [gen_in(shape1), gen_in(shape2)]

        shape1, shape2 = [], []
        for sp_i in strict_pattern[matched_list.index(True)]:
            if sp_i == '0':
                shape1.append(-1)
                shape2.append(-1)
            elif sp_i == '1':
                shape1.append(1)
                shape2.append(-1)
            elif sp_i == '2':
                shape1.append(-1)
                shape2.append(1)
        return [gen_in(shape1), gen_in(shape2)]


class SpecialScalarMode:
    """
    SpecialScalarMode
    """
    PATTERNS = [
        (SCALAR, BROADCAST),
        (BROADCAST, SCALAR),
    ]

    REGEX = [
        "^1+$",
        "^2+$",
    ]

    SHAPES_LIST = [
        [[1], [-1]],
        [[-1], [1]],
    ]

    @classmethod
    def gen_in(cls, shape, pattern):
        """
        generate input
        :param shape:
        :param pattern:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "support_broadcast": True,
                "mode": SPECIAL_SCALAR,
                "pattern": pattern
                }


class OriginalMode:
    """
    Original Mode
    """

    @classmethod
    def gen_in(cls, shape):
        """
        generate input
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "support_broadcast": True,
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
        :param shape:
        :return:
        """
        return {"shape": shape,
                "range": util.generate_range(shape),
                "mode": CONST,
                "support_broadcast": True,
                }


class StrictPattern:
    """
    StrictPattern
    """
    # 'KNOWN: dim > 1; UNKNOWN: dim = -1; ONE: dim == 1
    KNOWN, UNKNOWN, ONE = 1, -1, 0

    # (axis types, strict pattern) mapping
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
        # 'a, b all equals 1, for example: (a, b) = (1, 1)
        ONE = auto()
        # 'a, b all const and not equal 1, for example: (a, b) = (8, 8)
        CONST = auto()
        # 'a broadcast to b, for example: (a, b) = (1, 4), (a, b) = (1, -1)
        ATOB = auto()
        # 'b broadcast to a, for example: (a, b) = (4, 1), (a, b) = (-1, 1)
        BTOA = auto()
        # 'runtime-broadcast, for example: (a, b) = (-1, -1), (a, b) = (-1, 8), (a, b) = (8, -1)
        UNKNOWN_BROADCAST = auto()

    class Operator(Enum):
        """
        the fusion behavior of two contiguous axis
        """
        # can fuse axis
        FUSED = auto()
        # can not fuse axis
        ALONE = auto()

    @classmethod
    def get_state(cls, dim1, dim2):
        """
        get_state
        :param dim1:
        :param dim2:
        :return:
        """
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
