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
classifier of shape in elewise
"""
COMMON = "common"
BROADCAST = "broadcast"
SCALAR = "scalar"
SPECIAL = "special"
SPECIAL_SCALAR = "special_scalar"
CONST = "const"
ORIGINAL = "original"
VAR_BOUND_LIMIT = 2147483647


class ElewiseClassifier:
    """
    ElewiseClassifier
    """

    def __init__(self, ins: list, support_broadcast: bool = False):
        self.ins = ins
        self.support_broadcast = support_broadcast

        shapes = [item["shape"] for item in self.ins]
        dim_length = max([len(item) for item in shapes])
        self.dim_length = dim_length

        if support_broadcast and len(ins) != 2:
            raise RuntimeError("The size of input parameter must be 2, "
                               "if support broadcast")

    def classify(self):
        """
        :return:
        """
        if self.support_broadcast:
            self._fill_dims()
        if self._check_const():
            return self._classify_const()
        return self._classify_var()

    def _fill_dims(self):
        for item in self.ins:
            shape_i = list(item["shape"])
            d_v = self.dim_length - len(shape_i)
            item["shape"] = [1] * d_v + shape_i
            if "range" in item:
                item["range"] = [(1, 1)] * d_v + list(item["range"])

    def _check_const(self):
        shapes = [item["shape"] for item in self.ins]
        divide_axis_len = 0
        divide_axis_limit = 5
        for i in range(self.dim_length):
            shapes_i = [shape[i] for shape in shapes]
            min_dim_v = min(shapes_i)
            max_dim_v = max(shapes_i)
            if min_dim_v == -1 and max_dim_v == -1:
                return False
            if self.support_broadcast and min_dim_v == -1 and max_dim_v == 1:
                return False
            divide_axis_len += 1
        if self.support_broadcast:
            return divide_axis_len <= divide_axis_limit
        return True

    def _classify_const(self):
        shapes = [item["shape"] for item in self.ins]

        def division(i, _shapes):
            if i == self.dim_length:
                return [_shapes]

            shapes_i = [shape[i] for shape in shapes]
            min_dim_v, max_dim_v = min(shapes_i), max(shapes_i)
            if not self.support_broadcast:
                append_v(_shapes, max_dim_v)
                return division(i + 1, _shapes)

            if min_dim_v != -1:
                append_i(_shapes, i)
                return division(i + 1, _shapes)

            ret_shapes = []
            _shapes_copy = copy(_shapes)
            append_b(_shapes_copy, i, 1)
            ret_shapes.extend(division(i + 1, _shapes_copy))
            _shapes_copy = copy(_shapes)
            append_b(_shapes_copy, i, max_dim_v)
            ret_shapes.extend(division(i + 1, _shapes_copy))
            return ret_shapes

        def handle_broadcast(_shapes_list):
            ret = []
            for shapes_i in _shapes_list:
                shape1, shape2 = shapes_i[0], shapes_i[1]
                x_0 = {"shape": shape1,
                       "range": generate_range(shape1),
                       "mode": CONST,
                       "support_broadcast": True,
                       }
                x_1 = {"shape": shape2,
                       "range": generate_range(shape2),
                       "mode": CONST,
                       "support_broadcast": True,
                       }
                ret.append([x_0, x_1])
            return ret

        def handle_non_broadcast(_shapes_list):
            ret = []
            for shapes_i in _shapes_list:
                t_ins = []
                for shape_i in shapes_i:
                    x_0 = {"shape": shape_i,
                           "range": generate_range(shape_i),
                           "mode": CONST,
                           "support_broadcast": False,
                           }
                    t_ins.append(x_0)
                ret.append(t_ins)
            return ret

        def generate_range(shape):
            return [(i, i) for i in shape]

        def append_i(array_list, i):
            for _i, shape in enumerate(array_list):
                shape.append(shapes[_i][i])

        def append_v(array_list, dim_v):
            for _i, shape in enumerate(array_list):
                shape.append(dim_v)

        def append_b(array_list, i, dim_v):
            for _i, shape in enumerate(array_list):
                shape.append(max(shapes[_i][i], dim_v))

        def copy(array_list):
            return [array.copy() for array in array_list]

        shapes_list = division(0, [[] for _ in range(len(self.ins))])
        return handle_broadcast(shapes_list) if self.support_broadcast \
            else handle_non_broadcast(shapes_list)

    def _classify_var(self):
        in_len = len(self.ins)

        def handle_broadcast():
            def add_special(_ret):
                patters = [
                    (COMMON,),
                    (COMMON, BROADCAST),
                    (COMMON, BROADCAST, COMMON),
                    (BROADCAST,),
                    (BROADCAST, COMMON)
                ]
                broadcast_direction = _get_broadcast_direction(*self.ins)
                for pattern in patters:
                    if len(pattern) > self.dim_length:
                        continue
                    shape0, shape1 = [], []
                    range0, range1 = [], []
                    for axis_type in pattern:
                        if axis_type == BROADCAST:
                            if broadcast_direction == 0b010:
                                shape0.append(1)
                                shape1.append(-1)
                                range0.append((1, 1))
                                range1.append((1, None))
                                continue
                            if broadcast_direction == 0b001:
                                shape0.append(-1)
                                shape1.append(1)
                                range0.append((1, None))
                                range1.append((1, 1))
                                continue
                        shape0.append(-1)
                        shape1.append(-1)
                        range0.append((1, None))
                        range1.append((1, None))

                    x_0 = {"shape": shape0,
                           "range": range0,
                           "support_broadcast": True,
                           "mode": SPECIAL,
                           "pattern": pattern
                           }
                    x_1 = {"shape": shape1,
                           "range": range1,
                           "support_broadcast": True,
                           "mode": SPECIAL,
                           "pattern": pattern
                           }
                    _ret.append([x_0, x_1])

            def add_special_scalar(_ret):
                # shape pattern: (-1,) vs (1,)
                x_0 = {"shape": (-1,),
                       "range": [(1, None)],
                       "support_broadcast": True,
                       "mode": SPECIAL_SCALAR,
                       "pattern": (BROADCAST, SCALAR)
                       }
                x_1 = {"shape": (1,),
                       "range": [(1, 1)],
                       "support_broadcast": True,
                       "mode": SPECIAL_SCALAR,
                       "pattern": (BROADCAST, SCALAR)
                       }
                _ret.append([x_0, x_1])

                # shape pattern: (1,) vs (-1,)
                x_0 = {"shape": (1,),
                       "range": [(1, 1)],
                       "support_broadcast": True,
                       "mode": SPECIAL_SCALAR,
                       "pattern": (SCALAR, BROADCAST)
                       }
                x_1 = {"shape": (-1,),
                       "range": [(1, None)],
                       "support_broadcast": True,
                       "mode": SPECIAL_SCALAR,
                       "pattern": (SCALAR, BROADCAST)
                       }
                _ret.append([x_0, x_1])

            def add_original(_ret):
                if self.dim_length > 1:
                    item = []
                    for x_in in self.ins:
                        input_x = {"shape": x_in["shape"],
                                   "support_broadcast": True,
                                   "mode": ORIGINAL,
                                   }
                        if "range" in x_in:
                            input_x["range"] = x_in["range"]
                        item.append(input_x)
                    _ret.append(item)

            ret = []
            add_special(ret)
            add_special_scalar(ret)
            add_original(ret)
            return ret

        def handle_non_broadcast():
            input_x = {"shape": (-1,),
                       "range": [(1, None)],
                       "support_broadcast": False,
                       "mode": SPECIAL,
                       "pattern": (COMMON,)
                       }
            item = [input_x] * in_len
            return [item]

        return handle_broadcast() if self.support_broadcast \
            else handle_non_broadcast()


def _simplify_shape(d_1, d_2):
    def get_state(_a, _b):
        if _a == 1 and _b == 1:
            return 0
        if _a > 1 and _b > 1:
            return 1
        if _a == 1 and _b != 1:
            return 2
        if _a != 1 and _b == 1:
            return 3
        return 4

    def get_operator(state1, state2):
        if state1 == 0 or state2 == 0:
            return 0
        if state1 == 4 or state2 == 4:
            return 1
        if state1 == state2:
            return 0
        return 1

    def combine_range(r1_, r2_):
        def mul(_a, _b):
            if _a is None or _b is None:
                return None
            _bound = _a * _b
            return None if _bound > VAR_BOUND_LIMIT else _bound

        return [mul(a, b) for a, b in zip(r1_, r2_)]

    shape1, shape2 = d_1["shape"], d_2["shape"]
    range1, range2 = d_1["range"], d_2["range"]
    len1, len2 = len(shape1), len(shape2)

    diff12, diff21 = len1 - len2, len2 - len1
    shape1 = diff21 * [1] + list(shape1)
    shape2 = diff12 * [1] + list(shape2)
    range1 = diff21 * [(1, 1)] + list(range1)
    range2 = diff12 * [(1, 1)] + list(range2)
    f_shape1, f_shape2 = [1], [1]
    f_range1, f_range2 = [(1, 1)], [(1, 1)]

    # '0: a, b all equals 1, for example: (a, b) = (1, 1)
    # '1: a, b all const and not equal 1, for example: (a, b) = (8, 8)
    # '2: a broadcast to b, for example: (a, b) = (1, 4), (a, b) = (1, -1)
    # '3: b broadcast to a, for example: (a, b) = (4, 1), (a, b) = (-1, 1)
    # '4: runtime-broadcast, for example: (a, b) = (-1, -1), (a, b) = (-1, 8), (a, b) = (8, -1)
    state = 0
    for s_1, s_2, r_1, r_2 in zip(shape1, shape2, range1, range2):
        state_i = get_state(s_1, s_2)
        # 0: can fuse
        # 1: can not fuse
        operator = get_operator(state, state_i)
        if operator == 0:
            f_shape1[-1] = -1 if -1 in (f_shape1[-1], s_1) else f_shape1[-1] * s_1
            f_shape2[-1] = -1 if -1 in (f_shape2[-1], s_2) else f_shape2[-1] * s_2
            f_range1[-1] = combine_range(f_range1[-1], r_1)
            f_range2[-1] = combine_range(f_range2[-1], r_2)
        else:
            f_shape1.append(s_1)
            f_shape2.append(s_2)
            f_range1.append(r_1)
            f_range2.append(r_2)
        if state_i != 0:
            state = state_i

    return f_shape1, f_range1, f_shape2, f_range2


def _get_strict_pattern(shape1, shape2):
    def get_axis_type(_a, _b):
        a_ = -1 if _a < 0 else 2 if _a > 1 else 1
        b_ = -1 if _b < 0 else 2 if _b > 1 else 1
        # axis type consists of three binary digits
        #     No.1: common axis
        #     No.2: a broadcast to b
        #     No.3: b broadcast to a
        # for example: 0b010 means a broadcast b
        #              0b110 means many be common axis or a broadcast b
        #              0b111 means many be common axis or a broadcast b or b broadcast to a
        d_1 = {
            (-1, -1): 0b111,
            (-1, 1): 0b101,
            (-1, 2): 0b110,
            (1, -1): 0b110,
            (1, 1): 0b100,
            (1, 2): 0b010,
            (2, -1): 0b101,
            (2, 1): 0b001,
            (2, 2): 0b100,
        }
        return d_1[(a_, b_)]

    return [get_axis_type(a, b) for a, b in zip(shape1, shape2)]


def _get_broadcast_direction(d_1, d_2):
    """
    When can match one broadcast pattern and ascertain direction, return it.
    Otherwise, return None.
    """
    f_shape1, _, f_shape2, _ = _simplify_shape(d_1, d_2)
    strict_pattern = _get_strict_pattern(f_shape1, f_shape2)

    strict_known_direction = set()
    known_direction = set()
    unknown_direction = set()
    for _x in strict_pattern:
        if _x in (0b010, 0b001):
            strict_known_direction.add(_x & 0b011)
        elif _x in (0b110, 0b101):
            known_direction.add(_x & 0b011)
        elif _x in (0b111, 0b011):
            unknown_direction.add(_x & 0b011)

    if len(strict_known_direction) == 1:
        return strict_known_direction.pop()

    if len(strict_known_direction) == 0 and len(known_direction) == 1 \
            and len(unknown_direction) == 0:
        return known_direction.pop()

    return None
