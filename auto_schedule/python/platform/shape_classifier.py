#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

classifier of shape
"""
import math
from enum import Enum, auto

COMMON = "common"
BROADCAST = "broadcast"
REDUCE = "reduce"
SPECIAL = "special"
CONST = "const"
ORIGINAL = "original"


class Mode(Enum):
    """
    Mode
    """
    NONE = auto()
    ELEWISE = auto()
    ELEWISE_WITH_BROADCAST = auto()
    REDUCE = auto()


def classify(ins: list, mode: Mode = Mode.NONE):
    """
    :param ins:
    :param mode:
    :return:
    """
    if mode == Mode.NONE:
        return ins
    elif mode == Mode.ELEWISE:
        return _classify_elewise(ins, support_broadcast=False)
    elif mode == Mode.ELEWISE_WITH_BROADCAST:
        return _classify_elewise(ins, support_broadcast=True)
    elif mode == Mode.REDUCE:
        return _classify_reduce(ins)

    return ins


def _classify_elewise(ins: list, support_broadcast=False):
    """
    :param ins:
    :param support_broadcast:
    :return:
    """
    return [ins]
    # 'return ElewiseClassifier(ins, support_broadcast).classify()


def _classify_reduce(ins: list):
    """
    :param ins:
    :return:
    """
    return ReduceClassifier(ins).classify()


class ElewiseClassifier:
    """
    ElewiseClassifier
    """

    def __init__(self, ins: list, support_broadcast):
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
                x0 = {"shape": shape1,
                      "range": generate_range(shape1),
                      "mode": CONST,
                      "support_broadcast": True,
                      }
                x1 = {"shape": shape2,
                      "range": generate_range(shape2),
                      "mode": CONST,
                      "support_broadcast": True,
                      }
                ret.append([x0, x1])
            return ret

        def handle_non_broadcast(_shapes_list):
            ret = []
            for shapes_i in _shapes_list:
                t_ins = []
                for shape_i in shapes_i:
                    x0 = {"shape": shape_i,
                          "range": generate_range(shape_i),
                          "mode": CONST,
                          "support_broadcast": False,
                          }
                    t_ins.append(x0)
                ret.append(t_ins)
            return ret

        def generate_range(shape):
            return [(i, i) for i in shape]

        def append_i(array_list, i):
            for _i, shape in enumerate(array_list):
                shape.append(shapes[_i][i])

        def append_v(array_list, v):
            for _i, shape in enumerate(array_list):
                shape.append(v)

        def append_b(array_list, i, v):
            for _i, shape in enumerate(array_list):
                shape.append(max(shapes[_i][i], v))

        def copy(array_list):
            return [array.copy() for array in array_list]

        shapes_list = division(0, [[] for _ in range(len(self.ins))])
        return handle_broadcast(shapes_list) if self.support_broadcast \
            else handle_non_broadcast(shapes_list)

    def _classify_var(self):
        in_len = len(self.ins)

        def handle_broadcast():
            patters = [
                (COMMON,),
                (COMMON, BROADCAST),
                (COMMON, BROADCAST, COMMON),
                (BROADCAST,),
                (BROADCAST, COMMON)
            ]
            ret = []
            for pattern in patters:
                if len(pattern) > self.dim_length:
                    continue
                x = {"shape": (-1,) * len(pattern),
                     "range": [(1, None)] * len(pattern),
                     "support_broadcast": True,
                     "mode": SPECIAL,
                     "pattern": pattern
                     }
                ret.append([x] * in_len)

            if self.dim_length > 1:
                item = []
                for x_in in self.ins:
                    x = {"shape": x_in["shape"],
                         "support_broadcast": True,
                         "mode": ORIGINAL,
                         }
                    if "range" in x_in:
                        x["range"] = x_in["range"]
                    item.append(x)
                ret.append(item)
            return ret

        def handle_non_broadcast():
            x = {"shape": (-1,),
                 "range": [(1, None)],
                 "support_broadcast": False,
                 "mode": SPECIAL,
                 "pattern": (COMMON,)
                 }
            item = [x] * in_len
            return [item]

        return handle_broadcast() if self.support_broadcast \
            else handle_non_broadcast()


class ReduceClassifier:
    """
    ReduceClassifier
    """

    def __init__(self, ins: list):
        self.ins = ins

    def classify(self):
        """
        :return:
        """
        # data
        x0 = self.ins[0]
        dim_len = len(x0["shape"])
        # reduce
        x1 = self.ins[1]
        reduce_axis_size = x1["shape"][0]

        max_reduce_axis_size = math.ceil(dim_len / 2)
        if reduce_axis_size < 0:
            reduce_axis_size = max_reduce_axis_size
        else:
            reduce_axis_size = min(reduce_axis_size, max_reduce_axis_size)

        ret = []
        upper_bound = min(reduce_axis_size * 2, dim_len)
        for i in range(1, upper_bound):
            pattern0, reduce_axis0 = self._generate_pattern(i)
            tx0 = {
                "shape": [-1] * len(pattern0),
                "range": [(1, None)] * len(pattern0),
                "mode": SPECIAL,
                "pattern": pattern0
            }
            ret.append([tx0, reduce_axis0])

        pattern1, reduce_axis1 = self._generate_pattern(upper_bound)
        if upper_bound >= dim_len:
            pattern1 = pattern1[1:]
            reduce_axis1 = [i - 1 for i in reduce_axis1]
        tx1 = {
            "shape": [-1] * len(pattern1),
            "range": [(1, None)] * len(pattern1),
            "mode": SPECIAL,
            "pattern": pattern1
        }
        ret.append([tx1, reduce_axis1])

        return ret

    # noinspection PyMethodMayBeStatic
    def _generate_pattern(self, size):
        p = [COMMON, REDUCE]
        pattern = [p[i % 2] for i in range(size + 1)]
        reduce_axis = [i for i in range(1, size + 1, 2)]
        return pattern, reduce_axis
