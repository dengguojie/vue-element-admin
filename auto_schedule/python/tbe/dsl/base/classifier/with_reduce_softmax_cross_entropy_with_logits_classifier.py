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
from tbe.dsl.base import operation

from . import util

ONE = "one"
COMMON = "common"
BROADCAST = "broadcast"
COMMON_REDUCE = "common_reduce"
BROADCAST_REDUCE = "broadcast_reduce"
SPECIAL = "special"
CONST = "const"
ORIGINAL = "original"
ORIGINAL_AND_CUT = "original_and_cut"
COPY = "copy"
COPY_AND_CUT = "copy_and_cut"
MAX_INT32_VALUE = 2147483647
MAX_COEXIST_NUM = 10
BLOCK_SIZE_BYTE = 32


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
        self.dtype = self.ins[0]["dtype"]
        self.dtype_size = 2 if self.dtype == "float16" else 4
        shapes = [x["shape"] for x in self.ins]
        self.dim_length = max([len(s) for s in shapes])
        self.normalized_ins = self._normalize()
        f_shape1, f_range1, f_shape2, f_range2, = _simplify_shape(*self.normalized_ins)
        self.f_shapes = [f_shape1, f_shape2]
        self.f_ranges = [f_range1, f_range2]

    def classify(self):
        """
        classify
        :return:
        """
        def gen_template(shape, range, mode):
            return {"shape": shape,
                    "range": range,
                    "mode": mode
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

        [[(r00_l, r00_r), (r01_l, r01_r)], [(r10_l, r10_r), (r11_l, r11_r)]] = self.f_ranges
        tail_not_broadcast = (r01_l > 1 and r11_l > 1) or (r01_l == r01_r == 1 and r11_l == r11_r == 1)
        tail_may_broadcast = not tail_not_broadcast
        batch_not_broadcast = (r00_l > 1 and r10_l > 1) or (r00_l == r00_r == 1 and r10_l == r10_r == 1)
        batch_may_broadcast = not batch_not_broadcast
        ub_size = operation.get_context().get("ub_size")
        num_per_block = BLOCK_SIZE_BYTE // self.dtype_size
        bound_size = (ub_size // 4 // MAX_COEXIST_NUM) // 16 * 16
        range0, range1 = _process_range(self.f_ranges[0], self.f_ranges[1])
        r01_r = range0[0][1]
        r11_r = range0[1][1]
        may_cut_reduce = (r01_r >= bound_size or r11_r >= bound_size)
        not_cut_reduce = not may_cut_reduce

        if not_cut_reduce:
            if tail_not_broadcast and batch_not_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])
                return res

            if tail_not_broadcast and batch_may_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])
                return res

            if tail_may_broadcast and batch_not_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])
                return res

            if tail_may_broadcast and batch_may_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec9"),
                                gen_template(special_shape1, special_range1, "vec9")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec6"),
                                gen_template(special_shape1, special_range1, "vec6")])
                return res

        if may_cut_reduce:
            if tail_not_broadcast and batch_not_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY_AND_CUT),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY_AND_CUT)])
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])
                return res

            if tail_not_broadcast and batch_may_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL_AND_CUT),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL_AND_CUT)])
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec1_and_cut"),
                                gen_template(special_shape1, special_range1, "vec1_and_cut")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec4_and_cut"),
                                gen_template(special_shape1, special_range1, "vec4_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY_AND_CUT),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY_AND_CUT)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])
                return res

            if tail_may_broadcast and batch_not_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL_AND_CUT),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL_AND_CUT)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY_AND_CUT),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY_AND_CUT)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec2_and_cut"),
                                gen_template(special_shape1, special_range1, "vec2_and_cut")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec8_and_cut"),
                                gen_template(special_shape1, special_range1, "vec8_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])
                return res

            if tail_may_broadcast and batch_may_broadcast:
                res = []
                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL_AND_CUT),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL_AND_CUT)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], ORIGINAL),
                            gen_template(self.f_shapes[1], self.f_ranges[1], ORIGINAL)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY_AND_CUT),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY_AND_CUT)])

                res.append([gen_template(self.f_shapes[0], self.f_ranges[0], COPY),
                            gen_template(self.f_shapes[1], self.f_ranges[1], COPY)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec1_and_cut"),
                                gen_template(special_shape1, special_range1, "vec1_and_cut")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec4_and_cut"),
                                gen_template(special_shape1, special_range1, "vec4_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec2_and_cut"),
                                gen_template(special_shape1, special_range1, "vec2_and_cut")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec8_and_cut"),
                                gen_template(special_shape1, special_range1, "vec8_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec9_and_cut"),
                                gen_template(special_shape1, special_range1, "vec9_and_cut")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec9"),
                                gen_template(special_shape1, special_range1, "vec9")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec6_and_cut"),
                                gen_template(special_shape1, special_range1, "vec6_and_cut")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range(special_range0, special_range1)
                if is_legal_range(special_range0[0]) and is_legal_range(special_range0[1]) \
                        and is_legal_range(special_range1[0]) and is_legal_range(special_range1[1]):
                    res.append([gen_template(special_shape0, special_range0, "vec6"),
                                gen_template(special_shape1, special_range1, "vec6")])
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
