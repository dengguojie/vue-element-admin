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
classifier of softmax_cross_entropy_with_logits
"""
import re
from enum import Enum, auto
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from tbe.common.platform.platform_info import get_soc_spec
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

        def _is_legal_range(range0, range1):
            [(r00_l, r00_r), (r01_l, r01_r)], [(r10_l, r10_r), (r11_l, r11_r)] = range0, range1
            return (r00_l <= r00_r) and (r01_l <= r01_r) and (r10_l <= r10_r) and (r11_l <= r11_r)

        def _process_range_vs_2(dim_range):
            if dim_range[1] >= 2:
                return max(dim_range[0], 2), dim_range[1]
            else:
                return dim_range

        def _range_to_int(range_val):
            return MAX_INT32_VALUE if range_val is None else int(range_val)

        def _process_shape(range0, range1, shape0, shape1, is_copy=False):
            [(r00_l, r00_r), (r01_l, r01_r)], [(r10_l, r10_r), (r11_l, r11_r)] = range0, range1
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if r00_l == r00_r:
                dim00 = r00_l
            if r01_l == r01_r:
                dim01 = r01_l
            if r10_l == r10_r:
                dim10 = r10_l
            if r11_l == r11_r:
                dim11 = r11_l
            if is_copy and shape0 != shape1:
                if dim00 > 0:
                    dim10 = dim00
                if dim10 > 0:
                    dim00 = dim10
                if dim01 > 0:
                    dim11 = dim01
                if dim11 > 0:
                    dim01 = dim11
            shape0, shape1 = [dim00, dim01], [dim10, dim11]
            return shape0, shape1

        def _is_copy(shape0, shape1):
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if dim00 == dim10 and dim01 == dim11:
                return True
            return False

        def _is_vec1(shape0, shape1):
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if dim01 == dim11 and dim00 == 1 and dim10 != 1:
                return True
            return False

        def _is_vec2(shape0, shape1):
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if dim00 == dim10 and dim01 == 1 and dim11 != 1:
                return True
            return False

        def _is_vec4(shape0, shape1):
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if dim01 == dim11 and dim10 == 1 and dim00 != 1:
                return True
            return False

        def _is_vec6(shape0, shape1):
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if dim01 == 1 and dim10 == 1 and dim00 != 1 and dim11 != 1:
                return True
            return False

        def _is_vec8(shape0, shape1):
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if dim00 == dim10 and dim11 == 1 and dim01 != 1:
                return True
            return False

        def _is_vec9(shape0, shape1):
            [dim00, dim01], [dim10, dim11] = shape0, shape1
            if dim00 == 1 and dim11 == 1 and dim01 != 1 and dim10 != 1:
                return True
            return False

        def _process_range_step1(range0, range1):
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

        def _process_range_step2(range0, range1, is_cut_reduce=False, mode="copy"):
            dim00_range = range0[0]
            dim01_range = range0[1]
            dim10_range = range1[0]
            dim11_range = range1[1]
            # when mode is vec2 or vec6, dim01 is 1, range = [1, 1]
            process_dim01 = "vec2" not in mode and "vec6" not in mode

            # when mode is vec8 or vec9, dim11 is 1, range = [1, 1]
            process_dim11 = "vec8" not in mode and "vec9" not in mode
            if is_cut_reduce:
                dim01_range = (max(dim01_range[0], bound_size), dim01_range[1]) if process_dim01 else dim01_range
                dim11_range = (max(dim11_range[0], bound_size), dim11_range[1]) if process_dim11 else dim11_range
            else:
                dim01_range = (dim01_range[0], min(dim01_range[1], bound_size)) if process_dim01 else dim01_range
                dim11_range = (dim11_range[0], min(dim11_range[1], bound_size)) if process_dim11 else dim11_range

            range0 = [dim00_range, dim01_range]
            range1 = [dim10_range, dim11_range]
            return range0, range1

        self.f_ranges[0], self.f_ranges[1] = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
        [[(r00_l, r00_r), (r01_l, r01_r)], [(r10_l, r10_r), (r11_l, r11_r)]] = self.f_ranges
        tail_not_broadcast = (r01_l > 1 and r11_l > 1) or (r01_l == r01_r == 1 and r11_l == r11_r == 1)
        tail_may_broadcast = not tail_not_broadcast
        batch_not_broadcast = (r00_l > 1 and r10_l > 1) or (r00_l == r00_r == 1 and r10_l == r10_r == 1)
        batch_may_broadcast = not batch_not_broadcast
        ub_size = get_soc_spec("UB_SIZE")
        num_per_block = BLOCK_SIZE_BYTE // self.dtype_size
        bound_size = (ub_size // 4 // MAX_COEXIST_NUM) // 16 * 16
        range0, range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
        r01_r = range0[0][1]
        r11_r = range0[1][1]
        self.f_ranges = [range0, range1]
        may_cut_reduce = (r01_r >= bound_size or r11_r >= bound_size)
        not_cut_reduce = not may_cut_reduce

        if not_cut_reduce:
            if tail_not_broadcast and batch_not_broadcast:
                res = []
                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])
                return res

            if tail_not_broadcast and batch_may_broadcast:
                res = []

                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec1")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec1(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec4")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec4(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])

                return res

            if tail_may_broadcast and batch_not_broadcast:
                res = []
                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec2")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec2(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec8")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec8(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])
                return res

            if tail_may_broadcast and batch_may_broadcast:
                res = []
                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec1")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec1(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec4")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec4(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec2")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec2(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec8")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec8(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec9")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec9(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec9"),
                                gen_template(special_shape1, special_range1, "vec9")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec6")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec6(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec6"),
                                gen_template(special_shape1, special_range1, "vec6")])
                return res

        if may_cut_reduce:
            if tail_not_broadcast and batch_not_broadcast:
                res = []
                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode=COPY_AND_CUT)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY_AND_CUT),
                                gen_template(special_shape1, special_range1, COPY_AND_CUT)])

                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])
                return res

            if tail_not_broadcast and batch_may_broadcast:
                res = []
                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode=COPY_AND_CUT)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY_AND_CUT),
                                gen_template(special_shape1, special_range1, COPY_AND_CUT)])

                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec1_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec1(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec1_and_cut"),
                                gen_template(special_shape1, special_range1, "vec1_and_cut")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec1")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec1(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec4_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec4(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec4_and_cut"),
                                gen_template(special_shape1, special_range1, "vec4_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec4")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec4(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])
                return res

            if tail_may_broadcast and batch_not_broadcast:
                res = []
                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode=COPY_AND_CUT)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY_AND_CUT),
                                gen_template(special_shape1, special_range1, COPY_AND_CUT)])

                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec2_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec2(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec2_and_cut"),
                                gen_template(special_shape1, special_range1, "vec2_and_cut")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec2")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec2(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec8_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec8(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec8_and_cut"),
                                gen_template(special_shape1, special_range1, "vec8_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec8")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec8(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])
                return res

            if tail_may_broadcast and batch_may_broadcast:
                res = []
                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode=COPY_AND_CUT)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY_AND_CUT),
                                gen_template(special_shape1, special_range1, COPY_AND_CUT)])

                special_range0, special_range1 = _process_range_step1(self.f_ranges[0], self.f_ranges[1])
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode=COPY)
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                self.f_shapes[0], self.f_shapes[1], is_copy=True)
                if _is_legal_range(special_range0, special_range1) and _is_copy(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, COPY),
                                gen_template(special_shape1, special_range1, COPY)])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec1_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec1(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec1_and_cut"),
                                gen_template(special_shape1, special_range1, "vec1_and_cut")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec1")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec1(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec1"),
                                gen_template(special_shape1, special_range1, "vec1")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec4_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec4(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec4_and_cut"),
                                gen_template(special_shape1, special_range1, "vec4_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec4")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec4(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec4"),
                                gen_template(special_shape1, special_range1, "vec4")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec2_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec2(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec2_and_cut"),
                                gen_template(special_shape1, special_range1, "vec2_and_cut")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = self.f_shapes[1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec2")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec2(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec2"),
                                gen_template(special_shape1, special_range1, "vec2")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec8_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec8(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec8_and_cut"),
                                gen_template(special_shape1, special_range1, "vec8_and_cut")])

                special_shape0 = self.f_shapes[0]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec8")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec8(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec8"),
                                gen_template(special_shape1, special_range1, "vec8")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec9_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec9(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec9_and_cut"),
                                gen_template(special_shape1, special_range1, "vec9_and_cut")])

                special_shape0 = [1, self.f_shapes[0][1]]
                special_range0 = [(1, 1), _process_range_vs_2(self.f_ranges[0][1])]
                special_shape1 = [self.f_shapes[1][0], 1]
                special_range1 = [_process_range_vs_2(self.f_ranges[1][0]), (1, 1)]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec9")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec9(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec9"),
                                gen_template(special_shape1, special_range1, "vec9")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=True, mode="vec6_and_cut")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec6(special_shape0, special_shape1):
                    res.append([gen_template(special_shape0, special_range0, "vec6_and_cut"),
                                gen_template(special_shape1, special_range1, "vec6_and_cut")])

                special_shape0 = [self.f_shapes[0][0], 1]
                special_range0 = [_process_range_vs_2(self.f_ranges[0][0]), (1, 1)]
                special_shape1 = [1, self.f_shapes[1][1]]
                special_range1 = [(1, 1), _process_range_vs_2(self.f_ranges[1][1])]
                special_range0, special_range1 = _process_range_step1(special_range0, special_range1)
                special_range0, special_range1 = _process_range_step2(special_range0, special_range1,
                                                                      is_cut_reduce=False, mode="vec6")
                special_shape0, special_shape1 = _process_shape(special_range0, special_range1,
                                                                special_shape0, special_shape1, is_copy=False)
                if _is_legal_range(special_range0, special_range1) and _is_vec6(special_shape0, special_shape1):
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
