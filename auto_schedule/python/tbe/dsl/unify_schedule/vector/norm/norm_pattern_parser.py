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
norm pattern parser
"""
from typing import Dict
from typing import List
from typing import Union

from tbe.dsl.unify_schedule import util
from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.pattern_manager import PatternParser
from tbe.tvm.tensor import Tensor


class NormPatternParser(PatternParser):
    def __init__(self, outs, compute_type_size_map, compute_type_tensor_map):
        # type: (Union[Tensor, List[Tensor]], Dict[ComputeType, int], Dict[ComputeType, List[Tensor]]) -> None
        super().__init__(outs, compute_type_size_map, compute_type_tensor_map)

    def match(self):
        """
        check whether compute graph matches the current pattern
        """

        # norm
        # 1. support multi outs but out shape is before reduce shape or after reduce shape
        # 2. exist reduce and broadcast at the same time
        # 3. the axis of all reduce is same
        # 4. before reduce shape is equal to after broadcast shape

        def __judge_tvm_shape_equal(_shape_a, _shape_b):
            _length_a = len(_shape_a)
            _length_b = len(_shape_b)
            if _length_a != _length_b:
                return False
            for _idx in range(_length_a):
                if not util.expr_equal(_shape_a[_idx], _shape_b[_idx]):
                    return False

            return True

        def __judge_legal_output_shape(output_shape):
            return True if __judge_tvm_shape_equal(before_reduce_shape, output_shape) or \
                           __judge_tvm_shape_equal(after_reduce_shape, output_shape) else False

        placeholder_size = self.compute_type_size_map.get(ComputeType.PLACEHOLDER, 0)
        elewise_size = self.compute_type_size_map.get(ComputeType.ELEWISE, 0)
        broadcast_size = self.compute_type_size_map.get(ComputeType.BROADCAST, 0)
        cast_size = self.compute_type_size_map.get(ComputeType.CAST, 0)
        reduce_size = self.compute_type_size_map.get(ComputeType.REDUCE, 0)
        set_value_size = self.compute_type_size_map.get(ComputeType.SET_VALUE, 0)
        total = self.compute_type_size_map.get(ComputeType.ANY, 0)

        illegal_type_size = (reduce_size == 0 or broadcast_size == 0) or \
                            (placeholder_size + elewise_size + reduce_size + cast_size +
                             broadcast_size + set_value_size != total)
        if illegal_type_size:
            return False

        reduce_tensor_list = self.compute_type_tensor_map[ComputeType.REDUCE]
        broadcast_tensor_list = self.compute_type_tensor_map[ComputeType.BROADCAST]

        before_reduce_shape = reduce_tensor_list[0].op.input_tensors[0].shape
        after_reduce_shape = reduce_tensor_list[0].shape
        after_broadcast_shape = broadcast_tensor_list[0].shape

        if isinstance(self.outs, (list, tuple)):
            for single_out in self.outs:
                if not __judge_legal_output_shape(single_out.shape):
                    return False
        else:
            if not __judge_legal_output_shape(self.outs.shape):
                return False

        if not __judge_tvm_shape_equal(before_reduce_shape, after_broadcast_shape):
            return False

        if reduce_size > 1:
            for i in range(1, reduce_size):
                illegal_condition = \
                    not (__judge_tvm_shape_equal(before_reduce_shape,
                                                 reduce_tensor_list[i].op.input_tensors[0].shape) and
                         __judge_tvm_shape_equal(after_reduce_shape, reduce_tensor_list[i].shape))
                if illegal_condition:
                    return False

        if broadcast_size > 1:
            for i in range(1, broadcast_size):
                if not __judge_tvm_shape_equal(after_broadcast_shape, broadcast_tensor_list[i].shape):
                    return False

        return True

    def get_pattern(self):
        """
        return the current pattern
        """
        return Pattern.NORM
