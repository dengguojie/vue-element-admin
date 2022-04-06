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
tuple reduce pattern parser
"""
from typing import Dict
from typing import List
from typing import Set
from typing import Union

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.pattern_manager import PatternParser
from tbe.tvm.tensor import Tensor


class TupleReducePatternParser(PatternParser):
    def __init__(self, outs, compute_type_size_map, compute_type_tensor_map):
        # type: (Union[Tensor, List[Tensor]], Dict[ComputeType, int], Dict[ComputeType, List[Tensor]]) -> None
        super().__init__(outs, compute_type_size_map, compute_type_tensor_map)

    @staticmethod
    def _poset(_tensor):
        # type: (Tensor) -> Set[Tensor]
        tensors = set(_tensor.op.input_tensors)
        queue = set(_tensor.op.input_tensors)
        while queue:
            tensor = queue.pop()
            tensors.update(tensor.op.input_tensors)
            queue.update(tensor.op.input_tensors)
        return tensors

    def match(self):
        """
        check whether compute graph matches the current pattern
        """
        if ComputeType.REDUCE not in self.compute_type_tensor_map:
            return False
        reduce_tensors = self.compute_type_tensor_map.get(ComputeType.REDUCE)
        # check reduce tensors' shape
        reduce_tensor_shape = [tensor.shape for tensor in reduce_tensors]
        reduce_tensor_shape = set([tuple(map(str, _shape)) for _shape in reduce_tensor_shape])
        if len(reduce_tensor_shape) > 1:
            return False
        # check reduce axis
        reduce_axis = [axis.var for axis in reduce_tensors[0].op.body[0].axis]
        for tensor in reduce_tensors:
            axes = [axis.var for axis in tensor.op.body[0].axis]
            if not reduce_axis == axes:
                return False
        # check intermediate output, res tensor should not be any tensor's input tensors
        for tensor in self.compute_type_tensor_map[ComputeType.ANY]:
            if set(tensor.op.input_tensors).intersection(set(self.outs)):
                return False
        # check reduce node cannot be followed by any broadcast node
        # check broadcast node cannot be followed by any broadcast node
        if ComputeType.BROADCAST not in self.compute_type_tensor_map:
            return True
        broadcast_tensors = self.compute_type_tensor_map[ComputeType.BROADCAST]
        reduce_broadcast_tensor_set: Set = set(reduce_tensors).union(set(broadcast_tensors))
        for tensor in broadcast_tensors:
            if reduce_broadcast_tensor_set.intersection(self._poset(tensor)):
                return False

        return True

    def get_pattern(self):
        """
        return the current pattern
        """
        return Pattern.TUPLE_REDUCE
