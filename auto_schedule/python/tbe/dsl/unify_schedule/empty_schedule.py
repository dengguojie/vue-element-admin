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
from .vector_schedule import VectorSchedule


class EmptySchedule(VectorSchedule):
    """
    Empty Schedule
    """
    def _calc_double_buffer(self):
        pass

    def _calc_emit_insn(self):
        for tensor in self.graph_info.tensor_list:
            if tensor not in self.graph_info.input_tensor_set:
                self.emit_insn_list.append(VectorSchedule.EmitInsnInfo(tensor, 0,
                                                                       "phony_insn"))

    def _calc_compute_at(self):
        pass

    def _calc_storage_align(self):
        pass

    def _calc_constraint(self):
        pass

    def _calc_reorder(self):
        pass

    def _calc_tiling(self):
        pass

    def _calc_storage_bound(self):
        pass

    def _calc_compute_inline(self):
        pass

    def _calc_data_flow_control(self):
        pass

    def _calc_reduced_axis_indexes(self):
        pass

    def _calc_pragma(self):
        pass
