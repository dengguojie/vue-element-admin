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
Abstract skeleton class for Vector Schedule
"""

# Standard Packages
from abc import ABC
from abc import abstractmethod
from typing import Optional

from tbe.tvm.schedule import Schedule

from .vector_tilingcase import TilingCaseBase


class VectorScheduleBase(ABC):

    def __init__(self):
        self.schedule: Optional[Schedule] = None
        self.tiling_case = None

    def do_schedule(self, tiling_case) -> Schedule:
        self.tiling_case = tiling_case
        self._do_create_schedule()
        self._calc_schedule_generation_strategy()
        self._do_schedule_generation()
        return self.schedule

    def _calc_schedule_generation_strategy(self):
        self._calc_reduced_axis_indexes()
        self._calc_data_flow_control()
        self._calc_compute_inline()
        self._calc_tiling()
        self._calc_storage_bound()
        self._calc_reorder()
        self._calc_constraint()
        self._calc_storage_align()
        self._calc_compute_at()
        self._calc_emit_insn()
        self._calc_pragma()
        # DB is currently unavailable
        self._calc_double_buffer()

    def _do_schedule_generation(self):
        self._do_data_flow_control()
        self._do_compute_inline()
        self._do_tiling()
        self._do_storage_bound()
        self._do_reorder()
        self._do_constraint()
        self._do_storage_align()
        self._do_compute_at()
        self._do_emit_insn()
        self._do_pragma()
        # DB is currently unavailable
        self._do_double_buffer()

    @abstractmethod
    def _do_create_schedule(self):
        """"""

    @abstractmethod
    def _calc_reduced_axis_indexes(self):
        """"""

    @abstractmethod
    def _calc_data_flow_control(self):
        """"""

    @abstractmethod
    def _calc_compute_inline(self):
        """"""

    @abstractmethod
    def _calc_storage_bound(self):
        """"""

    @abstractmethod
    def _calc_tiling(self):
        """"""

    @abstractmethod
    def _calc_reorder(self):
        """"""

    @abstractmethod
    def _calc_constraint(self):
        """"""

    @abstractmethod
    def _calc_storage_align(self):
        """"""

    @abstractmethod
    def _calc_compute_at(self):
        """"""

    @abstractmethod
    def _calc_emit_insn(self):
        """"""

    @abstractmethod
    def _calc_pragma(self):
        """"""

    @abstractmethod
    def _calc_double_buffer(self):
        """"""

    @abstractmethod
    def _do_data_flow_control(self):
        """"""

    @abstractmethod
    def _do_compute_inline(self):
        """"""

    @abstractmethod
    def _do_storage_bound(self):
        """"""

    @abstractmethod
    def _do_tiling(self):
        """"""

    @abstractmethod
    def _do_reorder(self):
        """"""

    @abstractmethod
    def _do_constraint(self):
        """"""

    @abstractmethod
    def _do_storage_align(self):
        """"""

    @abstractmethod
    def _do_compute_at(self):
        """"""

    @abstractmethod
    def _do_emit_insn(self):
        """"""

    @abstractmethod
    def _do_pragma(self):
        """"""

    @abstractmethod
    def _do_double_buffer(self):
        """"""
