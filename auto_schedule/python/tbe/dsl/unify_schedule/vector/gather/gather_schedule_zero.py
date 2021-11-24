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
gather schedule zero
"""
from typing import Optional

from tbe import tvm
from tbe.dsl.base import operation

from ... import util
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import GatherPattern
from ...constants import Pattern
from ...schedule import Schedule
from .gather_tilingcase import GatherCompileInfo

DEFAULT = "default"

# block size in D architecture
BLOCK_SIZE_BYTE = 32


# 'pylint: disable=R0902, R0903
class GatherScheduleZeroShape(Schedule):
    """
    gather schedule
    """

    @classmethod
    def get_instance(cls, outs, tiling_case):  # type: (list[Any], Any) -> "Schedule"
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):  # type: () -> list[str]
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):  # type: () -> list[str]
        return [Pattern.GATHER]

    @classmethod
    def get_supported_sub_pattern(cls):  # type: () -> list[str]
        return [GatherPattern.ZERO_SCHEDULE]

    def __init__(self, outs, tiling_case):
        self._out_tensor = outs[0]
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._tiling_key = self._tiling_case.get("key")

        self._dtypes = set()

        # input -> outputs mapping relations
        self._in_out_map = {}

        self._input_tensors = set()

        self._params_dtype_size = 8

        self._emit_insn_map = {}

        self._compute_at_map = {}

        self._ub_size = util.get_ub_size()

        self._l1_size = util.get_l1_size()

        self._scope = "local.UB"

        self._gather_compute_type = 0

    def do_schedule(self):
        """
        schedule body
        :return:
        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out_tensor.op)
        self._schedule.tiling_key = self._tiling_key

        self._cal_cache_write()
        self._do_cache_write()

        self._cal_storage_bound()
        self._do_storage_bound()

        self._calc_tiling()
        self._do_tiling()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._add_compile_info()

        return self._schedule

    def _construct_compute_graph(self):

        visited_tensors = set()

        self.__dfs_sub_graph(self._out_tensor, visited_tensors)
        self._max_dtype_bytes = max(DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes)

        # params gm and indices gm by name
        for one_input_tensor in self._input_tensors:
            if one_input_tensor.name == self._params_name:
                self._params_gm_tensor = one_input_tensor
            elif one_input_tensor.name == self._indices_name:
                self._indices_gm_tensor = one_input_tensor

        self._params_dtype_size = DTYPE_BYTE_MAPPING[self._params_gm_tensor.dtype]

    def _cal_storage_bound(self):
        pass

    def _cal_cache_write(self):
        self._cache_write_tensor = self._out_tensor

    def _do_cache_write(self):
        self._gather_ub_tensor = self._schedule.cache_write(self._cache_write_tensor, self._scope)

    def _do_storage_bound(self):
        # gather buffer size
        self._gather_storage_bound = int(self._ub_size / self._params_dtype_size)
        self._schedule[self._gather_ub_tensor].set_buffer_size(self._gather_storage_bound)

    def _calc_tiling(self):
        pass

    def _do_tiling(self):
        u_o, u_i = self._schedule[self._out_tensor].split(self._out_tensor.op.axis[-1],
                                                          factor=self._gather_storage_bound)
        self._compute_at_axis = u_o
        self._gather_emit_at_axis = self._gather_ub_tensor.op.axis[-2]
        # res emit
        self._res_emit_at_axis = u_i

    def _calc_compute_at(self):
        # params indcies inputs
        for tensor_i in self._input_tensors:
            self._compute_at_map[tensor_i] = [self._out_tensor, self._compute_at_axis]

        # gather ub
        self._compute_at_map[self._gather_ub_tensor] = [self._out_tensor, self._compute_at_axis]

    def _do_compute_at(self):
        for tensor_i, param in self._compute_at_map.items():
            self._schedule[tensor_i].compute_at(self._schedule[param[0]], param[1])

    def _calc_emit_insn(self):
        self._emit_insn_map[self._gather_ub_tensor] = [self._gather_emit_at_axis, "dma_copy"]
        self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy"]

    def _do_emit_insn(self):
        for tensor_i, param in self._emit_insn_map.items():
            self._schedule[tensor_i].emit_insn(*param)

    def _add_compile_info(self):
        cpt_compute = operation.get_context().get_current_compute()
        cpt_schedule = cpt_compute.get_current_schedule()

        cpt_schedule.add(GatherCompileInfo.FAKE_SCHEDULE, False)

        # BASE INFO
        cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())
        cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
        cpt_schedule.add(GatherCompileInfo.L1_SIZE, self._l1_size)
        cpt_schedule.add(GatherCompileInfo.GATHER_TYPE, self._gather_compute_type)
        cpt_schedule.add(GatherCompileInfo.PARAMS_DTYPE_SIZE, self._params_dtype_size)
        cpt_schedule.add(GatherCompileInfo.INDICES_DTYPE_SIZE, 0)

        # CUSTOM INFO
        cpt_schedule.add(GatherCompileInfo.PARAMS_NUM, 0)
        cpt_schedule.add(GatherCompileInfo.INDICES_NUM, 0)
        cpt_schedule.add(GatherCompileInfo.PARAMS_L1_NUM, int(self._l1_size // self._params_dtype_size))
        cpt_schedule.add(GatherCompileInfo.PARAMS_UB_NUM, self._ub_size)
        cpt_schedule.add(GatherCompileInfo.SPECIAL_PATTERN, GatherCompileInfo.ZERO_SCHEDULE_PATTERN)
        cpt_schedule.add(GatherCompileInfo.BATCH_DIMS, 0)

    def __dfs_sub_graph(self, out, visited_tensors: set):

        if len(out.op.attrs) > 0:
            _gather_op_name = operation.get_context().get("_gather_mode")
            self._gather_compute_type = 0 if _gather_op_name == "gather" else 1
            if _gather_op_name in ["gather", "gather_nd"]:
                if "params_name" in out.op.attrs:
                    self._params_name = out.op.attrs["params_name"]

                if "indices_name" in out.op.attrs:
                    self._indices_name = out.op.attrs["indices_name"]

        for tensor_i in out.op.input_tensors:
            util.merge_value(self._in_out_map, tensor_i, out)
            self._dtypes.add(tensor_i.dtype)

            if util.is_placeholder(tensor_i):
                self._input_tensors.add(tensor_i)

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self.__dfs_sub_graph(tensor_i, visited_tensors)
