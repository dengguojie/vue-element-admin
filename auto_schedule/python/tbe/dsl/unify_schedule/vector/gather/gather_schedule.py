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
gather schedule
"""
from typing import Any

from tbe import tvm
from tbe.common.utils import op_tiling
from tbe.dsl.base import operation
from tbe.dsl.base.operation import get_compile_info

from ... import util
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import GatherPattern
from ...constants import Pattern
from ...schedule import Schedule
from .gather_tilingcase import TilingStrategy
from .gather_tilingcase import GatherCompileInfo

DEFAULT = "default"

# block size in D architecture
BLOCK_SIZE_BYTE = 32

# STORE AREA
PARAMS_STORE_GM = 0
PARAMS_STORE_UB = 1
PARAMS_STORE_L1 = 2

PARAMS_SCOPE = {
    PARAMS_STORE_UB: "local.UB",
    PARAMS_STORE_L1: "local.L1"
}


# 'pylint: disable=R0902, R0903
class GatherSchedule(Schedule):
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
        return [GatherPattern.NORMAL_SCHEDULE]

    def __init__(self, outs, tiling_case):
        self._out_tensor = outs[0]
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._tiling_key = self._tiling_case.get("key")

        self._special_pattern = (self._tiling_key % 10000) // 1000

        self._batch_dims = self._tiling_case.get("batch_dims", 0)

        # gather axis (real axis)
        self._axis = self._tiling_case.get("axis", 0)

        # last dim gather or no last dim gather splite no last dim
        self._is_need_align = self._tiling_case.get("is_need_align", False)

        self._store_area = self._tiling_case.get("store_area", PARAMS_STORE_GM)

        # is params align
        self._is_params_align = self._tiling_case.get("is_params_align", False)

        # DB
        self._is_db = self._tiling_case.get("is_db", False)

        self._scalar_mode = self._tiling_case.get("scalar_mode", False)

        self._remove_pad = self._tiling_case.get("remove_pad", False)

        self._tensor_swell = 4 if self._remove_pad else 1

        self._compile_info_swell = operation.get_context().get("_compile_info_swells")

        self._gather_compute_type = 0

        self._scope = "local.UB"

        self._input_tensors = set()
        self._params_gm_tensor = None
        self._indices_gm_tensor = None

        self._params_name = None
        self._indices_name = None

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._tensor_space = None
        self._ub_size = util.get_ub_size()
        self._params_ub_size = self._ub_size // 2
        self._l1_size = util.get_l1_size()
        self._tmp_ub_size = 0
        self._params_dtype_size = 4
        self._indices_dtype_size = 4

        # input -> outputs mapping relations
        self._in_out_map = {}

        self._cache_write_tensor = None

        self._compute_at_map = {}

        # const tiling
        self._const_block_axis = -1
        self._const_ub_axis = 0
        self._const_block_factor = 1
        self._const_ub_factor = 1

        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._block_bind_axis = None
        self._compute_at_axis = None

        self._emit_insn_map = {}

        self._out_shape = self._out_tensor.shape
        self._out_shape_len = len(self._out_shape)

        self._indices_storage_bound = 0
        self._gather_storage_bound = 0

        self._fake_schedule = False

    def do_schedule(self):
        """
        schedule body
        :return:
        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out_tensor.op)
        self._schedule.tiling_key = self._tiling_key

        self._cal_cache_read()
        self._do_cache_read()

        self._cal_cache_write()
        self._do_cache_write()

        self._cal_storage_bound()
        self._do_storage_bound()

        self._calc_tiling()

        if self._fake_schedule:
            return None

        self._do_tiling()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_storage_align()
        self._do_storage_align()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_double_buffer()
        self._do_double_buffer()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._add_compile_info()

        return self._schedule

    def _construct_compute_graph(self):

        visited_tensors = set()

        self.__dfs_sub_graph(self._out_tensor, visited_tensors)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        # params gm and indices gm by name
        for one_input_tensor in self._input_tensors:
            if one_input_tensor.name == self._params_name:
                self._params_gm_tensor = one_input_tensor
            elif one_input_tensor.name == self._indices_name:
                self._indices_gm_tensor = one_input_tensor

        self._params_dtype_size = DTYPE_BYTE_MAPPING[self._params_gm_tensor.dtype]
        self._indices_dtype_size = DTYPE_BYTE_MAPPING[self._indices_gm_tensor.dtype]

    def _cal_storage_bound(self):
        self._coexisting_quantity_gather = int(BLOCK_SIZE_BYTE / self._indices_dtype_size)
        self._coexisting_quantity_indices = 1
        self._coexisting_quantity = self._coexisting_quantity_gather * self._tensor_swell \
                                    + self._coexisting_quantity_indices

    def _cal_cache_read(self):
        pass

    def _do_cache_read(self):
        # indcies
        self._indices_ub_tensor = self._schedule.cache_read(self._indices_gm_tensor, self._scope,
                                                            self._in_out_map[self._indices_gm_tensor])

        # params in ub or l1
        if self._store_area > 0:
            self._params_inner_tensor = self._schedule.cache_read(self._params_gm_tensor,
                                                                  PARAMS_SCOPE[self._store_area],
                                                                  self._in_out_map[self._params_gm_tensor])

    def _cal_cache_write(self):
        self._cache_write_tensor = self._out_tensor

    def _do_cache_write(self):

        if self._remove_pad:
            # remove pad add one node
            self._removd_pad_tensor = self._schedule.cache_write(self._cache_write_tensor, self._scope)
            self._gather_ub_tensor = self._schedule.cache_write(self._removd_pad_tensor, self._scope)
        else:
            self._gather_ub_tensor = self._schedule.cache_write(self._cache_write_tensor, self._scope)

    def _calc_storage_align(self):
        self._align_factor = BLOCK_SIZE_BYTE // self._params_dtype_size

    def _do_storage_bound(self):

        if self._store_area == PARAMS_STORE_UB:
            # params in ub
            self.tensor_space = (self._ub_size - self._params_ub_size) // self._coexisting_quantity \
                                // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

            # set params ub storage bound
            self._params_storage_bound = int(self._params_ub_size // self._params_dtype_size)
            self._schedule[self._params_inner_tensor].set_buffer_size(self._params_storage_bound)
        else:
            self.tensor_space = self._ub_size // self._coexisting_quantity // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

            if self._store_area == PARAMS_STORE_L1:
                # params in l1
                # set params ub storage bound
                self._params_storage_bound = int(self._l1_size // self._params_dtype_size)
                self._schedule[self._params_inner_tensor].set_buffer_size(self._params_storage_bound)
            else:
                # PARAMS_STORE_GM
                if self._is_db:
                    # db
                    self.tensor_space = self.tensor_space // 2

        # indices buffer size
        self._indices_storage_bound = int(self.tensor_space // self._indices_dtype_size)
        self._schedule[self._indices_ub_tensor].set_buffer_size(self._indices_storage_bound)

        # gather buffer size
        self._gather_storage_bound = int(
            self.tensor_space * self._coexisting_quantity_gather // self._params_dtype_size)
        self._schedule[self._gather_ub_tensor].set_buffer_size(self._gather_storage_bound)

        # remove pad
        if self._remove_pad:
            self._schedule[self._removd_pad_tensor].set_buffer_size(self._gather_storage_bound)

    def _calc_tiling(self):
        funcs = {TilingStrategy.DYNAMIC: self._calc_tiling_dynamic,
                 TilingStrategy.STATIC: self._calc_tiling_static}

        funcs[self._tiling_strategy]()

    def _calc_tiling_dynamic(self):
        res = self._out_tensor
        shape = util.shape_to_list(res.shape)
        b_i = self._tiling_case["block_tiling_axis"]
        u_i = self._tiling_case["ub_tiling_axis"]
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_bound = (1, util.get_bound(shape[u_i])[1])
        self._block_tiling_vars[b_i] = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = operation.var_inner("_ub_factor_" + str(u_i), u_bound)

    def _calc_tiling_static(self):
        tmp_output_shape = [i.value for i in self._out_tensor.shape]
        outputs = [{"shape": tmp_output_shape, "dtype": self._out_tensor.dtype}]

        tmp_params_shape = tuple(i.value for i in self._params_gm_tensor.shape)
        tmp_indices_shape = tuple(i.value for i in self._indices_gm_tensor.shape)

        inputs = [{"shape": tmp_params_shape, "dtype": self._params_gm_tensor.dtype},
                  {"shape": tmp_indices_shape, "dtype": self._indices_gm_tensor.dtype}]

        base_info = [util.get_core_num(), self._ub_size, self._l1_size, self._gather_compute_type,
                     self._params_dtype_size, self._indices_dtype_size]

        custom_info = [int(self._l1_size // self._params_dtype_size),
                       int(self._params_ub_size // self._params_dtype_size),
                       self._batch_dims, False, self._batch_dims]

        tensor_sizes = {self._special_pattern: [self._gather_storage_bound, self._indices_storage_bound]}
        if self._special_pattern != GatherCompileInfo.BASE_SCHEDULE_PATTERN:
            tensor_sizes[GatherCompileInfo.BASE_SCHEDULE_PATTERN] = \
                [self._gather_storage_bound, self._indices_storage_bound]

        const_compile_info = {
            CompileInfo.BASE_INFO: base_info,
            GatherCompileInfo.CUSTOM_INFO: custom_info,
            GatherCompileInfo.CONST_AXIS: self._axis,
            GatherCompileInfo.TENSOR_SIZES: tensor_sizes
        }
        const_compile_info.update(get_compile_info())

        op_type = "AutoTiling"
        run_info = op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)
        tiling_format = {
            "tiling_key": "int",
            "block_axis": "int",
            "block_factor": "int",
            "ub_axis": "int",
            "ub_factor": "int"}

        tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
        const_tiling_key = tiling_data["tiling_key"]
        self._const_block_axis = tiling_data["block_axis"]
        self._const_block_factor = tiling_data["block_factor"]
        self._const_ub_axis = tiling_data["ub_axis"]
        self._const_ub_factor = tiling_data["ub_factor"]

        if operation.get_context().get(GatherCompileInfo.STATIC_SUCCESS) or const_tiling_key != self._tiling_key:
            operation.get_context().get_current_compute().get_current_schedule()\
                .add(GatherCompileInfo.FAKE_SCHEDULE, True)
            self._fake_schedule = True
        else:
            operation.get_context().add(GatherCompileInfo.STATIC_SUCCESS, True)

    def _do_tiling(self):
        funcs = {TilingStrategy.DYNAMIC: self._do_tiling_dynamic,
                 TilingStrategy.STATIC: self._do_tiling_static, }
        funcs[self._tiling_strategy]()

    def _do_tiling_dynamic(self):
        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]
        b_o, b_i = self._schedule[self._out_tensor].split(self._out_tensor.op.axis[b_idx],
                                                          factor=self._block_tiling_vars[b_idx])

        if b_idx == u_idx:
            u_o, u_i = self._schedule[self._out_tensor].split(b_i, factor=self._ub_tiling_vars[u_idx])
        else:
            u_o, u_i = self._schedule[self._out_tensor].split(self._out_tensor.op.axis[u_idx],
                                                              factor=self._ub_tiling_vars[u_idx])

        self._block_bind_axis = b_o
        self._compute_at_axis = u_o
        self._gather_emit_at_axis = self._gather_ub_tensor.op.axis[-1]

        # gather align
        if self._is_need_align:
            self._gather_align_axis = self._gather_ub_tensor.op.axis[-2]

        # remove pad
        if self._remove_pad:
            self._remove_pad_emit_at_axis = self._removd_pad_tensor.op.axis[0]

        # res emit
        self._res_emit_at_axis = u_i

    def _do_tiling_static(self):
        b_idx = self._const_block_axis
        u_idx = self._const_ub_axis

        b_o, b_i = self._schedule[self._out_tensor].split(self._out_tensor.op.axis[b_idx],
                                                          factor=self._const_block_factor)

        if b_idx == u_idx:
            u_o, u_i = self._schedule[self._out_tensor].split(b_i, factor=self._const_ub_factor)
        else:
            u_o, u_i = self._schedule[self._out_tensor].split(self._out_tensor.op.axis[u_idx],
                                                              factor=self._const_ub_factor)
        self._block_bind_axis = b_o
        self._compute_at_axis = u_o
        self._gather_emit_at_axis = self._gather_ub_tensor.op.axis[-1]

        # gather align
        if self._is_need_align:
            self._gather_align_axis = self._gather_ub_tensor.op.axis[-2]

        # remove pad handle
        if self._remove_pad:
            self._remove_pad_emit_at_axis = self._removd_pad_tensor.op.axis[0]

        # res emit
        self._res_emit_at_axis = u_i

    def _calc_multi_core(self):
        pass

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out_tensor].bind(self._block_bind_axis, block)

    def _do_storage_align(self):
        if self._is_need_align:
            self._schedule[self._gather_ub_tensor].storage_align(self._gather_align_axis, self._align_factor, 0)

        if self._is_params_align:
            self._schedule[self._params_inner_tensor].storage_align(self._params_inner_tensor.op.axis[-2],
                                                                    self._align_factor, 0)

    def _calc_compute_at(self):
        # params indcies inputs
        for tensor_i in self._input_tensors:
            self._compute_at_map[tensor_i] = [self._out_tensor, self._compute_at_axis]

        # params ub/l1
        if self._store_area > 0:
            self._compute_at_map[self._params_inner_tensor] = [self._out_tensor, self._block_bind_axis]

        # indices ub
        self._compute_at_map[self._indices_ub_tensor] = [self._out_tensor, self._compute_at_axis]

        # gather ub
        self._compute_at_map[self._gather_ub_tensor] = [self._out_tensor, self._compute_at_axis]

        # remove pad
        if self._remove_pad:
            self._compute_at_map[self._removd_pad_tensor] = [self._out_tensor, self._compute_at_axis]

    def _do_compute_at(self):
        for tensor_i, param in self._compute_at_map.items():
            self._schedule[tensor_i].compute_at(self._schedule[param[0]], param[1])

    def _calc_double_buffer(self):
        pass

    def _do_double_buffer(self):
        if self._is_db:
            self._schedule[self._indices_ub_tensor].double_buffer()
            self._schedule[self._gather_ub_tensor].double_buffer()

    def _calc_emit_insn(self):

        # indcies ub
        self._emit_insn_map[self._indices_ub_tensor] = [self._indices_ub_tensor.op.axis[0], "dma_copy"]

        # params_ub/params_l1 need
        if self._store_area == 1:
            self._emit_insn_map[self._params_inner_tensor] = [self._params_inner_tensor.op.axis[0], "dma_copy"]
        elif self._store_area == 2:
            self._emit_insn_map[self._params_inner_tensor] = [self._params_inner_tensor.op.axis[0], "dma_copy",
                                                              dict(mem_align=1)]

        # gather ub
        if self._store_area == 1:
            if self._scalar_mode:
                self._emit_insn_map[self._gather_ub_tensor] = [self._gather_emit_at_axis, "data_mov"]
            else:
                self._emit_insn_map[self._gather_ub_tensor] = [self._gather_emit_at_axis, "dma_copy"]
        elif self._store_area == 2:
            self._emit_insn_map[self._gather_ub_tensor] = [self._gather_emit_at_axis, "dma_copy", dict(mem_align=1)]
        else:
            self._emit_insn_map[self._gather_ub_tensor] = [self._gather_emit_at_axis, "dma_copy"]

        # remove pad
        if self._remove_pad:
            self._emit_insn_map[self._removd_pad_tensor] = [self._remove_pad_emit_at_axis, "remove_pad"]

        # res
        if self._is_db:
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy", dict(no_overlap=0)]
        else:
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy", dict(no_overlap=2)]

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
        cpt_schedule.add(GatherCompileInfo.INDICES_DTYPE_SIZE, self._indices_dtype_size)

        # CUSTOM INFO
        cpt_schedule.add(GatherCompileInfo.PARAMS_L1_NUM, int(self._l1_size // self._params_dtype_size))
        cpt_schedule.add(GatherCompileInfo.PARAMS_UB_NUM, int(self._params_ub_size // self._params_dtype_size))
        cpt_schedule.add(GatherCompileInfo.BATCH_DIMS, self._batch_dims)
        cpt_schedule.add(GatherCompileInfo.SPECIAL_PATTERN, self._special_pattern)
        cpt_schedule.add(GatherCompileInfo.PARAMS_NUM, self._gather_storage_bound)
        cpt_schedule.add(GatherCompileInfo.INDICES_NUM, self._indices_storage_bound)

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
