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
slice schedule
"""
from typing import Any

from tbe import tvm
from tbe.common.utils import op_tiling
from tbe.dsl.base import operation
from tbe.dsl.base.operation import get_compile_info

from ...import util
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import SlicePattern
from ...constants import Pattern
from ...schedule import Schedule
from .slice_tilingcase import TilingStrategy
from .slice_tilingcase import SliceCompileInfo
from .slice_tilingcase import DATA_MOV
from .slice_tilingcase import DEPAD
from .slice_tilingcase import UNALIGED_STRIDE
from .slice_tilingcase import BOTH_ALIGN
from .slice_tilingcase import ONE_DIM
from .slice_tilingcase import SCALAR
from .slice_tilingcase import LR_DEPAD
from .slice_tilingcase import DEFAULT

TILING_MODE_MAP = {
    1: DATA_MOV,
    2: DEPAD,
    3: UNALIGED_STRIDE,
    4: BOTH_ALIGN,
    5: ONE_DIM,
    6: SCALAR,
    7: LR_DEPAD
}

# block size in D architecture
BLOCK_SIZE_BYTE = 32

# reserve space
RESERVE_SPACE = 1024


class SliceSchedule(Schedule):
    """
    slice schedule
    """
    def __init__(self, outs, tiling_case):
        self._real_out = outs[0]
        self._out_tensor = self._real_out
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._tiling_key = self._tiling_case.get("key")
        self._tiling_mode = self._tiling_case.get("mode")

        self._scope = "local.UB"

        self._input_tensors = []
        self._out_dtype_bytes = DTYPE_BYTE_MAPPING.get(self._real_out.dtype)
        self._coexisting_quantity = 1
        self._tensor_space = None
        self._ub_size = util.get_ub_size() - RESERVE_SPACE

        self._cache_read_tensor = None
        self._cache_write_tensor = None

        self._compute_at_map = {}

        # const tiling
        self._const_block_axis = 0
        self._const_ub_axis = 0
        self._const_block_factor = 1
        self._const_ub_factor = 1
        self._const_block_dims = 1

        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._block_bind_axis = None
        self._compute_at_axis = None

        self._emit_insn_map = {}

        self._out_shape = self._real_out.shape
        self._out_shape_len = len(self._out_shape)

        self._tensor_storage_bound = None
        self._align_factor = None

        self._slice_ub_tensor = None
        self._input_x_tensor = None
        self._remove_pad_tensor = None
        self._remove_pad_emit_at_axis = None
        self._slice_ub_emit_at_axis = None
        self._cache_read_emit_at_axis = None
        self._res_emit_at_axis = None
        self._real_out_emit_at_axis = None

    @classmethod
    def get_instance(cls, outs, tiling_case):  # type: (list[Any], Any) -> "Schedule"
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):  # type: () -> list[str]
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):  # type: () -> list[str]
        return [Pattern.SLICE]

    @classmethod
    def get_supported_sub_pattern(cls):  # type: () -> list[str]
        return [SlicePattern.NORMAL_SCHEDULE]

    def do_schedule(self):
        """
        schedule body
        :return:
        """
        self._construct_compute_graph()

        self._calc_storage_bound()

        self._calc_tiling()

        self._add_fake_node()

        self._schedule = tvm.create_schedule(self._out_tensor.op)
        self._schedule.tiling_key = self._tiling_key

        self._calc_cache_read()
        self._calc_cache_write()

        self._do_cache_read()
        self._do_cache_write()

        self._do_tiling()

        self._do_double_buffer()

        self._do_storage_bound()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_storage_align()
        self._do_storage_align()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._set_constraint()

        self._add_compile_info()

        return self._schedule

    def _construct_compute_graph(self):

        visited_tensors = set()

        self.__dfs_sub_graph(self._real_out, visited_tensors)

    def _add_fake_node(self):
        if self._tiling_mode == LR_DEPAD:
            self._out_tensor = self._fake_node()

    def _fake_node(self):
        return tvm.compute(self._input_tensors[0].shape,
                           lambda *i: self._input_tensors[0][i] + self._real_out[i], name="slice_fake")

    def _calc_storage_align(self):
        self._align_factor = BLOCK_SIZE_BYTE // self._out_dtype_bytes

    def _calc_storage_bound(self):
        operation.add_compile_info_inner("_coex_list", [2, 4, 3])

    def _calc_tiling(self):
        funcs = {TilingStrategy.DYNAMIC: self._calc_tiling_dynamic,
                 TilingStrategy.STATIC: self._calc_tiling_static,
                 TilingStrategy.ZEROS: self._calc_tiling_zeros}

        funcs.get(self._tiling_strategy)()

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
        tmp_output_shape = [i.value for i in self._real_out.shape]
        outputs = [{"shape": tmp_output_shape, "dtype": self._real_out.dtype}]

        tmp_x_shape = tuple(i.value for i in self._input_tensors[0].shape)
        inputs = [{"shape": tmp_x_shape, "dtype": self._real_out.dtype}]

        base_info = [util.get_core_num(), self._ub_size, self._out_dtype_bytes]
        const_begins = operation.get_context().get("_const_begins")
        const_sizes = tmp_output_shape

        const_compile_info = {
            CompileInfo.BASE_INFO: base_info,
            SliceCompileInfo.CONST_BEGINS: const_begins,
            SliceCompileInfo.CONST_SIZES: const_sizes,
            SliceCompileInfo.IS_STATIC: True,
            SliceCompileInfo.END_MODE: 0,
        }

        const_compile_info.update(get_compile_info())
        op_type = "AutoTiling"
        run_info = op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)
        tiling_format = {
            "tiling_key": "int",
            "block_axis": "int",
            "block_factor": "int",
            "ub_axis": "int",
            "ub_factor": "int",
            "tiling_mode": "int",
            "block_dims": "int"
        }

        tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
        self._const_block_axis = tiling_data.get("block_axis")
        self._const_block_factor = tiling_data.get("block_factor")
        self._const_ub_axis = tiling_data.get("ub_axis")
        self._const_ub_factor = tiling_data.get("ub_factor")
        self._tiling_mode = TILING_MODE_MAP.get(tiling_data.get("tiling_mode"))
        self._const_block_dims = tiling_data.get("block_dims")

    def _calc_tiling_zeros(self):
        pass

    def _do_tiling(self):
        funcs = {TilingStrategy.DYNAMIC: self._do_tiling_dynamic,
                 TilingStrategy.STATIC: self._do_tiling_static,
                 TilingStrategy.ZEROS: self._do_tiling_zeros}

        funcs.get(self._tiling_strategy)()

    def _do_tiling_dynamic(self):
        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]
        b_o, b_i = self._schedule[self._out_tensor].split(self._out_tensor.op.axis[b_idx],
                                                          factor=self._block_tiling_vars.get(b_idx))

        if b_idx == u_idx:
            u_o, u_i = self._schedule[self._out_tensor].split(b_i, factor=self._ub_tiling_vars.get(u_idx))
        else:
            u_o, u_i = self._schedule[self._out_tensor].split(self._out_tensor.op.axis[u_idx],
                                                              factor=self._ub_tiling_vars.get(u_idx))

        self._block_bind_axis = b_o
        self._compute_at_axis = u_o

        # emit axis
        self._slice_ub_emit_at_axis = self._slice_ub_tensor.op.axis[u_idx]

        if self._tiling_mode in [DEPAD, SCALAR]:
            self._remove_pad_emit_at_axis = self._remove_pad_tensor.op.axis[u_idx]

        if self._tiling_mode == LR_DEPAD:
            self._cache_read_emit_at_axis = self._cache_read_tensor.op.axis[u_idx]
            self._real_out_emit_at_axis = self._real_out.op.axis[u_idx]

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

        # emit axis
        self._slice_ub_emit_at_axis = self._slice_ub_tensor.op.axis[u_idx]

        if self._tiling_mode in [DEPAD, SCALAR]:
            self._remove_pad_emit_at_axis = self._remove_pad_tensor.op.axis[u_idx]

        if self._tiling_mode == LR_DEPAD:
            self._cache_read_emit_at_axis = self._cache_read_tensor.op.axis[u_idx]
            self._real_out_emit_at_axis = self._real_out.op.axis[u_idx]

        self._res_emit_at_axis = u_i

    def _do_tiling_zeros(self):
        self._compute_at_axis = self._out_tensor.op.axis[0]
        self._slice_ub_emit_at_axis = self._slice_ub_tensor.op.axis[0]
        self._res_emit_at_axis = self._out_tensor.op.axis[0]

    def _calc_cache_read(self):
        self._input_x_tensor = self._input_tensors[0]

    def _do_cache_read(self):
        if self._tiling_mode == LR_DEPAD:
            self._cache_read_tensor = self._schedule.cache_read(self._input_x_tensor,
                                                                self._scope, [self._real_out, self._out_tensor])

    def _calc_cache_write(self):
        if self._tiling_mode == LR_DEPAD:
            self._cache_write_tensor = self._real_out
        else:
            self._cache_write_tensor = self._out_tensor

    def _do_cache_write(self):
        if self._tiling_mode in [DEPAD, SCALAR]:
            # add one node
            self._remove_pad_tensor = self._schedule.cache_write(self._cache_write_tensor, self._scope)
            self._slice_ub_tensor = self._schedule.cache_write(self._remove_pad_tensor, self._scope)
        else:
            self._slice_ub_tensor = self._schedule.cache_write(self._cache_write_tensor, self._scope)

    def _do_storage_bound(self):
        if self._tiling_mode == DATA_MOV:
            self._coexisting_quantity = 1
        if self._tiling_mode in [ONE_DIM, BOTH_ALIGN]:
            self._coexisting_quantity = 2
        if self._tiling_mode in [DEPAD, SCALAR, LR_DEPAD]:
            self._coexisting_quantity = 4
        if self._tiling_mode == UNALIGED_STRIDE:
            self._coexisting_quantity = 3

        self._tensor_space = self._ub_size // self._coexisting_quantity // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE
        self._tensor_storage_bound = int(self._tensor_space // self._out_dtype_bytes)
        if self._tiling_mode == LR_DEPAD:
            self._schedule[self._cache_read_tensor].set_buffer_size(self._tensor_storage_bound)
        self._schedule[self._slice_ub_tensor].set_buffer_size(self._tensor_storage_bound)
        if self._tiling_mode in [DEPAD, SCALAR]:
            self._schedule[self._remove_pad_tensor].set_buffer_size(self._tensor_storage_bound)

    def _calc_multi_core(self):
        pass

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out_tensor].bind(self._block_bind_axis, block)

    def _do_storage_align(self):
        if len(self._slice_ub_tensor.op.axis) >= 2:
            if self._tiling_mode in [DATA_MOV, DEPAD, SCALAR, UNALIGED_STRIDE]:
                self._schedule[self._slice_ub_tensor].storage_align(self._slice_ub_tensor.op.axis[-2],
                                                                    self._align_factor, 0)

    def _calc_compute_at(self):
        # slice ub
        self._compute_at_map[self._slice_ub_tensor] = [self._out_tensor, self._compute_at_axis]

        if self._tiling_mode in [DEPAD, SCALAR]:
            self._compute_at_map[self._remove_pad_tensor] = [self._out_tensor, self._compute_at_axis]

        if self._tiling_mode == LR_DEPAD:
            self._compute_at_map[self._cache_read_tensor] = [self._out_tensor, self._compute_at_axis]
            self._compute_at_map[self._real_out] = [self._out_tensor, self._compute_at_axis]

    def _do_compute_at(self):
        for tensor_i, param in self._compute_at_map.items():
            self._schedule[tensor_i].compute_at(self._schedule[param[0]], param[1])

    def _calc_emit_insn(self):
        if self._tiling_mode == DEPAD:
            self._emit_insn_map[self._slice_ub_tensor] = [self._slice_ub_emit_at_axis, "dma_copy",
                                                          {"gm_to_ub_gap_opt": 1}]
            self._emit_insn_map[self._remove_pad_tensor] = [self._remove_pad_emit_at_axis, "remove_pad"]
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy"]
        elif self._tiling_mode == UNALIGED_STRIDE:
            self._emit_insn_map[self._slice_ub_tensor] = [self._slice_ub_emit_at_axis, "dma_copy",
                                                          {"gm_to_ub_gap_opt": 1}]
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy", {"no_overlap": 3}]
        elif self._tiling_mode in [BOTH_ALIGN, ONE_DIM]:
            self._emit_insn_map[self._slice_ub_tensor] = [self._slice_ub_emit_at_axis, "dma_copy"]
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy", {"no_overlap": 0}]
        elif self._tiling_mode == SCALAR:
            self._emit_insn_map[self._slice_ub_tensor] = [self._slice_ub_emit_at_axis, "dma_copy",
                                                          {"gm_to_ub_gap_opt": 1}]
            self._emit_insn_map[self._remove_pad_tensor] = [self._remove_pad_emit_at_axis, "data_mov"]
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy"]
        elif self._tiling_mode == LR_DEPAD:
            self._emit_insn_map[self._cache_read_tensor] = [self._cache_read_emit_at_axis, "dma_copy"]
            self._emit_insn_map[self._slice_ub_tensor] = [self._slice_ub_emit_at_axis, "remove_pad"]
            self._emit_insn_map[self._real_out] = [self._real_out_emit_at_axis, "dma_copy"]
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "phony_insn"]
        else:
            self._emit_insn_map[self._slice_ub_tensor] = [self._slice_ub_emit_at_axis, "dma_copy",
                                                          {"gm_to_ub_gap_opt": 1}]
            self._emit_insn_map[self._out_tensor] = [self._res_emit_at_axis, "dma_copy"]

    def _do_emit_insn(self):
        for tensor_i, param in self._emit_insn_map.items():
            self._schedule[tensor_i].emit_insn(*param)

    def _set_constraint(self):
        if self._tiling_mode == LR_DEPAD and self._tiling_strategy == TilingStrategy.DYNAMIC:
            cond = (self._input_x_tensor.shape[self._out_shape_len - 1] >= self._out_shape[self._out_shape_len - 1])
            self._schedule.set_constraint(cond)

    def _do_double_buffer(self):
        if self._tiling_mode in [ONE_DIM, BOTH_ALIGN]:
            self._schedule[self._slice_ub_tensor].double_buffer()

    def _add_compile_info(self):
        cpt_compute = operation.get_context().get_current_compute()
        cpt_schedule = cpt_compute.get_current_schedule()

        # BASE INFO
        cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())
        cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
        cpt_schedule.add(SliceCompileInfo.X_DTYPE_SIZE, self._out_dtype_bytes)

        # dynamic static
        is_const = operation.get_context().get(SliceCompileInfo.IS_CONST)
        if is_const:
            operation.get_context().add(SliceCompileInfo.CONST_INFO,
                                        [self._tiling_key, self._const_block_dims])

    def __dfs_sub_graph(self, out, visited_tensors: set):
        for tensor_i in out.op.input_tensors:
            if util.is_placeholder(tensor_i):
                self._input_tensors.append(tensor_i)

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self.__dfs_sub_graph(tensor_i, visited_tensors)
