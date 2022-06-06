#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
concat schedule
"""
from typing import Optional
from typing import List

from tbe import tvm
from tbe.dsl.base import operation
from tbe.common.utils import op_tiling
from tbe.dsl.base.operation import get_compile_info

from tbe.dsl.unify_schedule import util
from tbe.dsl.unify_schedule.schedule import Schedule
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.constants import CompileInfo
from tbe.dsl.unify_schedule.constants import ConcatPattern
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from .concat_tilingcase import TilingStrategy
from .concat_tilingcase import ConcatTilingCase

DEFAULT = "default"
DMA_COPY = "dma_copy"
VECTOR_CONCAT = "vector_concat"
PHONY_INSN = "phony_insn"

BLOCK_SIZE_BYTE = 32


class ComputeAt:
    """
    ConcatSchedule ComputeAt
    """

    def __init__(self):
        self._compute_at_axis = None

    @property
    def compute_at_axis(self):
        """
        :return: compute_at_axis
        """
        return self._compute_at_axis

    @compute_at_axis.setter
    def compute_at_axis(self, axis):
        """
        set compute_at_axis
        :param axis:
        :return:
        """
        self._compute_at_axis = axis


class EmitInsn:
    """
    ConcatSchedule EmitInsn Bean
    """

    def __init__(self):
        self._emit_insn_axis = None

    @property
    def emit_insn_axis(self):
        """
        :return: emit_insn_axis
        """
        return self._emit_insn_axis

    @emit_insn_axis.setter
    def emit_insn_axis(self, axis):
        """
        :param axis:
        :return: emit_insn_axis
        """
        self._emit_insn_axis = axis


class Util:
    @staticmethod
    def is_const(strategy: TilingStrategy):
        return strategy == TilingStrategy.CONST

    @staticmethod
    def is_one_concat(strategy: TilingStrategy):
        return strategy == TilingStrategy.ONE_CONCAT

    @staticmethod
    def is_read_align(strategy: TilingStrategy):
        return strategy in [TilingStrategy.READ_ALIGN, TilingStrategy.READ_ALIGN_NO_UB]

    @staticmethod
    def is_last_half(strategy: TilingStrategy):
        return strategy in [TilingStrategy.LAST_HALF_DIVISIBLE, TilingStrategy.LAST_HALF_DIVISIBLE_NO_ALIGN]

    @staticmethod
    def is_single_split(strategy: TilingStrategy):
        return strategy in [TilingStrategy.GENERAL_NO_ALIGN,
                            TilingStrategy.READ_ALIGN_NO_UB, TilingStrategy.LAST_HALF_DIVISIBLE_NO_ALIGN]


class ConcatSchedule(Schedule):
    """
    ConcatSchedule
    """

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.CONCAT]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [ConcatPattern.C_0]

    def __init__(self, outs: List[tvm.tensor.Tensor], tiling_case):
        self._out: tvm.tensor.Tensor = outs[0]
        self._schedule: Optional[tvm.schedule] = None
        self._tiling_case: Optional[ConcatTilingCase] = tiling_case
        self._tiling_strategy: TilingStrategy = self._tiling_case.tiling_strategy
        self._enable_db: bool = self._tiling_case.enable_db
        self._no_cut = self._tiling_strategy in [TilingStrategy.NONE_CUT, TilingStrategy.EMPTY]
        self._is_all_one = False
        if self._tiling_strategy == TilingStrategy.LAST_ALL_ONE:
            self._tiling_strategy = TilingStrategy.GENERAL_NO_ALIGN
            self._is_all_one = True

        self._scope = "local.UB"

        self._in_out_map = {}

        self._input_tensors = []
        self._out_tensors = set()
        self._middle_tensors = set()

        self._cache_read_tensors = []
        self._cache_read_buffer_tensors_map = {}
        self._input_tensor_map = {}

        self._cache_write_tensors = set()
        self._cache_write_buffer_tensors_map = {}
        self._out_tensor_map = {}

        self._cache_read_buffers = []
        self._bind_buffer_tensors = []

        self._need_do_block = False
        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._block_split_axis = self._tiling_case.block_split_axis
        self._block_factor = 1
        self._low_ub_factor = 1
        self._high_ub_factor = 1
        self._const_one_concat = False
        self._const_no_align = False
        self._is_const_all_one = False
        self._const_read_align = False
        self._const_read_align_no_ub = False
        self._is_const_last_align = False
        self._const_last_align_factor = -1

        self._coexisting_quantity = 2

        self._block_out_axis = None

        self._storage_align_axis = -1
        self._storage_align_factor = []
        self._storage_align_factor_index = []

        self._reorder_axis = []

        self._compute_inline_tensors = set()

        self._compute_at_map = {}
        self._compute_at_axis = ComputeAt()

        self._emit_insn_axis = EmitInsn()
        self._emit_insn_map = {}

        self._ub_size = util.get_ub_size()
        self._tensor_space: Optional[int] = None

        self._constraints = []

    def do_schedule(self):
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case.tiling_key

        self._calc_cache_read()
        self._do_cache_read()

        self._set_scope()

        self._calc_bind_buffer()
        self._calc_mem_reuse()
        self._calc_tiling()
        self._calc_storage_align()
        self._calc_cache_write()
        self._do_cache_write()
        self._calc_storage_bound()
        self._calc_reorder()
        self._calc_compute_inline()
        self._calc_multi_core()
        self._calc_compute_at()
        self._calc_double_buffer()
        self._calc_constraints()
        self._calc_emit_insn()

        self._do_storage_align()
        self._do_mem_reuse()
        self._do_tiling()
        self._do_reorder()
        self._do_multi_core()
        self._do_bind_buffer()
        self._do_storage_bound()
        self._do_compute_at()
        self._do_compute_inline()
        self._do_double_buffer()
        self._do_constraints()
        self._do_emit_insn()

        self._add_compile_info()

        return self._schedule if self._check_tiling_case() else None

    def _construct_compute_graph(self):
        def _dfs_sub_graph(out):
            for tensor_i in out.op.input_tensors:
                util.merge_value(self._in_out_map, tensor_i, out)
                if util.is_placeholder(tensor_i):
                    self._input_tensors.append(tensor_i)
                else:
                    self._middle_tensors.add(tensor_i)
                _dfs_sub_graph(tensor_i)

        _dfs_sub_graph(self._out)
        self._out_tensors.add(self._out)

    def _calc_cache_read(self):
        self._cache_read_tensors.extend(self._input_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffers.append(buffer_tensor)
            self._cache_read_buffer_tensors_map[buffer_tensor] = tensor_i

    def _calc_cache_write(self):
        if Util.is_one_concat(self._tiling_strategy) or self._const_one_concat:
            return
        self._cache_write_tensors.update(self._out_tensors)

    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensors_map[buffer_tensor] = tensor_i

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._middle_tensors:
            sch[tensor_i].set_scope(self._scope)

    def _calc_bind_buffer(self):
        self._bind_buffer_tensors.extend(self._cache_read_buffers)

    def _calc_mem_reuse(self):
        pass

    def _calc_tiling(self):
        funcs = {TilingStrategy.GENERAL: self._calc_tiling_general,
                 TilingStrategy.LAST_HALF_DIVISIBLE: self._calc_tiling_general,
                 TilingStrategy.NONE_CUT: self._calc_tiling_none_cut,
                 TilingStrategy.CONST: self._calc_tiling_const,
                 TilingStrategy.ONE_CONCAT: self._calc_tiling_one_concat,
                 TilingStrategy.GENERAL_NO_ALIGN: self._calc_tiling_single_split,
                 TilingStrategy.LAST_HALF_DIVISIBLE_NO_ALIGN: self._calc_tiling_single_split,
                 TilingStrategy.READ_ALIGN: self._calc_tiling_general,
                 TilingStrategy.READ_ALIGN_NO_UB: self._calc_tiling_single_split,
                 TilingStrategy.EMPTY: self._calc_tiling_none_cut}
        funcs[self._tiling_strategy]()

    def _calc_tiling_single_split(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis
        u_l_i = 0
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_l_bound = (1, util.get_bound(shape[u_l_i])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._low_ub_factor = operation.var_inner("_ub_factor_" + str(u_l_i), u_l_bound)

    def _calc_tiling_general(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis
        u_l_i = 0
        u_h_i = 1
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_l_bound = (1, util.get_bound(shape[u_l_i])[1])
        u_h_bound = (1, util.get_bound(shape[u_h_i])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._low_ub_factor = operation.var_inner("_ub_factor_" + str(u_l_i), u_l_bound)
        self._high_ub_factor = operation.var_inner("_ub_factor_" + str(u_h_i), u_h_bound)

    def _calc_tiling_none_cut(self):
        pass

    def _calc_tiling_const(self):
        inputs = []
        for _input in self._input_tensors:
            input_shape = util.shape_to_list(_input.shape)
            inputs.append({"shape": input_shape, "dtype": _input.dtype})
        res = self._out
        output_shape = util.shape_to_list(res.shape)
        outputs = [{"shape": output_shape, "dtype": res.dtype}]
        const_compile_info = get_compile_info()
        new_compile_info = {
            CompileInfo.CORE_NUM: util.get_core_num(),
            CompileInfo.UB_SIZE: util.get_ub_size(),
            "_ori_axis": 1,
            "_only_const_tiling": True,
            "_is_const": False
        }
        const_compile_info.update(new_compile_info)

        op_type = "AutoTiling"
        run_info = op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)
        tiling_format = {
            "need_multi_core": "int",
            "const_one_concat": "int",
            "const_read_align": "int",
            "last_align_factor": "int",
            "block_axis": "int",
            "block_factor": "int",
            "low_ub_factor": "int",
            "high_ub_factor": "int"
        }
        tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
        self._block_dims = run_info["block_dim"]
        self._need_do_block = True if tiling_data["need_multi_core"] > 0 else False
        self._const_last_align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
        if self._need_do_block:
            self._const_one_concat = True if tiling_data["const_one_concat"] > 0 else False
            self._const_read_align = True if tiling_data["const_read_align"] > 0 else False
            self._const_last_align_factor = tiling_data["last_align_factor"]
            self._block_split_axis = tiling_data["block_axis"]
            self._block_factor = tiling_data["block_factor"]
            self._low_ub_factor = tiling_data["low_ub_factor"]
            self._high_ub_factor = tiling_data["high_ub_factor"]
            if self._high_ub_factor < 0:
                if self._const_read_align:
                    self._const_read_align_no_ub = True
                else:
                    self._const_no_align = True
            self._is_const_all_one = not self._const_one_concat and self._const_no_align \
                and output_shape[1] == len(self._input_tensors)
        self._no_cut = self._need_do_block is False
        self._is_const_last_align = self._const_last_align_factor != (
            BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype])

    def _calc_tiling_one_concat(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis
        u_i = 1
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_bound = (1, util.get_bound(shape[u_i])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._high_ub_factor = operation.var_inner("_ub_factor_" + str(u_i), u_bound)

    def _calc_storage_bound(self):
        self._coexisting_quantity = 2
        if Util.is_one_concat(self._tiling_strategy) or self._const_one_concat:
            self._coexisting_quantity = 1
            self._ub_size -= BLOCK_SIZE_BYTE
        if Util.is_read_align(self._tiling_strategy) or self._const_read_align:
            self._coexisting_quantity = 2
            shape = util.shape_to_list(self._out.shape)
            if shape[0] == 1:
                self._coexisting_quantity = 1

    def _calc_reorder(self):
        pass

    def _calc_compute_inline(self):
        pass

    def _calc_multi_core(self):
        pass

    def _calc_compute_at(self):
        if self._no_cut:
            return

        for tensor_i in self._input_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._middle_tensors - self._compute_inline_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_read_buffer_tensors_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_write_buffer_tensors_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

    def _calc_storage_align(self):
        const_no_storage_align = self._const_one_concat or self._const_no_align \
            or self._const_read_align or (self._is_const_last_align and self._const_no_align)
        no_storage_align = self._no_cut or Util.is_one_concat(self._tiling_strategy) \
            or Util.is_single_split(self._tiling_strategy) or Util.is_read_align(self._tiling_strategy)
        if const_no_storage_align or no_storage_align:
            return
        self._storage_align_axis = 0

    def _calc_double_buffer(self):
        pass

    def _calc_constraints(self):
        if Util.is_last_half(self._tiling_strategy):
            shape = util.shape_to_list(self._out.shape)
            ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
            half = ele_in_block // 2
            self._constraints.append(tvm.expr.EQ(shape[-1] % half, 0))
            self._constraints.append(tvm.expr.EQ(self._high_ub_factor % half, 0))

    def _calc_emit_insn(self):
        def calc_concat_name_and_attrs():
            instruction_name = VECTOR_CONCAT
            if Util.is_read_align(self._tiling_strategy) or self._const_read_align:
                instruction_name = DMA_COPY
                shape = util.shape_to_list(self._out.shape)
                if shape[0] == 1:
                    instruction_name = PHONY_INSN
            attrs = {}
            if self._tiling_strategy == TilingStrategy.GENERAL_NO_ALIGN or self._const_no_align or self._no_cut:
                attrs = {"concat_no_align": 1}
                if self._is_all_one or self._is_const_all_one:
                    attrs = {"concat_no_align": 2}
            if self._tiling_strategy == TilingStrategy.LAST_HALF_DIVISIBLE_NO_ALIGN or (
                    self._is_const_last_align and self._const_no_align):
                attrs = {"concat_no_align": 1,
                         "concat_last_align_factor": 2}
            elif self._tiling_strategy == TilingStrategy.LAST_HALF_DIVISIBLE or self._is_const_last_align:
                attrs = {"concat_last_align_factor": 2}

            if instruction_name == VECTOR_CONCAT:
                attrs["concat_tmp_reuse_src"] = True
            return instruction_name, attrs

        for source, target in self._cache_read_buffer_tensors_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], DMA_COPY]

        for tensor_i in (self._middle_tensors - self._compute_inline_tensors):
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], tensor_i.op.tag]

        for source, target in self._cache_write_buffer_tensors_map.items():
            insn_name, attrs = calc_concat_name_and_attrs()
            self._emit_insn_map[source] = [source.op.axis[0], insn_name, attrs]

        for tensor_i in self._out_tensors:
            attrs = {"no_overlap": 3,
                     "no_overlap_malloc_buf_for_tail": 0}
            if Util.is_one_concat(self._tiling_strategy) or self._const_one_concat:
                attrs = {"no_overlap": 2}
            if Util.is_read_align(self._tiling_strategy) or self._const_read_align:
                attrs = {"no_overlap": 0}
            self._emit_insn_map[tensor_i] = [self._emit_insn_axis, DMA_COPY, attrs]

    def _do_bind_buffer(self):
        if Util.is_const(self._tiling_strategy):
            self._do_const_bind_buffer()
        else:
            self._do_dynamic_bind_buffer()

    def _do_const_bind_buffer(self):
        def set_single_split_offset(row_factor):
            offset = 0
            for tensor_i in self._bind_buffer_tensors:
                sch[tensor_i].bind_buffer(tensor_i.op.axis[-1], 1, offset)
                shape = util.shape_to_list(tensor_i.shape)
                offset += (row_factor * shape[-1] + ele_in_block - 1) // ele_in_block * ele_in_block

        def set_double_split_offset(row_factor, col_factor, align_factor):
            first_offset = 0
            extent = 0
            extent_n = 0
            out_ub_axis = self._compute_at_axis.compute_at_axis
            out_block_axis = self._block_out_axis
            for i, tensor_i in enumerate(self._bind_buffer_tensors):
                shape = util.shape_to_list(tensor_i.shape)
                if self._block_split_axis == 0:
                    cond0 = extent > out_ub_axis * col_factor
                    cond1 = extent < (out_ub_axis + 1) * col_factor
                else:
                    row_block_nums = (shape[0] + row_factor - 1) // row_factor
                    cond0 = extent > (out_block_axis // row_block_nums) * (
                        col_factor * self._block_factor) + out_ub_axis * col_factor
                    cond1 = extent < (out_block_axis // row_block_nums) * (
                        col_factor * self._block_factor) + (out_ub_axis + 1) * col_factor
                offset = tvm.select(tvm.all(cond0, cond1), first_offset, 0)
                sch[tensor_i].bind_buffer(tensor_i.op.axis[-1], 1, offset)
                extent += shape[-1]
                extent_n += shape[-1]
                if extent_n < col_factor:
                    first_offset += ((row_factor * shape[-1] + align_factor - 1) // align_factor * align_factor)
                else:
                    first_offset = row_factor * ((extent_n %
                                                  col_factor + ele_in_block - 1) // ele_in_block * ele_in_block)
                extent_n = extent_n % col_factor

        sch = self._schedule
        ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
        if self._no_cut:
            out_shape = util.shape_to_list(self._out.shape)
            set_single_split_offset(out_shape[0])
        elif self._const_no_align or self._const_read_align_no_ub:
            set_single_split_offset(self._low_ub_factor)
        elif self._const_one_concat:
            set_double_split_offset(1, self._high_ub_factor, ele_in_block)
        else:
            set_double_split_offset(self._low_ub_factor, self._high_ub_factor, 1)

    def _do_dynamic_bind_buffer(self):
        def set_single_split_offset(row_factor):
            offset = 0
            for tensor_i in self._bind_buffer_tensors:
                sch[tensor_i].bind_buffer(tensor_i.op.axis[-1], 1, offset)
                shape = util.shape_to_list(tensor_i.shape)
                offset += (row_factor * shape[-1] + ele_in_block - 1) // ele_in_block * ele_in_block

        def set_double_split_offset(row_factor, col_factor):
            extent = 0
            out_ub_axis = self._compute_at_axis.compute_at_axis
            out_block_axis = self._block_out_axis
            for i, tensor_i in enumerate(self._bind_buffer_tensors):
                shape = util.shape_to_list(tensor_i.shape)
                if i == 0:
                    extent += shape[-1]
                    continue
                first_offset = operation.var_inner(f"_offset_{i}")
                if self._block_split_axis == 0:
                    cond0 = extent > out_ub_axis * col_factor
                    cond1 = extent < (out_ub_axis + 1) * col_factor
                else:
                    row_block_nums = (shape[0] + row_factor - 1) // row_factor
                    cond0 = extent > (out_block_axis // row_block_nums) * (
                        col_factor * self._block_factor) + out_ub_axis * col_factor
                    cond1 = extent < (out_block_axis // row_block_nums) * (
                        col_factor * self._block_factor) + (out_ub_axis + 1) * col_factor
                offset = tvm.select(tvm.all(cond0, cond1), first_offset, 0)
                sch[tensor_i].bind_buffer(tensor_i.op.axis[-1], 1, offset)
                extent += shape[-1]

        sch = self._schedule
        ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
        if self._no_cut:
            out_shape = util.shape_to_list(self._out.shape)
            set_single_split_offset(out_shape[0])
        elif Util.is_single_split(self._tiling_strategy):
            set_single_split_offset(self._low_ub_factor)
        elif self._tiling_strategy == TilingStrategy.ONE_CONCAT:
            set_double_split_offset(1, self._high_ub_factor)
        else:
            set_double_split_offset(self._low_ub_factor, self._high_ub_factor)

    def _do_mem_reuse(self):
        sch = self._schedule
        base_tensor = self._cache_read_buffers[0]
        for index in range(1, len(self._cache_read_buffers)):
            sch[base_tensor].reused_by(self._cache_read_buffers[index])

        shape = util.shape_to_list(self._out.shape)
        if (Util.is_read_align(self._tiling_strategy) or self._const_read_align) and shape[0] == 1:
            for tensor_i in self._cache_write_buffer_tensors_map.keys():
                sch[base_tensor].reused_by(tensor_i, reuse_data=True)

    def _do_tiling(self):
        funcs = {TilingStrategy.GENERAL: self._do_tiling_general,
                 TilingStrategy.LAST_HALF_DIVISIBLE: self._do_tiling_general,
                 TilingStrategy.NONE_CUT: self._do_tiling_none_cut,
                 TilingStrategy.CONST: self._do_tiling_const,
                 TilingStrategy.ONE_CONCAT: self._do_tiling_one_concat,
                 TilingStrategy.GENERAL_NO_ALIGN: self._do_tiling_single_split,
                 TilingStrategy.LAST_HALF_DIVISIBLE_NO_ALIGN: self._do_tiling_single_split,
                 TilingStrategy.READ_ALIGN: self._do_tiling_general,
                 TilingStrategy.READ_ALIGN_NO_UB: self._do_tiling_single_split,
                 TilingStrategy.EMPTY: self._do_tiling_none_cut}
        funcs[self._tiling_strategy]()

    def _do_tiling_single_split(self):
        sch = self._schedule
        res = self._out

        _low_ub_split_axis = 0
        low_u_o, low_u_i = sch[res].split(res.op.axis[_low_ub_split_axis], factor=self._low_ub_factor)
        b_o, b_i = sch[res].split(low_u_o, factor=self._block_factor)
        self._reorder_axis = [b_o, b_i, low_u_i, res.op.axis[1]]
        self._compute_at_axis.compute_at_axis = b_i
        self._emit_insn_axis.emit_insn_axis = low_u_i

    def _do_tiling_general(self):
        sch = self._schedule
        res = self._out

        _low_ub_split_axis = 0
        _high_ub_split_axis = 1
        low_u_o, low_u_i = sch[res].split(res.op.axis[_low_ub_split_axis], factor=self._low_ub_factor)
        high_u_o, high_u_i = sch[res].split(res.op.axis[_high_ub_split_axis], factor=self._high_ub_factor)
        if self._block_split_axis == _low_ub_split_axis:
            b_o, b_i = sch[res].split(low_u_o, factor=self._block_factor)
            self._reorder_axis = [b_o, b_i, high_u_o, low_u_i, high_u_i]
            self._compute_at_axis.compute_at_axis = high_u_o
        else:
            b_o, b_i = sch[res].split(high_u_o, factor=self._block_factor)
            self._reorder_axis = [b_o, low_u_o, b_i, low_u_i, high_u_i]
            self._compute_at_axis.compute_at_axis = b_i
        self._emit_insn_axis.emit_insn_axis = low_u_i

    def _do_tiling_none_cut(self):
        res = self._out
        self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_tiling_const(self):
        if self._need_do_block:
            if self._const_read_align:
                if self._const_read_align_no_ub:
                    self._do_tiling_single_split()
                else:
                    self._do_tiling_general()
            elif self._const_one_concat:
                self._do_tiling_one_concat()
            elif self._const_no_align:
                self._do_tiling_single_split()
            else:
                self._do_tiling_general()
        else:
            res = self._out
            self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_tiling_one_concat(self):
        sch = self._schedule
        res = self._out

        _high_ub_split_axis = 1
        high_u_o, high_u_i = sch[res].split(res.op.axis[_high_ub_split_axis], factor=self._high_ub_factor)
        if self._block_split_axis != _high_ub_split_axis:
            b_o, b_i = sch[res].split(res.op.axis[self._block_split_axis], factor=self._block_factor)
            self._reorder_axis = [b_o, b_i, high_u_o, high_u_i]
            self._compute_at_axis.compute_at_axis = high_u_o
        else:
            b_o, b_i = sch[res].split(high_u_o, factor=self._block_factor)
            self._reorder_axis = [b_o, res.op.axis[0], b_i, high_u_i]
            self._compute_at_axis.compute_at_axis = b_i
        self._emit_insn_axis.emit_insn_axis = high_u_i

    def _do_storage_align(self):
        sch = self._schedule
        if self._storage_align_axis == -1:
            return

        offset = 0
        bound = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
        extent_n = 0
        input_nums = len(self._cache_read_buffers)
        for i, tensor_i in enumerate(self._cache_read_buffers):
            shape = util.shape_to_list(tensor_i.shape)
            if Util.is_const(self._tiling_strategy):
                extent_n += shape[-1]
                if i == input_nums - 1 and extent_n == shape[-1] and extent_n % bound != 0:
                    align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
                elif extent_n <= self._high_ub_factor:
                    align_factor = 1
                else:
                    align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
                extent_n %= self._high_ub_factor
            else:
                align_factor = operation.var_inner(f"_align_factor_{i}", (1, bound))
                self._storage_align_factor.append(align_factor)
                self._storage_align_factor_index.append(i)
            sch[tensor_i].storage_align(tensor_i.op.axis[self._storage_align_axis], align_factor, offset)

        align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
        for tensor_i in self._cache_write_buffer_tensors_map:
            sch[tensor_i].storage_align(tensor_i.op.axis[self._storage_align_axis], align_factor, offset)

    def _do_reorder(self):
        if self._no_cut:
            return

        sch = self._schedule
        res = self._out
        sch[res].reorder(*self._reorder_axis)

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._middle_tensors.union(
            self._cache_read_buffer_tensors_map.keys()).union(
            self._cache_write_buffer_tensors_map.keys())

        tensor_space = self._ub_size // self._coexisting_quantity
        if self._enable_db:
            tensor_space = self._ub_size // 2 // self._coexisting_quantity
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_buffer_size(storage_bound)

    def _do_multi_core(self):
        if self._no_cut:
            return

        sch = self._schedule
        res = self._out
        block_axis = self._reorder_axis[:self._block_split_axis + 1]
        block_bind_axis = sch[res].fuse(*block_axis)
        block = tvm.thread_axis("blockIdx.x")
        self._block_out_axis = block
        sch[res].bind(block_bind_axis, block)

    def _do_compute_at(self):
        sch = self._schedule
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1].compute_at_axis)

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()

    def _do_double_buffer(self):
        if self._enable_db:
            sch = self._schedule

            tensors = self._middle_tensors.union(
                self._cache_read_buffer_tensors_map.keys()).union(
                self._cache_write_buffer_tensors_map.keys())

            for tensor_i in tensors:
                sch[tensor_i].double_buffer()

    def _do_constraints(self):
        sch = self._schedule
        for cond in self._constraints:
            if isinstance(cond, tvm.expr.Expr):
                sch.set_constraint(cond)

    def _do_emit_insn(self):
        sch = self._schedule

        for tensor_i, param in self._emit_insn_map.items():
            emit_insn_axis = param[0]
            if isinstance(emit_insn_axis, EmitInsn):
                emit_insn_axis = emit_insn_axis.emit_insn_axis
            if len(param) > 2:
                sch[tensor_i].emit_insn(emit_insn_axis, param[1], param[2])
            else:
                sch[tensor_i].emit_insn(emit_insn_axis, param[1])

    def _add_compile_info(self):
        if len(self._storage_align_factor_index) != 0:
            operation.add_compile_info_inner("_align_vars", self._storage_align_factor_index)
        if CompileInfo.CORE_NUM in get_compile_info() and self._tiling_strategy != TilingStrategy.CONST:
            return

        operation.add_compile_info_inner(CompileInfo.CORE_NUM, util.get_core_num())
        operation.add_compile_info_inner(CompileInfo.UB_SIZE, util.get_ub_size())
        operation.add_compile_info_inner("_only_const_tiling", False)
        if self._tiling_strategy == TilingStrategy.CONST:
            operation.add_compile_info_inner("_is_const", True)
            operation.add_compile_info_inner("_const_dims", self._block_dims)
        else:
            operation.add_compile_info_inner("_is_const", False)
        shape_is_var = []
        is_first = True
        is_same_input = operation.get_context().get("_is_same_input") or False
        for i, tensor_i in enumerate(self._input_tensors):
            shape = util.shape_to_list(tensor_i.shape)
            cur_shape = []
            for j, value in enumerate(shape):
                if isinstance(value, int):
                    cur_shape.append(False)
                else:
                    if (j == 0 and is_first) or j != 0:
                        cur_shape.append(True)
                        is_first = is_first and i != 0
                    else:
                        cur_shape.append(False)
            shape_is_var.append(cur_shape)
            if is_same_input:
                break
        operation.add_compile_info_inner("_concat_vars", shape_is_var)

    def _check_tiling_case(self):
        def none_cut_check():
            ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self._out.dtype]
            row_limit = isinstance(shape[0], int) and shape[0] > ele_in_block * 16
            col_limit = isinstance(shape[1], int) and shape[1] > 128
            if row_limit or col_limit:
                return False
            lower_bound = 1
            for item in shape:
                cur_bound = util.get_bound(item)[0]
                if cur_bound is None:
                    return False
                lower_bound *= cur_bound
            if not self._tensor_space // DTYPE_BYTE_MAPPING[self._out.dtype] >= lower_bound:
                return False
            return True

        if Util.is_const(self._tiling_strategy):
            return True
        shape = util.shape_to_list(self._out.shape)
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            return none_cut_check()
        elif self._tiling_strategy in [TilingStrategy.GENERAL_NO_ALIGN, TilingStrategy.LAST_HALF_DIVISIBLE_NO_ALIGN]:
            cur_bound = util.get_bound(shape[1])[0]
            if cur_bound is None or cur_bound > self._tensor_space / 16 * 2:
                return False
        elif Util.is_one_concat(self._schedule.tiling_key):
            m_bound = util.get_bound(shape[0])[0]
            n_bound = util.get_bound(shape[1])[1]
            if m_bound is None or n_bound is None:
                return True
            return n_bound / 128 > m_bound / util.get_core_num()
        elif self._is_all_one:
            for tensor in self._input_tensors:
                input_shape = util.shape_to_list(tensor.shape)
                if isinstance(input_shape[-1], int) and input_shape[-1] != 1:
                    return False
        return True
