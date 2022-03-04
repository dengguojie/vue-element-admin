#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2022-2022 Huawei Technologies Co., Ltd
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
split schedule
"""
from typing import Optional
from typing import List
from operator import mul
from functools import reduce

from tbe import tvm
from tbe.dsl.base import operation
from tbe.common.utils import op_tiling
from tbe.dsl.base.operation import get_compile_info

from tbe.dsl.unify_schedule import util
from tbe.dsl.unify_schedule.schedule import Schedule
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.constants import CompileInfo
from tbe.dsl.unify_schedule.constants import SplitPattern
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from .split_tilingcase import TilingStrategy
from .split_tilingcase import SplitTilingCase
from ...constants import FAKE_NODE_TAG

DEFAULT = "default"
SCOPE_UB = "local.UB"
DMA_COPY = "dma_copy"
COPY_UB_TO_UB = "dma_copy"
VNCHWCONV = "vnchwconv"
PHONY_INSN = "phony_insn"
SPLIT = "split"
SPLIT_GENERAL = "split_general"

BLOCK_SIZE_BYTE = 32
TRANSPOSE_FACTOR = 16


class ComputeAt:
    """
    SplitSchedule ComputeAt
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
    SplitSchedule EmitInsn Bean
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
    def is_base(strategy: TilingStrategy):
        return strategy == TilingStrategy.BASE

    @staticmethod
    def is_general(strategy: TilingStrategy):
        return strategy == TilingStrategy.GENERAL

    @staticmethod
    def is_all_align(strategy: TilingStrategy):
        return strategy == TilingStrategy.ALL_ALIGN

    @staticmethod
    def is_all_align_copy(strategy: TilingStrategy):
        return strategy == TilingStrategy.ALL_ALIGN_COPY

    @staticmethod
    def is_only_cut_m(strategy: TilingStrategy):
        return strategy == TilingStrategy.CUT_M

    @staticmethod
    def is_all_general(strategy: TilingStrategy):
        return strategy in [TilingStrategy.GENERAL, TilingStrategy.NONE_CUT]

    @staticmethod
    def is_transpose_strategy(strategy: TilingStrategy):
        return strategy in [TilingStrategy.GENERAL, TilingStrategy.NONE_CUT, TilingStrategy.CUT_M]


class SplitSchedule(Schedule):
    """
    SplitSchedule
    """

    def __init__(self, outs: List[tvm.tensor.Tensor], tiling_case):
        self._out: List[tvm.tensor.Tensor] = outs if isinstance(outs, (list, tuple)) else [outs]
        self._dtype = self._out[0].dtype
        self._schedule: Optional[tvm.schedule] = None
        self._tiling_case: Optional[SplitTilingCase] = tiling_case
        self._tiling_strategy: TilingStrategy = self._tiling_case.tiling_strategy
        self._enable_db: bool = self._tiling_case.enable_db
        self._no_cut = self._tiling_strategy in [TilingStrategy.NONE_CUT, TilingStrategy.EMPTY]
        self._avg_split = operation.get_context().get("_avg_split")

        self._in_out_map = {}

        self._input_tensors = set()
        self._out_tensors = []
        self._middle_tensors = set()

        self._cache_read_tensors = []

        self._cache_write_tensors = []

        self._cache_read_buffers = []
        self._cache_write_buffers = []

        self._block_split_axis = self._tiling_case.block_split_axis
        self._block_factor = 1
        self._avg_split_block_factor = 1
        self._row_factor_var = 1
        self._low_ub_factor = []
        self._high_ub_factor = []
        self._block_dims = 1

        self._in_reshape_map = {}
        self._out_reshape_map = {}
        self._in_reshape_tensors = set()
        self._out_reshape_tensors = set()
        self._in_transpose_tensors = set()
        self._out_transpose_tensors = set()

        self._coexisting_quantity = 3

        self._storage_align_map = {}

        self._compute_align_map = {}

        self._reorder_axis = []

        self._compute_inline_tensors = set()

        self._compute_at_map = {}
        self._compute_at_axis: List[ComputeAt] = [ComputeAt() for _ in range(len(self._out))]

        self._emit_insn_axis: List[EmitInsn] = [EmitInsn() for _ in range(len(self._out))]
        self._emit_insn_map = {}

        self._ub_size = util.get_ub_size()
        self._tensor_space: Optional[int] = None

        self._constraints = []
        self._const_invalid = False

        self._is_const = Util.is_const(self._tiling_strategy)
        self._is_base = Util.is_base(self._tiling_strategy)
        self._is_general = Util.is_general(self._tiling_strategy)
        self._is_all_align = Util.is_all_align(self._tiling_strategy)
        self._is_all_align_copy = Util.is_all_align_copy(self._tiling_strategy)
        self._is_only_cut_m = Util.is_only_cut_m(self._tiling_strategy)
        self._is_all_general = Util.is_all_general(self._tiling_strategy)
        self._is_transpose_strategy = Util.is_transpose_strategy(self._tiling_strategy)

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.SPLIT]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [SplitPattern.S_0]

    def do_schedule(self):
        self._construct_compute_graph()
        self._calc_tiling()
        if self._const_invalid:
            return None

        self._schedule = self._create_schedule()
        self._schedule.tiling_key = self._tiling_case.tiling_key

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._calc_reshape()
        self._calc_transpose()
        self._do_reshape()
        self._do_transpose()

        self._calc_storage_align()
        self._calc_compute_align()
        self._calc_storage_bound()
        self._calc_reorder()
        self._calc_compute_inline()
        self._calc_multi_core()
        self._calc_compute_at()
        self._calc_double_buffer()
        self._calc_axis_group()
        self._calc_constraints()
        self._calc_emit_insn()
        self._calc_set_store_predicate()

        self._do_storage_align()
        self._do_compute_align()
        self._do_tiling()
        self._do_reorder()
        self._do_multi_core()
        self._do_storage_bound()
        self._do_compute_at()
        self._do_compute_inline()
        self._do_double_buffer()
        self._do_constraints()
        self._do_emit_insn()
        self._do_axis_group()
        self._do_set_store_predicate()

        self._add_compile_info()

        return self._schedule if self._check_tiling_case() else None

    def _construct_compute_graph(self):
        def _dfs_sub_graph(out):
            for tensor_i in out.op.input_tensors:
                util.merge_value(self._in_out_map, tensor_i, out)
                if util.is_placeholder(tensor_i):
                    self._input_tensors.add(tensor_i)
                else:
                    self._middle_tensors.add(tensor_i)
                _dfs_sub_graph(tensor_i)

        for _out in self._out:
            _dfs_sub_graph(_out)
        self._out_tensors.extend(self._out)

    def _fake_node(self, dst_shape, outputs, axis):
        def concat_func(*indices):
            func = None
            concat_axis_size = sum(t.shape[axis] for t in outputs)
            for tensor_i in reversed(outputs):
                index = []
                for i, _ in enumerate(dst_shape):
                    if i == axis:
                        index.append(indices[i] - (concat_axis_size - tensor_i.shape[axis]))
                    else:
                        index.append(indices[i])
                if func is None:
                    func = tensor_i(*index)
                else:
                    func = tvm.select(indices[axis] < concat_axis_size, tensor_i(*index), func)
                concat_axis_size -= tensor_i.shape[axis]
            return func

        with tvm.tag_scope(FAKE_NODE_TAG):
            concat = tvm.compute(dst_shape, concat_func, name="fake_node")
        return concat

    def _create_schedule(self):
        if len(self._out) > 1 and not self._is_base:
            dst_shape = util.shape_to_list(list(self._input_tensors)[0].shape)
            self._out = [self._fake_node(dst_shape, self._out, 1)]
        res_op = []
        for _out in self._out:
            res_op.append(_out.op)
        return tvm.create_schedule(res_op)

    def _calc_cache_read(self):
        self._cache_read_tensors.extend(self._input_tensors)

    def _do_cache_read(self):
        input_tensor = self._cache_read_tensors[0]
        for tensor_i in self._out:
            if len(self._out) == 1:
                use_tensors = self._in_out_map.get(input_tensor)
            else:
                use_tensors = [tensor_i]
            buffer_tensor = self._schedule.cache_read(input_tensor, SCOPE_UB, use_tensors)
            self._cache_read_buffers.append(buffer_tensor)

    def _calc_cache_write(self):
        self._cache_write_tensors.extend(self._out_tensors)

    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, SCOPE_UB)
            self._cache_write_buffers.append(buffer_tensor)

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._middle_tensors:
            sch[tensor_i].set_scope(SCOPE_UB)

    def _calc_reshape(self):
        if not self._is_transpose_strategy:
            return
        for tensor_i in self._cache_read_buffers:
            read_cache_cache = self._schedule.cache_read(tensor_i, SCOPE_UB, self._cache_write_buffers)
            if self._no_cut:
                self._in_reshape_map[read_cache_cache] = [TRANSPOSE_FACTOR]
            else:
                self._in_reshape_map[read_cache_cache] = [self._low_ub_factor[0], TRANSPOSE_FACTOR]

        for tensor_i in self._cache_write_buffers:
            if self._no_cut:
                self._out_reshape_map[tensor_i] = [TRANSPOSE_FACTOR]
            else:
                self._out_reshape_map[tensor_i] = [self._low_ub_factor[0], TRANSPOSE_FACTOR]

    def _calc_transpose(self):
        pass

    def _calc_tiling(self):
        funcs = {TilingStrategy.NONE_CUT: self._calc_tiling_none_cut,
                 TilingStrategy.CONST: self._calc_tiling_const,
                 TilingStrategy.BASE: self._calc_tiling_base_split,
                 TilingStrategy.GENERAL: self._calc_tiling_general_split,
                 TilingStrategy.ALL_ALIGN: self._calc_tiling_align_split,
                 TilingStrategy.ALL_ALIGN_COPY: self._calc_tiling_align_copy_split,
                 TilingStrategy.CUT_M: self._calc_tiling_single_split,
                 TilingStrategy.EMPTY: self._calc_tiling_none_cut}
        funcs.get(self._tiling_strategy)()

    def _calc_tiling_align_copy_split(self):
        shape = util.shape_to_list(list(self._input_tensors)[0].shape)
        b_i = self._block_split_axis
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_l_bound = (1, util.get_bound(shape[0])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._low_ub_factor = [operation.var_inner("_ub_factor_0", u_l_bound)]

    def _calc_tiling_align_split(self):
        shape = util.shape_to_list(list(self._input_tensors)[0].shape)
        b_i = self._block_split_axis
        u_l_i = 0
        u_h_i = 1
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_l_bound = (1, util.get_bound(shape[u_l_i])[1])
        u_h_bound = (1, util.get_bound(shape[u_h_i])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._low_ub_factor = [operation.var_inner("_ub_factor_" + str(u_l_i), u_l_bound)]
        self._high_ub_factor = [operation.var_inner("_ub_factor_" + str(u_h_i), u_h_bound)]

    def _calc_tiling_single_split(self):
        shape = util.shape_to_list(list(self._input_tensors)[0].shape)
        b_i = self._block_split_axis
        b_bound = (1, util.get_bound(shape[b_i])[1])
        ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self._dtype)
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._row_factor_var = 1
        low_factor = TRANSPOSE_FACTOR * ele_in_block
        self._low_ub_factor = [low_factor]

    def _calc_tiling_general_split(self):
        shape = util.shape_to_list(list(self._input_tensors)[0].shape)
        b_i = self._block_split_axis
        u_h_i = 1
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_h_bound = (1, util.get_bound(shape[u_h_i])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self._dtype)
        low_factor = TRANSPOSE_FACTOR * ele_in_block
        self._low_ub_factor = [low_factor]
        self._high_ub_factor = [operation.var_inner("_ub_factor_" + str(u_h_i), u_h_bound)]

    def _calc_tiling_base_split(self):
        res = self._out[0]
        shape = util.shape_to_list(res.shape)
        b_i = 0
        u_l_i = 0
        u_h_i = 1
        b_bound = (1, util.get_bound(shape[b_i])[1])
        self._block_factor = operation.var_inner("_block_factor", b_bound)
        if self._avg_split:
            u_l_bound = (1, util.get_bound(shape[u_l_i])[1])
            u_h_bound = (1, util.get_bound(shape[u_h_i])[1])
            self._avg_split_block_factor = operation.var_inner("_avg_block_factor", u_h_bound)
            low_ub_factor = operation.var_inner(f"_ub_factor_{u_l_i}", u_l_bound)
            high_ub_factor = operation.var_inner(f"_ub_factor_{u_h_i}", u_h_bound)
        for index, _out in enumerate(self._out):
            shape = util.shape_to_list(_out.shape)
            u_l_bound = (1, util.get_bound(shape[u_l_i])[1])
            u_h_bound = (1, util.get_bound(shape[u_h_i])[1])
            if not self._avg_split:
                low_ub_factor = operation.var_inner(f"_ub_factor_{u_l_i}_{index}", u_l_bound)
                high_ub_factor = operation.var_inner(f"_ub_factor_{u_h_i}_{index}", u_h_bound)
            self._low_ub_factor.append(low_ub_factor)
            self._high_ub_factor.append(high_ub_factor)

    def _calc_tiling_none_cut(self):
        pass

    def _calc_tiling_const(self):
        def get_tiling():
            inputs = []
            for _input in self._input_tensors:
                input_shape = util.shape_to_list(_input.shape)
                inputs.append({"shape": input_shape, "dtype": _input.dtype})
            outputs = []
            output_split_shape = []
            for _output in self._out_tensors:
                output_shape = util.shape_to_list(_output.shape)
                outputs.append({"shape": output_shape, "dtype": _output.dtype})
                output_split_shape.append(output_shape[1])
            ori_compile_info = get_compile_info()
            new_compile_info = {
                CompileInfo.PATTERN: ori_compile_info.get(CompileInfo.PATTERN),
                CompileInfo.CORE_NUM: util.get_core_num(),
                CompileInfo.UB_SIZE: util.get_ub_size(),
                "_ori_axis": 1,
                "_only_const_tiling": True,
                "_is_const": False,
                "_avg_split": self._avg_split,
            }

            op_type = "AutoTiling"
            run_info = op_tiling.do_op_tiling(op_type, new_compile_info, inputs, outputs)
            return run_info

        def decode_tiling(run_info):
            output_num = 1 if self._avg_split else len(self._out_tensors)
            tiling_format = {
                "need_multi_core": "int",
                "is_base": "int",
                "is_all_align": "int",
                "only_cut_m": "int",
                "block_axis": "int",
                "block_factor": "int",
                "avg_block_factor": "int",
                "low_ub_factors": [output_num, "int"],
                "high_ub_factors": [output_num, "int"]
            }
            tiling_data = op_tiling.decode(run_info.get("tiling_data"), tiling_format)
            self._block_dims = run_info.get("block_dim")
            need_do_block = tiling_data.get("need_multi_core") > 0
            const_is_base = tiling_data.get("is_base") > 0
            const_is_all_align = tiling_data.get("is_all_align") > 0
            const_only_cut_m = tiling_data.get("only_cut_m") > 0
            self._block_split_axis = tiling_data.get("block_axis")
            self._block_factor = tiling_data.get("block_factor")
            self._low_ub_factor = tiling_data.get("low_ub_factors")
            self._high_ub_factor = tiling_data.get("high_ub_factors")
            if self._avg_split and const_is_base:
                self._avg_split_block_factor = tiling_data.get("avg_block_factor")
                self._low_ub_factor = [self._low_ub_factor[0]] * len(self._out_tensors)
                self._high_ub_factor = [self._high_ub_factor[0]] * len(self._out_tensors)
            input_shape = util.shape_to_list(list(self._input_tensors)[0].shape)
            self._is_all_align_copy = const_is_all_align and const_only_cut_m
            if self._is_all_align_copy:
                const_is_all_align = False
                const_only_cut_m = False
            self._no_cut = not need_do_block or reduce(mul, input_shape, 1) == 0
            self._is_base = const_is_base
            self._is_general = not any([const_is_base, const_is_all_align, const_only_cut_m,
                                        self._no_cut, self._is_all_align_copy])
            self._is_all_align = const_is_all_align
            self._is_only_cut_m = const_only_cut_m
            self._is_all_general = self._is_general or self._no_cut
            self._is_transpose_strategy = self._is_all_general or self._is_only_cut_m
            mode = operation.get_context().get_current_compute().get("_mode")
            if mode == SPLIT and self._is_transpose_strategy:
                self._const_invalid = True
            if mode == SPLIT_GENERAL and not self._is_transpose_strategy:
                self._const_invalid = True

        run_info = get_tiling()
        decode_tiling(run_info)

    def _calc_storage_bound(self):
        self._coexisting_quantity = 3
        if self._is_base:
            self._coexisting_quantity = 2
            if len(self._out_tensors) == 1:
                self._coexisting_quantity = 1
        if self._is_all_align:
            self._coexisting_quantity = 1
        if self._is_all_align_copy:
            self._coexisting_quantity = 2

    def _calc_reorder(self):
        pass

    def _calc_compute_inline(self):
        if self._is_transpose_strategy:
            self._compute_inline_tensors.update(self._in_reshape_map.keys())
            self._compute_inline_tensors.update(self._in_reshape_tensors)
            self._compute_inline_tensors.update(self._cache_write_buffers)

        if self._is_all_align or self._is_base:
            self._compute_inline_tensors.update(self._cache_write_buffers)

    def _calc_multi_core(self):
        pass

    def _calc_compute_at(self):
        if self._no_cut:
            return

        for index, out_tensor in enumerate(self._out):
            self._compute_at_map[self._cache_read_buffers[index]] = [out_tensor, self._compute_at_axis[index]]
            self._compute_at_map[self._cache_write_buffers[index]] = [out_tensor, self._compute_at_axis[index]]

        if self._is_all_align_copy:
            for tensor_i in self._cache_write_buffers:
                self._compute_at_map[tensor_i] = [self._out[0], self._compute_at_axis[0]]

        for tensor_i in self._in_transpose_tensors:
            self._compute_at_map[tensor_i] = [self._out[0], self._compute_at_axis[0]]
        for tensor_i in self._out_transpose_tensors:
            self._compute_at_map[tensor_i] = [self._out[0], self._compute_at_axis[0]]
        for tensor_i in self._out_reshape_tensors:
            self._compute_at_map[tensor_i] = [self._out[0], self._compute_at_axis[0]]
        if not self._is_base:
            for tensor_i in self._out_tensors:
                self._compute_at_map[tensor_i] = [self._out[0], self._compute_at_axis[0]]

    def _calc_storage_align(self):
        offset = 0
        factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self._out[0].dtype)
        if self._is_general:
            for tensor_i in self._cache_read_buffers:
                self._storage_align_map[tensor_i] = [tensor_i.op.axis[-2], factor, offset]
        if self._is_base and len(self._out) != 1:
            for tensor_i in self._cache_read_buffers:
                self._storage_align_map[tensor_i] = [tensor_i.op.axis[-2], factor, offset]
            for tensor_i in self._cache_write_buffers:
                self._storage_align_map[tensor_i] = [tensor_i.op.axis[-2], factor, offset]

    def _calc_compute_align(self):
        factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self._out[0].dtype)
        if self._is_general and not self._avg_split:
            for tensor_i in self._out_transpose_tensors:
                self._compute_align_map[tensor_i] = [tensor_i.op.axis[-2], factor]
            for tensor_i in self._out_reshape_tensors:
                self._compute_align_map[tensor_i] = [tensor_i.op.axis[-1], factor]
        if self._is_base:
            for tensor_i in self._cache_write_buffers:
                self._compute_align_map[tensor_i] = [tensor_i.op.axis[-1], factor]

    def _calc_double_buffer(self):
        pass

    def _calc_axis_group(self):
        pass

    def _calc_constraints(self):
        if self._is_only_cut_m or self._no_cut:
            factor = self._row_factor_var if self._is_only_cut_m else 1
            input_shape = util.shape_to_list(list(self._input_tensors)[0].shape)
            self._constraints.append(factor * input_shape[1] < 255)
            if self._avg_split:
                out_shape = util.shape_to_list(self._out_tensors[0].shape)
                self._constraints.append(factor * out_shape[1] < 255)
            else:
                for out in self._out_tensors:
                    out_shape = util.shape_to_list(out.shape)
                    self._constraints.append(factor * out_shape[1] < 255)
        elif self._is_general:
            self._constraints.append(self._high_ub_factor[0] < 255)

    def _calc_emit_insn(self):
        for tensor_i in self._cache_read_buffers:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], DMA_COPY]

        for tensor_i in self._cache_write_buffers:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], COPY_UB_TO_UB]

        if self._is_base:
            attrs = {"no_overlap": 3,
                     "no_overlap_malloc_buf_for_tail": 0}
            if len(self._out_tensors) == 1:
                attrs = {"no_overlap": 0}
            for tensor_i, axis in zip(self._out_tensors, self._emit_insn_axis):
                self._emit_insn_map[tensor_i] = [axis, DMA_COPY, attrs]
        else:
            attrs = {}
            is_no_overlap = self._is_all_align or self._is_only_cut_m or \
                (self._is_all_general and self._avg_split) or self._is_all_align_copy
            if is_no_overlap:
                attrs = {"no_overlap": 0}
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], DMA_COPY, attrs]
            for index, tensor_i in enumerate(self._out):
                self._emit_insn_map[tensor_i] = [self._emit_insn_axis[index], PHONY_INSN]

        for tensor_i in self._in_transpose_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], VNCHWCONV]
        for tensor_i in self._out_transpose_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], COPY_UB_TO_UB]
        for tensor_i in self._out_reshape_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], VNCHWCONV]

    def _calc_set_store_predicate(self):
        pass

    def _do_tiling(self):
        funcs = {TilingStrategy.NONE_CUT: self._do_tiling_none_cut,
                 TilingStrategy.CONST: self._do_tiling_const,
                 TilingStrategy.GENERAL: self._do_tiling_general_split,
                 TilingStrategy.BASE: self._do_tiling_base_split,
                 TilingStrategy.CUT_M: self._do_tiling_single_split,
                 TilingStrategy.ALL_ALIGN: self._do_tiling_general_split,
                 TilingStrategy.ALL_ALIGN_COPY: self._do_tiling_single_split,
                 TilingStrategy.EMPTY: self._do_tiling_none_cut}
        funcs.get(self._tiling_strategy)()

    def _do_tiling_single_split(self):
        sch = self._schedule
        _low_ub_split_axis = 0
        _high_ub_split_axis = 1
        for index, _out in enumerate(self._out):
            m_out, m_inner = sch[_out].split(_out.op.axis[_low_ub_split_axis], factor=self._low_ub_factor[index])
            b_out, b_inner = sch[_out].split(m_out, factor=self._block_factor)
            self._reorder_axis.append([b_out, b_inner, m_inner, _out.op.axis[_high_ub_split_axis]])
            self._compute_at_axis[index].compute_at_axis = b_inner
            self._emit_insn_axis[index].emit_insn_axis = m_inner

    def _do_tiling_general_split(self):
        sch = self._schedule
        _low_ub_split_axis = 0
        _high_ub_split_axis = 1
        for index, _out in enumerate(self._out):
            m_out, m_inner = sch[_out].split(_out.op.axis[_low_ub_split_axis], factor=self._low_ub_factor[index])
            n_out, n_inner = sch[_out].split(_out.op.axis[_high_ub_split_axis], factor=self._high_ub_factor[index])
            if self._block_split_axis == _low_ub_split_axis:
                b_out, b_inner = sch[_out].split(m_out, factor=self._block_factor)
                self._reorder_axis.append([b_out, b_inner, n_out, m_inner, n_inner])
                self._compute_at_axis[index].compute_at_axis = n_out
            else:
                b_out, b_inner = sch[_out].split(n_out, factor=self._block_factor)
                self._reorder_axis.append([m_out, b_out, b_inner, m_inner, n_inner])
                self._compute_at_axis[index].compute_at_axis = b_inner
            self._emit_insn_axis[index].emit_insn_axis = m_inner

    def _do_tiling_base_split(self):
        sch = self._schedule
        for index, _out in enumerate(self._out):
            m_out, m_inner = sch[_out].split(_out.op.axis[0], factor=self._block_factor)
            m_inner_out, m_inner_inner = sch[_out].split(m_inner, factor=self._low_ub_factor[index])
            if self._avg_split:
                n_out, n_inner = sch[_out].split(_out.op.axis[1], factor=self._avg_split_block_factor)
                n_inner_out, n_inner_inner = sch[_out].split(n_inner, factor=self._high_ub_factor[index])
                self._compute_at_axis[index].compute_at_axis = n_inner_out
                self._reorder_axis.append([m_out, n_out, m_inner_out, n_inner_out, m_inner_inner, n_inner_inner])
            else:
                n_out, n_inner = sch[_out].split(_out.op.axis[1], factor=self._high_ub_factor[index])
                self._compute_at_axis[index].compute_at_axis = n_out
                self._reorder_axis.append([m_out, m_inner_out, n_out, m_inner_inner, n_inner])
            self._emit_insn_axis[index].emit_insn_axis = m_inner_inner

    def _do_tiling_none_cut(self):
        for index, _out in enumerate(self._out):
            self._emit_insn_axis[index].emit_insn_axis = _out.op.axis[0]

    def _do_tiling_const(self):
        if self._no_cut:
            self._do_tiling_none_cut()
        elif self._is_only_cut_m:
            self._do_tiling_single_split()
        elif self._is_base:
            self._do_tiling_base_split()
        else:
            self._do_tiling_general_split()

    def _reshape(self, tensor, factors):
        sch = self._schedule
        axis = tensor.op.axis[0]
        for index, factor in enumerate(factors):
            if index == len(factors) - 1:
                _, axis = sch[tensor].split(axis, nparts=factor, tail_strategy="round_up")
            else:
                _, axis = sch[tensor].split(axis, factor=factor, tail_strategy="round_up")
        reshape_tensor = sch.cache_write(tensor, SCOPE_UB)
        return reshape_tensor

    def _transpose(self, tensor, permute):
        sch = self._schedule
        reorder_axis = []
        for axis in permute:
            reorder_axis.append(tensor.op.axis[axis])
        sch[tensor].reorder(*reorder_axis)
        transpose_tensor = sch.cache_write(tensor, SCOPE_UB)
        return transpose_tensor

    def _do_reshape(self):
        for tensor_i, factors in self._in_reshape_map.items():
            reshape_tensor = self._reshape(tensor_i, factors)
            self._in_reshape_tensors.add(reshape_tensor)

        for tensor_i, factors in self._out_reshape_map.items():
            reshape_tensor = self._reshape(tensor_i, factors)
            self._out_reshape_tensors.add(reshape_tensor)

    def _do_transpose(self):
        for tensor_i in self._in_reshape_tensors:
            if self._no_cut:
                transpose_tensor = self._transpose(tensor_i, [1, 2, 0])
            else:
                transpose_tensor = self._transpose(tensor_i, [0, 2, 3, 1])
            self._in_transpose_tensors.add(transpose_tensor)

        for tensor_i in self._out_reshape_tensors:
            if self._no_cut:
                transpose_tensor = self._transpose(tensor_i, [1, 2, 0])
            else:
                transpose_tensor = self._transpose(tensor_i, [0, 2, 3, 1])
            self._out_transpose_tensors.add(transpose_tensor)

    def _do_storage_align(self):
        sch = self._schedule
        for tensor_i, params in self._storage_align_map.items():
            sch[tensor_i].storage_align(*params)

    def _do_compute_align(self):
        sch = self._schedule
        for tensor_i, params in self._compute_align_map.items():
            sch[tensor_i].compute_align(*params)

    def _do_reorder(self):
        if self._no_cut:
            return

        sch = self._schedule
        for _out, reorder_axes in zip(self._out, self._reorder_axis):
            sch[_out].reorder(*reorder_axes)

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._middle_tensors.union(
            self._cache_read_buffers).union(
            self._cache_write_buffers).union(
            self._in_transpose_tensors).union(
            self._out_transpose_tensors).union(
            self._out_reshape_tensors)

        tensor_space = self._ub_size // self._coexisting_quantity
        if self._enable_db:
            tensor_space = self._ub_size // 2 // self._coexisting_quantity
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING.get(tensor_i.dtype))
            sch[tensor_i].set_buffer_size(storage_bound)

    def _do_multi_core(self):
        if self._no_cut:
            return
        sch = self._schedule
        block = tvm.thread_axis("blockIdx.x")
        block_axis_end = self._block_split_axis + 1
        if self._is_base:
            block_axis_end = block_axis_end + 1 if self._avg_split else block_axis_end
        for _out, reorder_axes in zip(self._out, self._reorder_axis):
            block_axis = reorder_axes[:block_axis_end]
            block_bind_axis = sch[_out].fuse(*block_axis)
            sch[_out].bind(block_bind_axis, block)

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
                self._cache_read_buffers).union(
                self._cache_write_buffers)

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

    def _do_set_store_predicate(self):
        if self._is_transpose_strategy:
            sch = self._schedule
            ori_var = operation.get_context().get_current_compute().get("_ori_var")
            for tensor_i in self._cache_read_buffers:
                sch[tensor_i].set_store_predicate(tensor_i.op.axis[0] < ori_var)
            for tensor_i in self._out_tensors:
                sch[tensor_i].set_store_predicate(tensor_i.op.axis[0] < ori_var)

    def _do_axis_group(self):
        sch = self._schedule
        group_id = tvm.make.Call("int32", "axis_group", [0, "overwrite"], tvm.expr.Call.Extern, None, 0)
        if self._is_base and len(self._out_tensors) == 1:
            for tensor_i, axes in zip(self._out, self._reorder_axis):
                sch[tensor_i].pragma(axes[-1], "axis_group", group_id)
                sch[tensor_i].pragma(axes[-2], "axis_group", group_id)

        if (self._is_transpose_strategy and self._avg_split) or self._is_only_cut_m or self._no_cut:
            if self._is_only_cut_m or self._no_cut:
                for tensor_i in self._in_transpose_tensors:
                    sch[tensor_i].pragma(tensor_i.op.axis[-2], "axis_group", group_id)
                    sch[tensor_i].pragma(tensor_i.op.axis[-3], "axis_group", group_id)
            for tensor_i in self._out_reshape_tensors:
                sch[tensor_i].pragma(tensor_i.op.axis[-1], "axis_group", group_id)
                sch[tensor_i].pragma(tensor_i.op.axis[-2], "axis_group", group_id)

        is_continuous_out = (self._is_all_general and self._avg_split) or \
            self._is_only_cut_m or self._no_cut or self._is_all_align_copy
        if is_continuous_out:
            for tensor_i in self._out_tensors:
                sch[tensor_i].pragma(tensor_i.op.axis[-1], "axis_group", group_id)
                sch[tensor_i].pragma(tensor_i.op.axis[-2], "axis_group", group_id)

    def _add_compile_info(self):
        operation.get_context().get_current_compute().get_current_schedule().add("_split_num", len(self._out_tensors))
        if CompileInfo.CORE_NUM in get_compile_info() and not self._is_const:
            return

        operation.add_compile_info_inner(CompileInfo.CORE_NUM, util.get_core_num())
        operation.add_compile_info_inner(CompileInfo.UB_SIZE, util.get_ub_size())
        operation.add_compile_info_inner("_avg_split", self._avg_split)
        operation.add_compile_info_inner("_split_is_const",
                                         operation.get_context().get_current_compute().set_default("_split_is_const",
                                                                                                   False))
        operation.add_compile_info_inner("_only_const_tiling", False)
        if self._is_const:
            operation.add_compile_info_inner("_is_const", True)
            operation.add_compile_info_inner("_const_dims", self._block_dims)
        else:
            operation.add_compile_info_inner("_is_const", False)
        shape_is_var = []
        shape = util.shape_to_list(list(self._input_tensors)[0].shape)
        for value in shape:
            if isinstance(value, int):
                shape_is_var.append(False)
            else:
                shape_is_var.append(True)
        operation.add_compile_info_inner("_split_vars", shape_is_var)

    def _check_tiling_case(self):
        if self._is_only_cut_m or self._no_cut:
            input_shape = util.shape_to_list(list(self._input_tensors)[0].shape)
            if isinstance(input_shape[1], int):
                return input_shape[1] <= 255
            else:
                bound = operation.get_context().get_var(input_shape[1].name).get_bound()
                if bound[0] is None or bound[0] > 255:
                    return False
        return True
