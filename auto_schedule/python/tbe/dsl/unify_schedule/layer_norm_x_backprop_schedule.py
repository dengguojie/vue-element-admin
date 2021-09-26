#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
layer_norm_x_backprop schedule
"""
from .constants import Pattern
from tbe import tvm
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base.operation import add_compile_info

from . import util
from .constants import DTYPE_BYTE_MAPPING
from .layer_norm_x_backprop_tilingcase import TilingStrategy

# block size in D architecture
BLOCK_SIZE_BYTE = 32

# temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192
FP_32 = 8
FP_16 = 16


@register_schedule(pattern=Pattern.LAYER_NORM_X_BACKPROP)
def schedule(outs, tiling_case):
    """
    schedule for layer_norm_x_backprop dynamic shape
    """

    return LayerNormXBackpropSchedule(outs, tiling_case).do_schedule()


class LayerNormXBackpropSchedule:
    """
    LayerNormXBackpropSchedule
    """

    def __init__(self, outs, tiling_case):
        self._out = None
        self._outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._tiling_key = self._tiling_case.get("key")
        self._open_double_buffer = self._tiling_case.get("is_need_db", False)

        self._scope = "local.UB"

        self._in_out_map = {}
        self._input_tensors = set()
        self._middle_tensors = set()
        self._out_tensors = set()

        self._broadcast_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._broadcast_axis_num = {}

        self._cache_read_tensors = set()
        self._cache_read_buffer_tensor_map = {}
        self._placeholder_tensor_map = {}

        self._cache_write_buffer_tensor_map = {}
        self._cache_write_tensor_map = {}

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._ub_size = util.get_ub_size()
        self._correct_factor = 2 if self._open_double_buffer else 1
        self._tmp_ub_size = 0

        self._compute_at_map = {}

        self._compute_at_axis = None
        self._compute_at_axis_idx = None
        self._emit_insn_axis = None
        self.sum_x_block_outer = None

        self._ir_axes = []
        self._inner_shape = []

        self._emit_insn_map = {}

    def do_schedule(self):
        """
        do schedule
        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case["key"]

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()
        self._do_compute_inline()

        self._calc_storage_bound()
        self._do_storage_bound()

        self._calc_tiling()
        self._do_tiling()

        self._calc_compute_at()
        self._do_compute_at()
        self._do_multi_core()

        self._do_double_buffer()

        self._calc_emit_insn()
        self._do_emit_insn()
        self._add_compile_info()

        return self._schedule

    def _do_compute_inline(self):
        pass

    def _construct_compute_graph(self):
        def match_scalar_scene(tensor_):
            # condition:
            # 1. tensor --> tensor
            # 2. broadcast tensor is output
            # 3. next compute support scalar
            if len(tensor_.op.input_tensors) != 0 and \
                    util.get_tensor_size(tensor_.op.input_tensors[0]) != 1:
                return False
            if tensor_ in self._out_tensors:
                return False
            if all(util.support_scalar(tensor_o) for tensor_o in
                   self._in_out_map.get(tensor_)):
                return True
            return False

        def _no_broadcast(_src_shapes, _dst_shapes):
            _src_shapes = util.shape_to_list(_src_shapes)
            _dst_shapes = util.shape_to_list(_dst_shapes)
            broadcast_num = 0
            for x, y in zip(_src_shapes, _dst_shapes):
                if not expr_equal(x, y):
                    broadcast_num += 1
            return broadcast_num

        self._out_tensors = set(self._outs)
        visited_tensors = set()

        for out in self._out_tensors:
            self._dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        for tensor_i in self._broadcast_tensors:
            if match_scalar_scene(tensor_i):
                self._absorbable_broadcast_tensors.add(tensor_i)

        for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
            if tensor_i.op.tag != "broadcast":
                src_shapes = tensor_i.op.input_tensors[0].shape[0:]
                dst_shapes = tensor_i.shape[0:]
                self._broadcast_axis_num[tensor_i] = _no_broadcast(src_shapes, dst_shapes)

        self._out = self._outs[0]

    def _dfs_sub_graph(self, out, visited_tensors: set):
        for tensor_i in out.op.input_tensors:
            util.merge_value(self._in_out_map, tensor_i, out)
            self._dtypes.add(tensor_i.dtype)

            if util.is_placeholder(tensor_i):
                self._input_tensors.add(tensor_i)
            else:
                self._middle_tensors.add(tensor_i)

                if tensor_i.op.tag.find("unified_broadcast") != -1:
                    self._broadcast_tensors.add(tensor_i)

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self._dfs_sub_graph(tensor_i, visited_tensors)

    def _set_scope(self):
        for tensor_i in self._middle_tensors:
            self._schedule[tensor_i].set_scope(self._scope)

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(
                tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._placeholder_tensor_map[tensor_i] = buffer_tensor

    def _calc_cache_write(self):
        pass

    def _do_cache_write(self):
        for tensor_i in self._out_tensors:
            self.final_out_buffer = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[self.final_out_buffer] = tensor_i

    def _calc_storage_bound(self):

        def _r_coexisting(_tensor):
            if _tensor in dependent_map:
                return len(dependent_map)
            if util.is_vtranspose_broadcast(_tensor):
                self._tmp_ub_size += VTRANSPOSE_TEMP_SPACE
            _need_space = []
            for _tensor_i in _tensor.op.input_tensors:
                _need_space.append(_r_coexisting(_tensor_i))

            _current_space = len(dependent_map) + 1

            if util.get_dsl_insn(_tensor) == "unified_broadcast" and self._broadcast_axis_num.get(_tensor, 0) > 1:
                _current_space += 1

            _need_space.append(_current_space)
            _refresh_dependent(_tensor)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = self._in_out_map[_tensor].copy()
            return max(_need_space)

        def _refresh_dependent(_tensor):
            for _tensor_i in _tensor.op.input_tensors:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        coexisting_quantities = []
        dependent_map = {}
        for tensor_i in self._out.op.input_tensors:
            coexisting_quantities.append(_r_coexisting(tensor_i))

        current_space = len(dependent_map) + 1
        coexisting_quantities.append(current_space)

        self._coexisting_quantity = max(coexisting_quantities)

        if self._coexisting_quantity == 1:
            self._tmp_ub_size += BLOCK_SIZE_BYTE
        if len(self._broadcast_tensors) > 0:
            self._tmp_ub_size += BLOCK_SIZE_BYTE

    def _do_storage_bound(self):

        self._ub_size -= self._correct_factor * self._tmp_ub_size
        tensor_space = self._ub_size // self._coexisting_quantity
        if self._open_double_buffer:
            tensor_space = tensor_space // 2
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        tensors = list(self._middle_tensors) + \
            list(self._cache_read_buffer_tensor_map.keys()) + \
            list(self._cache_write_buffer_tensor_map.keys())

        for tensor_i in tensors:
            self.storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            self._schedule[tensor_i].set_buffer_size(self.storage_bound)

    def _calc_tiling(self):
        funcs = {TilingStrategy.NONE_CUT: self._calc_tiling_none_cut}
        funcs[self._tiling_strategy]()

    def _calc_tiling_none_cut(self):
        pass

    def _do_tiling(self):
        funcs = {TilingStrategy.NONE_CUT: self._do_tiling_none_cut}
        funcs[self._tiling_strategy]()

    def _do_tiling_none_cut(self):
        res = self._out
        core_num = util.get_core_num()
        block_split_axis_index = 0

        block_split_axis = res.op.axis[block_split_axis_index]
        self.sum_x_block_outer, _ = self._schedule[res].split(
            block_split_axis, nparts=core_num
        )
        self._is_split_ub = True
        ub_factor = 32 // self._max_dtype_bytes
        self._ub_split_axis_index = 1
        if self._is_split_ub:
            ub_outer, ub_inner = self._schedule[res].split(
                res.op.axis[self._ub_split_axis_index], factor=ub_factor
            )
        self._compute_at_axis = ub_outer
        self._emit_at_axis = ub_inner

    def _do_multi_core(self):
        if self.sum_x_block_outer is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self.sum_x_block_outer, block)

    def _calc_compute_at(self):
        for tensor_i in self._middle_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_read_buffer_tensor_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_write_buffer_tensor_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

    def _do_compute_at(self):
        sch = self._schedule
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1])

    def _do_double_buffer(self):
        if self._open_double_buffer:
            sch = self._schedule

            tensors = list(self._cache_read_buffer_tensor_map.keys())

            for tensor_i in tensors:
                sch[tensor_i].double_buffer()

    def _calc_emit_insn(self):
        def _get_emit_insn_map(tensor_):
            insn_map = {"elewise_single_cast": "vector_conv",
                        "elewise_single_VS_max": "vector_maxs",
                        "elewise_single_VS_min": "vector_mins",
                        "elewise_single_log": "vector_ln",
                        "elewise_single_exp": "vector_exp",
                        "elewise_single_relu": "vector_relu",
                        "elewise_single_abs": "vector_abs",
                        "elewise_single_not": "vector_not",
                        "elewise_single_sqrt": "vector_sqrt",
                        "elewise_single_rsqrt": "vector_rsqrt",
                        "elewise_binary_mul": "vector_mul",
                        "elewise_binary_sub": "vector_sub",
                        "elewise_single_VS_mul": "vector_muls",
                        "elewise_binary_div": "vector_div",
                        "elewise_binary_add": "vector_add",
                        "elewise_single_VS_add": "vector_adds",
                        "elewise_binary_min": "vector_min",
                        "elewise_binary_max": "vector_max",
                        "elewise_binary_vcmpv_gt": "vector_gt",
                        "elewise_binary_vcmpv_ge": "vector_ge",
                        "elewise_binary_vcmpv_lt": "vector_lt",
                        "elewise_binary_vcmpv_le": "vector_le",
                        "elewise_binary_vcmpv_eq": "vector_eq",
                        "elewise_binary_vcmpv_ne": "vector_ne",
                        "elewise_binary_or": "vector_or",
                        "elewise_binary_and": "vector_and",
                        "elewise_multiple_mla": "vector_multiple",
                        "elewise_multiple_madd": "vector_multiple",
                        "elewise_multiple_maddrelu": "vector_multiple",
                        "tuple_reduce_sum": "vector_reduce_sum",
                        "reduce_sum":  "vector_reduce_sum",
                        "broadcast": "vector_dup",
                        "unified_broadcast":  "vector_broadcast",
                        "broadcast_for_tensor": "unified_broadcast"}
            if tensor_.op.tag.find("|") != -1:
                str_list = tensor_.op.tag.split("|")
                insn = insn_map.get(str_list[0])
            else:
                insn = insn_map.get(tensor_.op.tag)
            return insn

        for source, target in self._cache_read_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], "dma_copy"]

        for tensor_i in self._middle_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], _get_emit_insn_map(tensor_i)]

        for source, target in self._cache_write_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], _get_emit_insn_map(target)]

        for tensor_i in self._out_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[2], "dma_copy"]

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            if tensor_i.op.name in ["broadcast_tensor_0", "broadcast_tensor_1", "broadcast_tensor_2",
                                    "broadcast_tensor_3", "broadcast_tensor_4"]:
                sch[tensor_i].emit_insn(tensor_i.op.axis[1], param[1])
                continue
            sch[tensor_i].emit_insn(param[0], param[1])

    def _add_compile_info(self):
        add_compile_info("UB_SIZE", self._ub_size)
        add_compile_info("CORE_NUM", util.get_core_num())
        add_compile_info("MAX_DTYPE", self._max_dtype_bytes)
        add_compile_info("COEXISTING_QUANTITY", self._coexisting_quantity)
