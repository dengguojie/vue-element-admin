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
layer_norm_x_backprop_v2 schedule
"""

from tbe import dsl
from tbe import tvm
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base.operation import add_compile_info

from . import util
from .constants import DTYPE_BYTE_MAPPING
from .constants import INSN_MAPPING
from .constants import FAKE_NODE_TAG
from .constants import Pattern
from .layer_norm_x_backprop_v2_tilingcase import TilingStrategy

# block size in D architecture
BLOCK_SIZE_BYTE = 32

# temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192


@register_schedule(pattern=Pattern.LAYER_NORM_X_BACKPROP_V2)
def schedule(outs, tiling_case):
    """
    schedule for layer_norm_x_backprop_v2 dynamic shape
    """

    return LayerNormXBackpropScheduleV2(outs, tiling_case).do_schedule()


class LayerNormXBackpropScheduleV2:
    """
    LayerNormXBackpropScheduleV2
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
        self._cache_read_buffer_tensor_map = {}

        self._fake_middle_tensors = set()
        self._cache_write_buffer_tensor_map = {}

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 7
        self._ub_size = util.get_ub_size()
        self._correct_factor = 2 if self._open_double_buffer else 1
        self._tmp_ub_size = 0

        self._compute_at_map = {}

        self._block_bind_axis = None
        self._compute_at_axis = None
        self._compute_at_axis_idx = None
        self._emit_insn_axis = None
        self.sum_x_block_outer = None

        self._emit_insn_map = {}

    def do_schedule(self):
        """
        do schedule
        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case["key"]

        self._do_cache_read()
        self._do_cache_write()
        self._set_scope()
        self._do_storage_bound()

        self._do_tiling()

        self._do_compute_at()
        self._do_multi_core()

        self._calc_emit_insn()
        self._do_emit_insn()
        self._add_compile_info()

        return self._schedule

    def _fake_node(self, _out_tensors):
        if len(self._outs) != 2:
            raise RuntimeError("real out tensors only have two")
        pd_x, res_for_gamma = _out_tensors
        if pd_x.dtype == "float16":
            pd_x_ub = dsl.cast_to(pd_x, res_for_gamma.dtype)
            res = dsl.vadd(pd_x_ub, res_for_gamma)
        else:
            res = dsl.vadd(pd_x, res_for_gamma)
        return res

    def _construct_compute_graph(self):
        self._out_tensors = set(self._outs)
        visited_tensors = set()

        for out in self._out_tensors:
            self._dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors
        self._out = self._fake_node(list(self._out_tensors))
        self._dfs_sub_graph(self._out, visited_tensors)
        self._fake_middle_tensors = self._middle_tensors - self._pure_middle_tensors - self._out_tensors

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

    def _do_cache_read(self):
        for tensor_i in self._input_tensors:
            buffer_tensor = self._schedule.cache_read(
                tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i

    def _do_cache_write(self):
        for tensor_i in self._out_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[buffer_tensor] = tensor_i

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._pure_middle_tensors:
            sch[tensor_i].set_scope(self._scope)

    def _do_storage_bound(self):
        tensor_space = self._ub_size // self._coexisting_quantity
        if self._open_double_buffer:
            tensor_space = tensor_space // 2
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        sch = self._schedule
        tensors = self._middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_buffer_size(storage_bound)

    def _do_tiling(self):
        funcs = {TilingStrategy.NONE_CUT: self._do_tiling_none_cut}
        funcs[self._tiling_strategy]()

    def _do_tiling_none_cut(self):
        res = self._out
        last_dim = int(res.shape[-1])
        core_num = util.get_core_num()
        ub_factor = self._tensor_space // (self._max_dtype_bytes * last_dim)

        ub_outer, ub_inner = self._schedule[res].split(res.op.axis[1], factor=ub_factor)
        fuse_axis = self._schedule[res].fuse(res.op.axis[0], ub_outer)
        block_outer, block_inner = self._schedule[res].split(fuse_axis, nparts=core_num)

        self.sum_x_block_outer = block_outer
        self._compute_at_axis = block_inner
        self._emit_insn_axis = ub_inner

    def _do_multi_core(self):
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[self._out].bind(self.sum_x_block_outer, block)

    def _do_compute_at(self):
        sch = self._schedule
        for tensors in [self._middle_tensors,
                        self._cache_read_buffer_tensor_map,
                        self._cache_write_buffer_tensor_map]:
            for tensor_i in tensors:
                sch[tensor_i].compute_at(sch[self._out], self._compute_at_axis)

    def _calc_emit_insn(self):
        def _get_emit_insn_map(tensor_):
            tag = tensor_.op.tag
            if tag.find("|") != -1:
                insn = tag.split("|")[0]
            else:
                insn = tag
            return INSN_MAPPING.get(insn, insn)

        for source, target in self._cache_read_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], "dma_copy"]

        for tensor_i in self._pure_middle_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], _get_emit_insn_map(tensor_i)]

        for source, target in self._cache_write_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], _get_emit_insn_map(target)]

        for tensor_i in self._out_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], "dma_copy"]

        for tensor_i in self._fake_middle_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], "phony_insn"]

        if len(self._out_tensors) > 1:
            self._emit_insn_map[self._out] = [self._emit_insn_axis, "phony_insn"]
        else:
            self._emit_insn_map[self._out] = [self._emit_insn_axis, "dma_copy"]

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            sch[tensor_i].emit_insn(param[0], param[1])

    def _add_compile_info(self):
        add_compile_info("UB_SIZE", self._ub_size)
        add_compile_info("CORE_NUM", util.get_core_num())
        add_compile_info("MAX_DTYPE", self._max_dtype_bytes)
        add_compile_info("COEXISTING_QUANTITY", self._coexisting_quantity)
