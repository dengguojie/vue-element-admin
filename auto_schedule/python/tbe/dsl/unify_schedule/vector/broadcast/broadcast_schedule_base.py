#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
base broadcast schedule
"""
import copy
from typing import Optional
from functools import reduce
from operator import mul

from tbe import tvm
from tbe.common.utils import op_tiling
from tbe.dsl.base import operation
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.operation import get_compile_info
from tbe.common.platform import SHORT_SOC_VERSION
from tbe.common.platform import ASCEND_910B
from tbe.common.platform.platform_info import get_soc_spec

from tbe.dsl.base import d_format_util
from tbe.dsl.padding import padding
from tbe.dsl.padding.padding import ActionValueType, ActionType
from ... import util
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import INSN_MAPPING
from ...constants import TERNARY_INSNS
from .broadcast_tilingcase import TilingStrategy
from .broadcast_tilingcase import BroadcastTilingCase
from .broadcast_helper import GraphHelper

# block size in D architecture
BLOCK_SIZE_BYTE = 32
ONE_DIM_ALIGN = 128
SPECIAL_FACTOR_ALIGN = 256
ONE_REPEAT_BYTES = 256

# temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192
MAX_EXTEND_NODE_NUM = 2
ROW_LIMIT = 16
CONST_DB_MIDDLE_NODES_LIMIT = 6
CONST_AHEAD_DB_FACTOR = 4
CONST_BRC_INLINE_LIMIT = 4
LAST_LOOP_REG_MOV = 2
LAST_LOOP_COEXISTING_QUANTITY_LIMIT = 2
MISSALIGN_STRIDE_WHITH_MALLOC_BUF = 3
MISSALIGN_STRIDE_COEXISTING_QUANTITY_LIMIT = 3
BRC_LAST_DUP_LIMIT = 16

CONST = "const"
BROADCAST = "broadcast"
VECTOR_BROADCAST = "vector_broadcast"
UNKNOWN_BROADCAST = "unknown_broadcast"
DMA_COPY = "dma_copy"
PHONY_INSN = "phony_insn"
PURE_BRC = "pure_brc"


# 'pylint: disable=R0902, R0903
class BaseBroadcastSchedule:
    """
    BaseBroadcastSchedule
    """

    class ComputeAt:
        """
        BaseBroadcast ComputeAt
        """

        def __init__(self):
            self._compute_at_axis = None

        @property
        def compute_at_axis(self):
            """
            return compute_at_axis
            """
            return self._compute_at_axis

        @compute_at_axis.setter
        def compute_at_axis(self, axis):
            self._compute_at_axis = axis

    class EmitInsn:
        """
        BaseBroadcast EmitInsn Bean
        """

        def __init__(self):
            self._emit_insn_axis = None

        @property
        def emit_insn_axis(self):
            """
            return emit_insn_axis
            """
            return self._emit_insn_axis

        @emit_insn_axis.setter
        def emit_insn_axis(self, axis):
            self._emit_insn_axis = axis

    def __init__(self, outs, tiling_case):
        # init compute graph variable
        self._init_compute_graph(outs)

        # init tiling variable
        self._init_tiling(tiling_case)

        #  init schedule variable
        self._init_schedule()

    def _init_compute_graph(self, outs):
        self._out: Optional[tvm.tensor.Tensor] = None
        self._outs = outs
        self._input_tensors = set()
        self._middle_tensors = set()
        self._pure_middle_tensors = set()
        self._middle_out_tensors = set()
        self._out_tensors = set()
        self._broadcast_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._before_broadcast_tensors = set()
        self._broadcast_axis_num = {}
        self._dtypes = set()
        self._outs_dtypes = set()
        self._max_dtype_bytes = 4
        self._max_brc_bytes = 2
        self._coexisting_quantity = 1
        self._in_out_map = {}
        self._compute_root_tensors = set()
        self._is_vnchwconv_align = True
        self._brc_avoid_bank_conflict = False

    def _init_tiling(self, tiling_case):
        self._tiling_case: Optional[BroadcastTilingCase] = tiling_case
        self._tiling_strategy = self._tiling_case.tiling_strategy
        self._is_pure_brc = operation.get_context().get_current_compute().get("_mode") == PURE_BRC
        self._is_no_store_align = True
        self._is_store_align = self._tiling_case.is_storage_align
        self._need_do_block = False
        self._block_dims = 1
        self._block_split_axis = \
            -1 if self._tiling_case.block_split_axis is None else self._tiling_case.block_split_axis
        self._block_factor = 1
        self._ub_split_axis = 0 if self._tiling_case.ub_split_axis is None else self._tiling_case.ub_split_axis
        self._ub_factor = 1
        self._ub_factor_is_one = False
        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._ub_factor_align = ONE_DIM_ALIGN

    def _init_schedule(self):
        self._schedule = None
        self._enable_db = self._tiling_case.enable_db
        self._is_one_dim = self._tiling_case.is_one_dim
        self._mode = operation.get_context().get_current_compute().get("_mode")
        self._is_pure_brc_common_db = self._is_pure_brc and self._enable_db and \
            (operation.get_context().get("_pure_brc_common") or False)
        self._scope = "local.UB"
        self._block_bind_axis = None
        self._compute_at_axis = self.ComputeAt()
        self._compute_at_axis_idx = None
        self._emit_insn_axis = self.EmitInsn()
        self._tensor_space = None
        self._ub_size = util.get_ub_size()
        self._correct_factor = 2 if self._enable_db else 1
        self._tmp_ub_size = 0
        self._min_storage_bound = -1
        self._broadcast_by_no_other_use = {}
        self._cache_read_tensors = set()
        self._cache_read_buffer_tensor_map = {}
        self._input_tensor_map = {}
        self._middle_out_cache_read_buffer_map = {}
        self._cache_write_tensors = set()
        self._cache_write_buffer_tensor_map = {}
        self._out_tensor_map = {}
        self._middle_out_cache_write_buffer_map = {}
        self._broadcast_store_predicate = set()
        self._store_predicate_common_tensors = set()
        self._all_pre_node_broadcast = set()
        self._compute_inline_tensors = set()
        self._const_brc_inline_tensor = set()
        self._store_align_tensors = set()
        self._axis_group_tensors = set()
        self._compute_at_map = {}
        self._copy_out_tensors = set()
        self._remove_out_tensors = set()
        self._inner_shape = []
        self._constraints = set()
        self._mem_reuse_map = {}
        self._data_reuse_map = {}
        self._emit_insn_map = {}
        self._compute_align_map = {}
        self._remove_pad_map = {}
        self._5hd_actions = set()
        self._set_value_cache_read_map = {}

    def do_schedule(self):
        """
        :return:
        """
        self._construct_compute_graph()
        self._visit_before_broadcast_node()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case.tiling_key

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._calc_remove_pad()

        self._calc_unify_tiling()
        self._calc_set_value()
        self._calc_compute_inline()
        self._calc_storage_align()
        self._calc_multi_core()
        self._calc_ub_align()
        self._calc_compute_at()
        self._calc_double_buffer()
        self._calc_mem_reuse()
        self._calc_constraints()
        self._calc_emit_insn()

        self._do_tiling()
        self._do_storage_bound()
        self._do_compute_inline()
        self._do_set_value()
        self._do_multi_core()
        self._do_storage_align()
        self._do_ub_align()
        self._do_remove_pad()
        self._do_compute_at()
        self._do_store_predicate()
        self._do_double_buffer()
        self._do_mem_reuse()
        self._do_constraints()
        self._do_emit_insn()
        self._do_axis_group()

        self._add_compile_info()

        return self._schedule if self._check_tiling_case() else None

    def _get_ub_tensor(self, tensor_i):
        if tensor_i in self._input_tensor_map:
            return self._input_tensor_map.get(tensor_i)
        if tensor_i in self._out_tensor_map:
            return self._out_tensor_map.get(tensor_i)
        return tensor_i

    def _get_all_ub_tensors(self, tensors):
        def get_multi_ub_tensor(_tensor):
            ub_tensors_set = set()
            if tensor_i in self._input_tensor_map:
                ub_tensors_set.add(self._input_tensor_map.get(tensor_i))
            if tensor_i in self._out_tensor_map:
                ub_tensors_set.add(self._out_tensor_map.get(tensor_i))
            if not ub_tensors_set:
                return {tensor_i}
            return ub_tensors_set

        ub_tensors = set()
        for tensor_i in tensors:
            ub_tensors.update(get_multi_ub_tensor(tensor_i))
        return ub_tensors

    def _calc_unify_tiling(self):
        def enable_brc_inline():
            for _tensor in self._broadcast_tensors - self._compute_inline_tensors:
                shapes = util.shape_to_list(_tensor.shape)
                ele_in_block = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(_tensor.dtype))
                one_repeat = int(ONE_REPEAT_BYTES // DTYPE_BYTE_MAPPING.get(_tensor.dtype))
                if isinstance(shapes[-1], int) and one_repeat % shapes[-1] == 0 and \
                   ele_in_block < shapes[-1] <= CONST_BRC_INLINE_LIMIT * ele_in_block:
                    return True
            return False

        def broadcast_is_align():
            for _tensor in self._broadcast_tensors - self._compute_inline_tensors:
                src_shapes = util.shape_to_list(_tensor.shape)
                ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(_tensor.dtype)
                if not isinstance(src_shapes[-1], int) or src_shapes[-1] % ele_in_block != 0:
                    return False
            return True

        self._brc_avoid_bank_conflict = not broadcast_is_align()
        operation.add_compile_info_inner("_brc_avoid_bank_conflict", self._brc_avoid_bank_conflict)
        if self._tiling_strategy == TilingStrategy.CONST:
            enable_ahead_calc = self._ahead_calc()
            if enable_ahead_calc:
                middle_num = len(self._middle_tensors - self._compute_root_tensors)
                input_num = len(self._input_tensors - self._compute_root_tensors)
                self._enable_db = False
                if input_num * CONST_AHEAD_DB_FACTOR >= middle_num:
                    self._enable_db = True
                if enable_brc_inline():
                    self._tmp_ub_size += ONE_REPEAT_BYTES
                self._calc_tiling()
            else:
                self._get_const_storage_bound()
                self._enable_db = True
                if self._coexisting_quantity > CONST_DB_MIDDLE_NODES_LIMIT:
                    self._enable_db = False
                for tensor_i in self._broadcast_tensors - self._compute_inline_tensors:
                    dst_shape = util.shape_to_list(tensor_i.shape)
                    ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(tensor_i.dtype)
                    if dst_shape[-1] % ele_in_block != 0:
                        self._enable_db = False
                        break
                if enable_brc_inline():
                    self._tmp_ub_size += ONE_REPEAT_BYTES
                self._calc_tiling()
                self._calc_store_predicate()
        else:
            self._calc_store_predicate()
            self._calc_storage_bound()
            if self._is_store_align and self._ub_split_axis != len(self._out.shape) - 1:
                predicate_nodes = len(self._broadcast_store_predicate.union(self._store_predicate_common_tensors))
                if self._coexisting_quantity - predicate_nodes < 2:
                    self._coexisting_quantity += 1
            if enable_brc_inline():
                self._tmp_ub_size += ONE_REPEAT_BYTES
            self._calc_tiling()

    def _ahead_calc(self):
        # step 0: find all brc
        brc_nodes = GraphHelper.find_outermost_brc(self._out)
        # step 1: grouping
        groups_map = GraphHelper.brc_grouping(brc_nodes, self._max_dtype_bytes)
        GraphHelper.update_groups_by_out(groups_map, self._max_dtype_bytes, self._outs)
        groups = sorted(groups_map.items(), key=lambda x: x[0])
        # step 2: clac nodes
        self._calc_storage_bound()
        max_available_ub_bytes = (((self._ub_size - self._tmp_ub_size) // self._coexisting_quantity) //
                                  BLOCK_SIZE_BYTE) * BLOCK_SIZE_BYTE
        max_available_ub = max_available_ub_bytes // self._max_dtype_bytes
        is_only_last_brc = GraphHelper.only_last_brc(self._input_tensors)

        ahead_size = 0
        for src_size, brc_group in groups:
            before_brc_shape = util.shape_to_list(brc_group[0].op.input_tensors[0].shape)
            ele_in_block = BLOCK_SIZE_BYTE // self._max_dtype_bytes
            dst_shape = util.shape_to_list(brc_group[0].shape)
            # cond 1: last align
            is_last_align = before_brc_shape[-1] % ele_in_block == 0
            # cond 2: cut last axis, dividing by 2 may be too risky,
            # so take a conservative strategy get max_abailable_ub
            is_last_brc_cut_last = is_only_last_brc and before_brc_shape[-1] == 1 and \
                dst_shape[-1] > max_available_ub and not util.is_v220()
            if not (is_last_align or is_last_brc_cut_last):
                continue
            for tensor_i in brc_group:
                if tensor_i in self._compute_root_tensors:
                    continue
                is_out = tensor_i in self._outs and tensor_i not in self._broadcast_tensors
                tensor_root = tensor_i
                if not is_out:
                    tensor_root = tensor_i.op.input_tensors[0]
                all_nodes = GraphHelper.get_all_nodes(tensor_root)
                in_out_maps = GraphHelper.get_in_out_map(tensor_root)
                coexisting_quantity = GraphHelper.max_live_node(tensor_root, in_out_maps)
                max_dtype_bytes = max(DTYPE_BYTE_MAPPING.get(_tensor.dtype) for _tensor in all_nodes)
                cur_max_size = coexisting_quantity * src_size * max_dtype_bytes
                if ahead_size + cur_max_size <= max_available_ub_bytes:
                    self._compute_root_tensors.update(all_nodes)
                    if not is_out:
                        ahead_size += (src_size * DTYPE_BYTE_MAPPING.get(tensor_i.dtype))

        dst_shape = util.shape_to_list(self._out.shape)
        dst_size = reduce(mul, dst_shape, 1)
        if not self._compute_root_tensors or dst_size <= max_available_ub:
            self._tmp_ub_size = 0
            self._coexisting_quantity = 1
            return False

        self._tmp_ub_size = 0
        self._coexisting_quantity = 1
        self._calc_storage_bound()
        self._tmp_ub_size += ahead_size
        return True

    def _visit_before_broadcast_node(self):
        def dfs_graph(_out):
            for tensor_i in _out.op.input_tensors:
                if tensor_i in visited_tensors:
                    continue
                visited_tensors.add(tensor_i)
                self._before_broadcast_tensors.add(tensor_i)
                dfs_graph(tensor_i)

        out_brc = GraphHelper.find_outermost_brc(self._out)
        visited_tensors = set()
        for out in out_brc:
            dfs_graph(out)

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
            if all(util.support_scalar(tensor_o) for tensor_o in self._in_out_map.get(tensor_)):
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

        def _pre_handle_placeholder():
            for index, out in enumerate(self._outs):
                if util.is_placeholder(out):
                    copy_out = _copy_node(out)
                    self._outs[index] = copy_out
                    self._copy_out_tensors.add(copy_out)
                    self._out_tensors.add(copy_out)
                    self._out_tensors.remove(out)

        self._out_tensors = set(self._outs)
        _pre_handle_placeholder()

        visited_tensors = set()
        for out in self._out_tensors:
            if util.is_broadcast(out):
                self._broadcast_tensors.add(out)
            self.__dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)
            self._outs_dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING.get(dtype) for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)
        # out uint1 dtype needs ub_factor align to 256
        if "uint1" in self._outs_dtypes:
            self._ub_factor_align = max(self._ub_factor_align, SPECIAL_FACTOR_ALIGN)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors
        self._middle_out_tensors = self._middle_tensors.intersection(
            self._out_tensors)

        pure_out_tensors = list(self._out_tensors - self._middle_out_tensors)
        if len(pure_out_tensors) > 1:
            self._out = _fake_node(pure_out_tensors)
            self.__dfs_sub_graph(self._out, visited_tensors)
        else:
            self._out = pure_out_tensors[0]

        for tensor_i in self._broadcast_tensors:
            if match_scalar_scene(tensor_i):
                self._absorbable_broadcast_tensors.add(tensor_i)

        if self._broadcast_tensors:
            self._max_brc_bytes = max(DTYPE_BYTE_MAPPING.get(tensor_i.dtype) for tensor_i in self._broadcast_tensors)

        ub_idx = self._ub_split_axis
        for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
            if util.get_dsl_insn(tensor_i) != BROADCAST:
                src_shapes = tensor_i.op.input_tensors[0].shape[ub_idx:]
                dst_shapes = tensor_i.shape[ub_idx:]
                self._broadcast_axis_num[tensor_i] = _no_broadcast(src_shapes, dst_shapes)

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)
        self._cache_read_tensors.update(self._middle_out_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(tensor_i, self._scope, self._in_out_map.get(tensor_i))
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._input_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_read_buffer_map[tensor_i] = buffer_tensor

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._out_tensors - self._copy_out_tensors)

    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[buffer_tensor] = tensor_i
            self._out_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_write_buffer_map[tensor_i] = \
                    buffer_tensor

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._pure_middle_tensors:
            sch[tensor_i].set_scope(self._scope)

    def _get_const_storage_bound(self):
        res = self._out
        output_shape = util.shape_to_list(res.shape)
        const_tmp_ub_size = 0
        const_coexisting_quantity = 1
        for i in range(len(output_shape) - 1, -1, -1):
            self._ub_split_axis = i
            self._tmp_ub_size = 0
            self._coexisting_quantity = 1
            self._need_do_block = True
            self._calc_store_predicate()
            self._need_do_block = False
            self._calc_storage_bound()
            const_tmp_ub_size = max(const_tmp_ub_size, self._tmp_ub_size)
            const_coexisting_quantity = max(const_coexisting_quantity, self._coexisting_quantity)
            self._broadcast_store_predicate.clear()
            self._all_pre_node_broadcast.clear()
            self._store_predicate_common_tensors.clear()

        self._tmp_ub_size = const_tmp_ub_size
        self._coexisting_quantity = const_coexisting_quantity
        self._ub_split_axis = 0

    def _calc_store_predicate(self):
        def _dfs_cur_tensor(tensor_i):
            for _tensor in tensor_i.op.input_tensors:
                all_pre_node.add(_tensor)
                _dfs_cur_tensor(_tensor)

        def _is_only_calc_one(tensor_i):
            if len(tensor_i.op.input_tensors) != 1:
                return False
            if self._tiling_strategy == TilingStrategy.ONE_CUT or \
                    (self._tiling_strategy == TilingStrategy.CONST and self._need_do_block):
                u_i = self._ub_split_axis
                src_shape = tensor_i.op.input_tensors[0].shape
                dst_shape = tensor_i.shape
                return not expr_equal(src_shape[u_i], dst_shape[u_i])
            return False

        def _has_ternary_insns():
            for tensor_i in self._out_tensors | self._pure_middle_tensors:
                insn = util.get_dsl_insn(tensor_i)
                if insn in TERNARY_INSNS:
                    return True
            return False

        if _has_ternary_insns():
            return

        for tensor_i in self._broadcast_tensors:
            if not _is_only_calc_one(tensor_i):
                continue
            cur_tensor = tensor_i
            pre_tensor = tensor_i
            all_pre_node = set()
            if tensor_i in self._absorbable_broadcast_tensors:
                cur_tensor = tensor_i.op.input_tensors[0]
                pre_tensor = cur_tensor
            elif tensor_i in self._remove_pad_map:
                cur_tensor = self._remove_pad_map.get(tensor_i)
                all_pre_node.add(tensor_i)
            self._broadcast_store_predicate.add(cur_tensor)
            _dfs_cur_tensor(pre_tensor)
            self._all_pre_node_broadcast.update(all_pre_node)

        disable_store_predicate = False
        for tensor_i in self._all_pre_node_broadcast:
            common_tensor = self._in_out_map.get(tensor_i) - \
                            (self._all_pre_node_broadcast | self._broadcast_store_predicate)
            if len(common_tensor) > 0:
                # common in multi output
                if tensor_i in self._out_tensors:
                    disable_store_predicate = True
                    break
                self._store_predicate_common_tensors.add(tensor_i)
        extend_node_num = len(self._broadcast_store_predicate) + len(self._store_predicate_common_tensors)
        if disable_store_predicate or extend_node_num > MAX_EXTEND_NODE_NUM:
            self._broadcast_store_predicate.clear()
            self._all_pre_node_broadcast.clear()
            self._store_predicate_common_tensors.clear()

    def _calc_storage_bound(self):
        pass

    def _calc_tiling(self):
        funcs = {TilingStrategy.ALL_CUT: self._calc_tiling_all_cut,
                 TilingStrategy.NONE_CUT: self._calc_tiling_none_cut,
                 TilingStrategy.ONE_CUT: self._calc_tiling_one_cut,
                 TilingStrategy.STATIC: self._calc_tiling_static,
                 TilingStrategy.CONST: self._calc_tiling_const,
                 }
        funcs.get(self._tiling_strategy)()

    def _calc_tiling_all_cut(self):
        res = self._out
        for _i, _x in enumerate(res.shape):
            bound = (1, util.get_bound(_x)[1])
            self._block_tiling_vars[_i] = operation.var_inner("_block_factor_" + str(_i), bound)
            self._ub_tiling_vars[_i] = operation.var_inner("_ub_factor_" + str(_i), bound)

    def _calc_tiling_none_cut(self):
        pass

    def _calc_tiling_one_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis
        u_i = self._ub_split_axis
        b_bound = (1, 2147483647)
        u_bound = self._tiling_case.ub_factor_bound
        if u_bound is None:
            u_bound = (1, util.get_bound(shape[u_i])[1])
        self._block_tiling_vars[b_i] = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = operation.var_inner("_ub_factor_" + str(u_i), u_bound)
        self._block_factor = self._block_tiling_vars.get(b_i)
        self._ub_factor = self._ub_tiling_vars.get(u_i)

    def _calc_tiling_static(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis
        u_i = self._ub_split_axis
        b_bound = (1, util.get_bound(shape[b_i])[1])
        self._block_tiling_vars[b_i] = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = self._tiling_case.ub_factor_bound
        self._block_factor = self._block_tiling_vars.get(b_i)
        self._ub_factor = self._ub_tiling_vars.get(u_i)

    def _calc_tiling_const(self):
        def _get_const_tiling():
            input_shapes = []
            inputs = []
            max_dim_length = len(output_shape)
            for _input in self._input_tensors:
                input_shape = util.shape_to_list(_input.shape)
                input_shapes.append([1] * (max_dim_length - len(input_shape)) + input_shape)
                inputs.append({"shape": input_shape, "dtype": _input.dtype})
            outputs = [{"shape": output_shape, "dtype": res.dtype}]
            if not inputs:
                inputs = copy.deepcopy(outputs)
                max_dim_length = 0

            input_shapes = list(map(list, zip(*input_shapes)))
            for i in range(max_dim_length - 1, -1, -1):
                if any(input_shapes[i][0] != s for s in input_shapes[i]):
                    broadcast_axis[i] = True

            max_available_ub = ((((self._ub_size - self._tmp_ub_size) // self._coexisting_quantity) //
                                 BLOCK_SIZE_BYTE) * BLOCK_SIZE_BYTE) // self._max_dtype_bytes
            max_available_ub_db = ((((self._ub_size - 2 * self._tmp_ub_size) // 2 // self._coexisting_quantity) //
                                    BLOCK_SIZE_BYTE) * BLOCK_SIZE_BYTE) // self._max_dtype_bytes
            if self._enable_db:
                max_available_ub = max_available_ub_db
            base_info = {"000": [util.get_core_num(), self._max_dtype_bytes,
                                 max_available_ub, max_available_ub_db, self._max_brc_bytes]}

            only_const_tiling = True
            is_const_shapes = False
            support_broadcast = operation.get_context().get("_support_broadcast")
            use_special_pattern = False
            const_compile_info = {
                CompileInfo.FLAG_INFO: [only_const_tiling, is_const_shapes, support_broadcast, use_special_pattern],
                CompileInfo.BASE_INFO: base_info,
                CompileInfo.SOC_VERSION: get_soc_spec(SHORT_SOC_VERSION),
                CompileInfo.BROADCAST_AXIS: broadcast_axis,
                CompileInfo.UB_FACTOR_ALIGN: self._ub_factor_align,
                CompileInfo.CLASSIFY_INPUTS_NUM: operation.get_context().get("_classify_inputs_num"),
                CompileInfo.IS_VNCHWCONV_ALIGN: self._is_vnchwconv_align
            }
            const_compile_info.update(get_compile_info())

            op_type = "AutoTiling"
            return op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)

        res = self._out
        output_shape = util.shape_to_list(res.shape)
        broadcast_axis = [False] * len(output_shape)
        if output_shape == [0]:
            self._block_dims = 1
            self._need_do_block = False
            return

        run_info = _get_const_tiling()

        tiling_format = {
            "need_tiling_cut": "int",
            "is_store_align": "int",
            "block_axis": "int",
            "block_factor": "int",
            "ub_axis": "int",
            "ub_factor": "int",
            "is_need_db": "int"}
        tiling_data = op_tiling.decode(run_info.get('tiling_data'), tiling_format)
        self._block_dims = run_info.get("block_dim")
        self._need_do_block = True if tiling_data.get("need_tiling_cut") > 0 else False
        self._is_store_align = True if tiling_data.get("is_store_align") > 0 else False
        if self._need_do_block:
            self._block_split_axis = tiling_data.get("block_axis")
            self._block_factor = tiling_data.get("block_factor")
            self._ub_split_axis = tiling_data.get("ub_axis")
            self._ub_factor = tiling_data.get("ub_factor")
            if self._ub_factor == 1 and self._ub_split_axis < len(output_shape) - 1 and \
                    not broadcast_axis[self._ub_split_axis]:
                self._ub_split_axis += 1
                self._ub_factor = output_shape[self._ub_split_axis]
            avg_num = (output_shape[self._ub_split_axis] + self._ub_factor - 1) // self._ub_factor
            new_factor = (output_shape[self._ub_split_axis] + avg_num - 1) // avg_num
            ele_in_block = BLOCK_SIZE_BYTE // self._max_dtype_bytes
            if self._is_one_dim:
                ele_in_block = self._ub_factor_align
            new_factor = (new_factor + ele_in_block - 1) // ele_in_block * ele_in_block
            self._ub_factor = min(self._ub_factor, new_factor)

        self._enable_db = self._enable_db or (True if tiling_data.get("is_need_db", 0) == 1 else False)

    def _is_last_align(self, _tensor):
        src_shapes = util.shape_to_list(_tensor.op.input_tensors[0].shape)
        ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(_tensor.dtype)
        if isinstance(src_shapes[-1], int) and src_shapes[-1] % ele_in_block == 0 and \
                _tensor not in self._out_tensors:
            use_tensors = list(self._in_out_map.get(_tensor, []))
            for u_tensor in use_tensors:
                if util.is_vcmp_insn(u_tensor) or util.is_vsel_insn(u_tensor) or \
                        util.is_vcmpsel_insn(u_tensor) or util.is_ternary_insn(u_tensor):
                    return False
            return True
        return False

    def _calc_compute_inline(self):
        def no_broadcast(_src_shapes, _dst_shapes):
            _src_shapes = util.shape_to_list(_src_shapes)
            _dst_shapes = util.shape_to_list(_dst_shapes)
            for x, y in zip(_src_shapes, _dst_shapes):
                if not expr_equal(x, y):
                    return False
            return True

        def update_store_predicate(_tensor):
            if _tensor in self._broadcast_store_predicate:
                self._broadcast_store_predicate.remove(_tensor)
                self._all_pre_node_broadcast.remove(_tensor.op.input_tensors[0])
                self._broadcast_store_predicate.add(_tensor.op.input_tensors[0])

        # ternary instruct out must be memory reused with input, so ternary instruct input can not do compute_inline
        def is_ternary_ins_input(_tensor):
            if _tensor not in self._in_out_map.keys():
                return False
            for tensor_out in self._in_out_map.get(_tensor):
                tensor_out_insn = util.get_dsl_insn(tensor_out)
                if tensor_out_insn in TERNARY_INSNS:
                    return True
            return False

        def is_const_compute_inline(_tensor):
            if util.get_dsl_insn(_tensor) == BROADCAST or is_ternary_ins_input(_tensor):
                return False
            src_shapes = _tensor.op.input_tensors[0].shape[ub_idx:]
            dst_shapes = _tensor.shape[ub_idx:]
            if no_broadcast(src_shapes, dst_shapes):
                return True
            return False

        def is_dynamic_compute_inline(_tensor):
            return self._broadcast_axis_num.get(_tensor) == 0 \
                   and not is_ternary_ins_input(_tensor)

        self._compute_inline_tensors = self._absorbable_broadcast_tensors.copy()
        if self._tiling_strategy == TilingStrategy.CONST:
            ub_idx = self._ub_split_axis
            for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
                if is_const_compute_inline(tensor_i):
                    update_store_predicate(tensor_i)
                    self._compute_inline_tensors.add(self._get_ub_tensor(tensor_i))
        else:
            for tensor_i in self._broadcast_axis_num:
                if is_dynamic_compute_inline(tensor_i):
                    self._compute_inline_tensors.add(self._get_ub_tensor(tensor_i))

        for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
            if not tensor_i.op.input_tensors:
                continue
            if self._is_last_align(tensor_i):
                update_store_predicate(tensor_i)
                self._compute_inline_tensors.add(self._get_ub_tensor(tensor_i))
                use_tensors = set(self._in_out_map.get(tensor_i, set()))
                self._const_brc_inline_tensor.update(use_tensors)

        if self._is_pure_brc_common_db:
            for tensor_i in self._broadcast_tensors:
                if tensor_i in self._out_tensor_map:
                    self._compute_inline_tensors.add(self._out_tensor_map.get(tensor_i))
                else:
                    self._compute_inline_tensors.add(tensor_i)

        self.__calc_tensor_space()

        for tensor_i in self._broadcast_tensors - self._compute_inline_tensors:
            need_update_var_range = False
            if util.get_dsl_insn(tensor_i) != BROADCAST:
                ub_under_shapes = util.shape_to_list(tensor_i.op.input_tensors[0].shape[self._ub_split_axis + 1:])
                ub_under_size = reduce(lambda x, y: x * y, ub_under_shapes or [1])
                if (self._min_storage_bound // ub_under_size) == 1 and isinstance(self._ub_factor, tvm.expr.Var):
                    self._compute_inline_tensors.add(self._get_ub_tensor(tensor_i))
                    update_store_predicate(tensor_i)
                    need_update_var_range = True
            if need_update_var_range:
                self._constraints.add(tvm.expr.EQ(self._ub_factor, 1))
                self._ub_factor_is_one = False
                out_ub_shapes = util.shape_to_list(self._out.shape[self._ub_split_axis + 1:])
                for in_var, out_var in zip(ub_under_shapes, out_ub_shapes):
                    if isinstance(out_var, tvm.expr.Var):
                        self._constraints.add(tvm.expr.EQ(out_var, in_var))

    def _calc_storage_align(self):
        if not self._is_store_align:
            return

        for tensor_i in self._cache_read_buffer_tensor_map:
            self._store_align_tensors.add(tensor_i)

        for tensor_i in self._middle_tensors - self._compute_inline_tensors:
            self._store_align_tensors.add(tensor_i)

        for tensor_i in set(self._cache_write_buffer_tensor_map.keys()) - self._compute_inline_tensors:
            self._store_align_tensors.add(tensor_i)

    def _calc_multi_core(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            self._block_bind_axis = None

    def _calc_remove_pad(self):
        pass

    def _calc_ub_align(self):
        pass

    def _calc_compute_at(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT or \
                (self._tiling_strategy == TilingStrategy.CONST and not self._need_do_block):
            self._compute_at_map.clear()
            return

        tmp_compute_at_map = {}
        for tensor_i in self._input_tensors:
            tmp_compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._middle_tensors - self._compute_inline_tensors:
            tmp_compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_read_buffer_tensor_map:
            tmp_compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_write_buffer_tensor_map:
            if tensor_i not in self._compute_inline_tensors:
                tmp_compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        before_ub_tensor = self._get_all_ub_tensors(self._compute_root_tensors)
        for tensor_i, value in tmp_compute_at_map.items():
            if not (tensor_i in before_ub_tensor or tensor_i in self._compute_root_tensors):
                self._compute_at_map[tensor_i] = value

    def _calc_double_buffer(self):
        pass

    def _calc_mem_reuse(self):
        ternary_reuse_map = {
            "elewise_binary_scalar_axpy": 1,
            "elewise_multiple_mla": 2,
            "elewise_multiple_madd": 1,
            "elewise_multiple_maddrelu": 1,
        }

        # one of the input of the ternary instruction must be reused with the output, refer to "ternary_reuse_map"
        # consider "vmadd": A=A*A+B, output reuse the second A, input_tensors is 2, which need to be completed to 3
        for tensor_i in self._out_tensors | (self._pure_middle_tensors - self._compute_inline_tensors):
            insn = util.get_dsl_insn(tensor_i)
            args = ""
            if tensor_i.op.tag.find("|") != -1:
                args = tensor_i.op.tag.split("|")[1].split(",")
            if insn in TERNARY_INSNS:
                src_tensors = []
                index = 0
                first_same_tensor = None
                for i in args:
                    if i == "1":
                        if first_same_tensor not in src_tensors:
                            first_same_tensor = tensor_i.op.input_tensors[index]
                            index += 1
                        src_tensors.append(first_same_tensor)
                    else:
                        src_tensors.append(tensor_i.op.input_tensors[index])
                        index += 1
                reuse_index = ternary_reuse_map.get(insn)
                src_tensor = src_tensors[reuse_index]
                src_tensor = self._get_ub_tensor(src_tensor)
                if src_tensor in self._remove_pad_map:
                    src_tensor = self._remove_pad_map.get(src_tensor)
                dst_tensor = self._get_ub_tensor(tensor_i)
                util.merge_value(self._mem_reuse_map,
                                 src_tensor,
                                 dst_tensor)
        for tensor_i, write_buffer in self._middle_out_cache_write_buffer_map.items():
            util.merge_value(self._mem_reuse_map,
                             self._middle_out_cache_read_buffer_map.get(tensor_i),
                             write_buffer)

    def _calc_constraints(self):
        def add_condition(condition):
            if isinstance(condition, tvm.expr.Expr):
                self._constraints.add(condition)

        for tensor_i in self._broadcast_tensors:
            if util.is_unknown_broadcast(tensor_i):
                src_shapes = util.shape_to_list(tensor_i.op.input_tensors[0].shape)
                dst_shapes = util.shape_to_list(tensor_i.shape)
                for src_shape, dst_shape in zip(src_shapes, dst_shapes):
                    if not expr_equal(src_shape, dst_shape):
                        add_condition(src_shape <= dst_shape)

        shapes = util.shape_to_list(self._out.shape)
        ub_shapes = shapes[self._ub_split_axis + 1:]
        for shape in ub_shapes:
            add_condition(shape <= self._min_storage_bound)
        if self._tiling_strategy != TilingStrategy.NONE_CUT:
            ub_shapes.insert(0, self._ub_factor)
            shape_size = reduce(lambda x, y: x * y, ub_shapes)
            add_condition(shape_size <= self._min_storage_bound)
            add_condition(self._ub_factor <= self._min_storage_bound)

    def _calc_emit_insn(self):
        def get_insn(tensor_):
            tag = tensor_.op.tag
            if tensor_.op.tag.find("|") != -1:
                insn = tag.split("|")[0]
            else:
                insn = tag
            return INSN_MAPPING.get(insn, insn)

        for source, target in self._cache_read_buffer_tensor_map.items():
            if target in self._middle_out_tensors:
                self._emit_insn_map[source] = [source.op.axis[0], PHONY_INSN]
            else:
                self._emit_insn_map[source] = [source.op.axis[0], DMA_COPY]

        for tensor_i in self._pure_middle_tensors - self._compute_inline_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], get_insn(tensor_i)]

        for source, target in self._cache_write_buffer_tensor_map.items() - self._compute_inline_tensors:
            self._emit_insn_map[source] = [source.op.axis[0], get_insn(target)]

        if len(self._out_tensors) > 1:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], DMA_COPY]
            if len(self._out_tensors) - len(self._middle_out_tensors) > 1:
                self._emit_insn_map[self._out] = [self._emit_insn_axis, PHONY_INSN]
            else:
                self._emit_insn_map[self._out] = [self._emit_insn_axis, DMA_COPY]
        else:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [self._emit_insn_axis, DMA_COPY]

    def _do_tiling(self):
        funcs = {TilingStrategy.ALL_CUT: self._do_tiling_all_cut,
                 TilingStrategy.NONE_CUT: self._do_tiling_none_cut,
                 TilingStrategy.ONE_CUT: self._do_tiling_one_cut,
                 TilingStrategy.STATIC: self._do_tiling_static,
                 TilingStrategy.CONST: self._do_tiling_const,
                 }
        funcs.get(self._tiling_strategy)()

    def _do_tiling_all_cut(self):
        sch = self._schedule
        res = self._out
        block_axes = []
        ub_axes = []
        inner_axes = []
        for _i, _x in enumerate(res.op.axis):
            x_o, x_i = sch[res].split(_x, factor=self._block_tiling_vars.get(_i))
            x_io, x_ii = sch[res].split(x_i, factor=self._ub_tiling_vars.get(_i))
            block_axes.append([x_o, _i])
            ub_axes.append([x_io, _i])
            inner_axes.append([x_ii, _i])
            self._inner_shape.append([self._ub_tiling_vars.get(_i), _i])
        ir_axes = block_axes + ub_axes + inner_axes
        ordered_axes = [x[0] for x in ir_axes]
        sch[res].reorder(*ordered_axes)
        self._block_bind_axis = sch[res].fuse(*[x[0] for x in block_axes])
        self._compute_at_axis.compute_at_axis = ub_axes[-1][0]
        self._compute_at_axis_idx = ub_axes[-1][1]
        self._emit_insn_axis.emit_insn_axis = inner_axes[0][0]

    def _do_tiling_none_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        for _i, _x in enumerate(res.op.axis):
            self._inner_shape.append([shape[_i], _i])
        self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_tiling_one_cut(self):
        sch = self._schedule
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_idx = self._block_split_axis
        u_idx = self._ub_split_axis
        block_axes = []
        ub_axes = []
        inner_axes = []

        # 1. split ub first
        u_o, u_i = sch[res].split(res.op.axis[u_idx], factor=self._ub_tiling_vars.get(u_idx))
        ub_axes.append([u_o, u_idx])
        inner_axes.append([u_i, u_idx])
        self._inner_shape.append([self._ub_tiling_vars[u_idx], u_idx])
        for i in range(u_idx):
            block_axes.append([res.op.axis[i], i])
        block_axes.append([u_o, u_idx])

        # 2. fuse block axis
        block_fuse_axis = sch[res].fuse(*[x[0] for x in block_axes])

        # 3. spilt block
        b_o, b_i = sch[res].split(block_fuse_axis, factor=self._block_tiling_vars.get(b_idx))

        for i in range(u_idx + 1, len(res.op.axis)):
            inner_axes.append([res.op.axis[i], i])
            self._inner_shape.append([shape[i], i])

        self._block_bind_axis = b_o
        self._compute_at_axis.compute_at_axis = b_i
        self._compute_at_axis_idx = u_idx
        self._emit_insn_axis.emit_insn_axis = inner_axes[0][0]

    def _do_tiling_static(self):
        self._do_tiling_one_cut()

    def _do_tiling_const(self):
        sch = self._schedule
        res = self._out
        shape = util.shape_to_list(res.shape)
        block_axes = []
        ub_axes = []
        inner_axes = []
        b_idx = self._block_split_axis
        u_idx = self._ub_split_axis

        if self._need_do_block:
            # 1. split ub first
            u_o, u_i = sch[res].split(res.op.axis[u_idx], factor=self._ub_factor)
            ub_axes.append([u_o, u_idx])
            inner_axes.append([u_i, u_idx])
            self._inner_shape.append([self._ub_factor, u_idx])
            for i in range(u_idx):
                block_axes.append([res.op.axis[i], i])
            block_axes.append([u_o, u_idx])

            # 2. fuse block axis
            block_fuse_axis = sch[res].fuse(*[x[0] for x in block_axes])

            # 3. spilt block
            b_o, b_i = sch[res].split(block_fuse_axis, factor=self._block_factor)

            for i in range(u_idx + 1, len(res.op.axis)):
                inner_axes.append([res.op.axis[i], i])
                self._inner_shape.append([shape[i], i])

            self._block_bind_axis = b_o
            self._compute_at_axis.compute_at_axis = b_i
            self._compute_at_axis_idx = u_idx
            self._emit_insn_axis.emit_insn_axis = inner_axes[0][0]
        else:
            self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_storage_bound(self):

        def calc_min_bound():
            min_bound = 0
            if const_set_storage_bound:
                for tensor_i in self._broadcast_tensors - self._compute_inline_tensors:
                    dst_shape = util.shape_to_list(tensor_i.shape)
                    if self._need_do_block:
                        dst_shape[self._ub_split_axis] = 1 if dst_shape[-1] == 1 else self._ub_factor
                    if dst_shape[-1] != 1:
                        dst_shape[-1] = (dst_shape[-1] + ele_in_block - 1) // ele_in_block * ele_in_block
                    storage_bound = reduce(mul, dst_shape[self._ub_split_axis:], 1)
                    brc_size.add(storage_bound)
                if not brc_size:
                    return min_bound

                min_bound = min(ROW_LIMIT * ele_in_block * ele_in_block,
                                int(self._tensor_space // self._max_dtype_bytes))
                for _tensor in self._broadcast_tensors - self._compute_inline_tensors:
                    if not _tensor.op.input_tensors:
                        continue
                    s_shape = util.shape_to_list(_tensor.op.input_tensors[0].shape)
                    d_shape = util.shape_to_list(_tensor.shape)
                    if s_shape[-1] == 1 and d_shape[-1] != 1:
                        if s_shape[self._ub_split_axis] != 1:
                            s_shape[self._ub_split_axis] = self._ub_factor
                        shape_size = reduce(mul, s_shape[self._ub_split_axis:], 1)
                        row_size = ROW_LIMIT * ele_in_block
                        if self._is_vnchwconv_align:
                            row_factor = (shape_size + row_size - 1) // row_size
                        else:
                            row_factor = shape_size // row_size
                        min_bound = max(row_factor * row_size * ele_in_block, min_bound)
            return min_bound

        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        output_shape = util.shape_to_list(self._out.shape)
        ele_in_block = BLOCK_SIZE_BYTE // self._max_brc_bytes
        const_set_storage_bound = self._tiling_strategy == TilingStrategy.CONST and \
            (output_shape[-1] % ele_in_block == 0 or self._is_store_align)
        brc_size = set()
        min_bound = calc_min_bound()

        before_ub_tensor = self._get_all_ub_tensors(self._compute_root_tensors)
        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING.get(tensor_i.dtype))
            if tensor_i in before_ub_tensor or tensor_i in self._compute_root_tensors:
                continue
            if const_set_storage_bound:
                dst_shape = util.shape_to_list(tensor_i.shape)
                use_tensor = list(self._in_out_map.get(self._get_ori_tensor(tensor_i), []))
                if len(use_tensor) == 1 and self._broadcast_by_no_other_use.get(self._get_ori_tensor(use_tensor[0])):
                    dst_shape = util.shape_to_list(use_tensor[0].shape)
                ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(tensor_i.dtype)
                if self._need_do_block:
                    dst_shape[self._ub_split_axis] = 1 if dst_shape[self._ub_split_axis] == 1 else self._ub_factor
                if dst_shape[-1] != 1:
                    dst_shape[-1] = (dst_shape[-1] + ele_in_block - 1) // ele_in_block * ele_in_block
                real_storage_bound = reduce(mul, dst_shape[self._ub_split_axis:], 1)
                if real_storage_bound in brc_size and real_storage_bound < min_bound:
                    real_storage_bound = min_bound
                if real_storage_bound % ele_in_block != 0:
                    real_storage_bound = (real_storage_bound + ele_in_block - 1) // ele_in_block * ele_in_block
                if self._brc_avoid_bank_conflict:
                    extent_size = int((ROW_LIMIT * BLOCK_SIZE_BYTE) // DTYPE_BYTE_MAPPING.get(tensor_i.dtype))
                    real_storage_bound += extent_size
                storage_bound = min(storage_bound, int(real_storage_bound))
            sch[tensor_i].set_buffer_size(storage_bound)

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()

    def _do_storage_align(self):
        sch = self._schedule
        before_ub_tensor = self._get_all_ub_tensors(self._before_broadcast_tensors)
        for tensor_i in self._store_align_tensors:
            align_factor = int(BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(tensor_i.dtype))
            src_shape = util.shape_to_list(tensor_i.shape)
            if len(src_shape) - self._ub_split_axis == 1:
                continue
            if tensor_i in self._before_broadcast_tensors or tensor_i in before_ub_tensor:
                if src_shape[-1] == 1:
                    continue
                align_factor = tvm.select(src_shape[-1] == 1, 1, align_factor)
                if self._get_ori_tensor(tensor_i) not in self._input_tensors:
                    self._axis_group_tensors.add(tensor_i)
            sch[tensor_i].storage_align(tensor_i.op.axis[-2], align_factor, 0)

    def _do_axis_group(self):
        sch = self._schedule
        group_id = tvm.make.Call("int32", "axis_group", [0, "append"], tvm.expr.Call.Extern, None, 0)
        for tensor_i in self._axis_group_tensors:
            for axis_id in range(self._ub_split_axis, len(tensor_i.shape)):
                sch[tensor_i].pragma(tensor_i.op.axis[axis_id], "axis_group", group_id)

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _do_ub_align(self):
        pass

    def _do_remove_pad(self):
        pass

    def _do_compute_at(self):
        sch = self._schedule
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1].compute_at_axis)

    def _do_store_predicate(self):

        def calc_predicate_condition(tensor, ub_split_src):
            u_idx = self._ub_split_axis
            b_idx = self._block_split_axis
            ub_factor = None
            block_factor = None
            ub_out_shape = None

            if self._tiling_strategy == TilingStrategy.CONST:
                ub_factor = self._ub_factor
                block_factor = self._block_factor
            else:
                ub_factor = self._ub_tiling_vars.get(u_idx)
                block_factor = self._block_tiling_vars.get(b_idx)
            ub_out_shape = tvm.floordiv(self._out.shape[u_idx] - 1, ub_factor) + 1

            if self._enable_db:
                return tvm.any(
                    (self._block_bind_axis * block_factor + self._compute_at_axis.compute_at_axis) % ub_out_shape < 2,
                    self._compute_at_axis.compute_at_axis < 2,
                    ub_split_src != 1)
            return tvm.any(
                (self._block_bind_axis * block_factor + self._compute_at_axis.compute_at_axis) % ub_out_shape < 1,
                self._compute_at_axis.compute_at_axis < 1,
                ub_split_src != 1)

        sch = self._schedule
        u_idx = self._ub_split_axis
        for tensor_i in self._broadcast_store_predicate:
            input_tensors = tensor_i.op.input_tensors
            is_vreduce_tensor = len(input_tensors) > 0 and util.is_broadcast(input_tensors[0]) \
                and tensor_i in self._remove_pad_cache_read_buffer
            if util.is_broadcast(tensor_i):
                ub_split_src = tensor_i.op.input_tensors[0].shape[u_idx]
            elif is_vreduce_tensor:
                ub_split_src = tensor_i.op.input_tensors[0].op.input_tensors[0].shape[u_idx]
            else:
                ub_split_src = tensor_i.shape[u_idx]
            cond = calc_predicate_condition(tensor_i, ub_split_src)
            sch[self._get_ub_tensor(tensor_i)].set_store_predicate(cond)
            sch[self._get_ub_tensor(tensor_i)].mem_unique()
        for tensor_i in self._all_pre_node_broadcast:
            if util.is_broadcast(tensor_i) and not util.is_scalar_broadcast(tensor_i):
                ub_split_src = tensor_i.op.input_tensors[0].shape[u_idx]
            else:
                ub_split_src = tensor_i.shape[u_idx]
            cond = calc_predicate_condition(tensor_i, ub_split_src)
            sch[self._get_ub_tensor(tensor_i)].set_store_predicate(cond)
        for tensor_i in self._store_predicate_common_tensors:
            if tensor_i in self._input_tensor_map:
                sch[self._input_tensor_map.get(tensor_i)].mem_unique()
            else:
                sch[tensor_i].mem_unique()

    def _do_double_buffer(self):
        if self._enable_db:
            sch = self._schedule

            tensors = self._pure_middle_tensors \
                .union(self._cache_read_buffer_tensor_map.keys()) \
                .union(self._cache_write_buffer_tensor_map.keys())

            before_ub_tensor = self._get_all_ub_tensors(self._compute_root_tensors)
            for tensor_i in tensors:
                if not (tensor_i in before_ub_tensor or tensor_i in self._compute_root_tensors):
                    sch[tensor_i].double_buffer()

    def _do_mem_reuse(self):
        sch = self._schedule
        for _a, _b in self._mem_reuse_map.items():
            for b_i in _b:
                sch[_a].reused_by(b_i)

    def _do_constraints(self):
        sch = self._schedule
        for cond in self._constraints:
            sch.set_constraint(cond)

    def _get_ori_tensor(self, tensor_i):
        if tensor_i in self._cache_read_buffer_tensor_map:
            return self._cache_read_buffer_tensor_map.get(tensor_i)
        if tensor_i in self._cache_write_buffer_tensor_map:
            return self._cache_write_buffer_tensor_map.get(tensor_i)
        return tensor_i

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            emit_insn_axis = param[0]
            if isinstance(emit_insn_axis, self.EmitInsn):
                emit_insn_axis = emit_insn_axis.emit_insn_axis
            if len(param) > 2:
                sch[tensor_i].emit_insn(emit_insn_axis, param[2])
            attrs = {}
            tensor_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING.get(tensor_i.dtype))
            ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(tensor_i.dtype)
            last_dim_threshold = int(tensor_bound // (ele_in_block * ROW_LIMIT) / ele_in_block * ele_in_block)
            if param[1] in (VECTOR_BROADCAST, UNKNOWN_BROADCAST):
                if param[1] == UNKNOWN_BROADCAST:
                    if self._tiling_strategy == TilingStrategy.NONE_CUT:
                        src_shapes = tensor_i.op.input_tensors[0].shape
                    else:
                        u_idx = self._ub_split_axis
                        src_shapes = tensor_i.op.input_tensors[0].shape[u_idx + 1:]
                        if self._ub_factor_is_one:
                            src_shapes.append(1)
                        else:
                            src_shapes.append(self._ub_factor)
                    is_all_const = all(isinstance(s, int) for s in util.shape_to_list(src_shapes))
                    if is_all_const:
                        param[1] = VECTOR_BROADCAST
                attrs = {"dynamic_fuse": False,
                         "dynamic_split": False}
                if not util.is_v220():
                    attrs["enable_align_broadcast"] = True
                if self._is_vnchwconv_align:
                    attrs["enough_buffer"] = True
                if self._broadcast_by_no_other_use.get(self._get_ori_tensor(tensor_i)):
                    attrs["reuse_src_tensor"] = True
                if not self._is_store_align:
                    attrs["fuse_axis_threshold"] = last_dim_threshold
                if self._brc_avoid_bank_conflict:
                    attrs["avoid_bank_conflict"] = True
                attrs["last_dup_threshold"] = BRC_LAST_DUP_LIMIT
                if tensor_i in self._compute_align_map:
                    attrs["last_src_valid_element"] = self._compute_align_map.get(tensor_i)[1]
            elif tensor_i in self._out_tensors:
                if self._is_one_dim:
                    attrs = {"no_overlap": 0}
                elif self._is_pure_brc_common_db:
                    attrs = {"no_overlap": LAST_LOOP_REG_MOV}
                elif self._is_store_align and self._ub_split_axis != len(self._out.shape) - 1:
                    attrs = {"no_overlap": MISSALIGN_STRIDE_WHITH_MALLOC_BUF, "no_overlap_malloc_buf_for_tail": 1}
                    out_ub_tensor = self._out_tensor_map.get(tensor_i)
                    none_reuse_tensors = self._broadcast_store_predicate.union(self._store_predicate_common_tensors)
                    none_reuse_ub_tensors = self._get_all_ub_tensors(none_reuse_tensors)
                    predicate_nodes = len(none_reuse_tensors)
                    real_coexisting_quantity = self._coexisting_quantity - predicate_nodes
                    if real_coexisting_quantity < LAST_LOOP_COEXISTING_QUANTITY_LIMIT:
                        attrs = {"no_overlap": LAST_LOOP_REG_MOV}
                    elif real_coexisting_quantity < MISSALIGN_STRIDE_COEXISTING_QUANTITY_LIMIT and \
                            out_ub_tensor not in none_reuse_ub_tensors:
                        attrs = {"no_overlap": MISSALIGN_STRIDE_WHITH_MALLOC_BUF, "no_overlap_malloc_buf_for_tail": 0}
            if param[1] == "vector_dup":
                attrs = {"trans_assign_opt" : True}
            if tensor_i in self._const_brc_inline_tensor:
                attrs["use_ba_pattern_brc"] = True
            sch[tensor_i].emit_insn(emit_insn_axis, param[1], attrs)

    def _add_compile_info(self):
        before_node_nums = operation.get_context().get("_node_nums") or 0
        current_node_nums = max(len(self._middle_tensors), before_node_nums)
        operation.get_context().add("_node_nums", current_node_nums)
        cpt_compute = operation.get_context().get_current_compute()
        cpt_schedule = cpt_compute.get_current_schedule()
        if self._mode == CONST:
            # const shape: one compute, one schedule
            cpt_compute.add("_const_block_dim", self._block_dims)
        else:
            cpt_schedule.add(CompileInfo.MAX_DTYPE, self._max_dtype_bytes)
            cpt_schedule.add(CompileInfo.COEXISTING_QUANTITY, self._coexisting_quantity)
            cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
            cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())
            cpt_schedule.add("_tiling_key", self._schedule.tiling_key)

        operation.add_compile_info_inner(CompileInfo.UB_FACTOR_ALIGN, self._ub_factor_align)
        operation.add_compile_info_inner(CompileInfo.IS_VNCHWCONV_ALIGN, self._is_vnchwconv_align)
        cpt_schedule.add(CompileInfo.MAX_BRC_TYPE, self._max_brc_bytes)

    def _check_tiling_case(self):
        if self._tiling_strategy == TilingStrategy.CONST:
            return True
        lower_bound = 1
        under_ub_len = len(self._inner_shape)
        ele_in_block = int(BLOCK_SIZE_BYTE // self._max_dtype_bytes)
        for index, item in enumerate(self._inner_shape[::-1]):
            cur_bound = util.get_bound(item[0])[0]
            if cur_bound is None:
                return False
            if index == 0 and under_ub_len != 1 and cur_bound % ele_in_block != 0:
                cur_bound = (cur_bound + ele_in_block - 1) // ele_in_block * ele_in_block
            if index == 0 and under_ub_len != 1 and not self._is_store_align and not util.is_v220():
                tensor_bound = int(self._tensor_space // self._max_dtype_bytes)
                last_dim_threshold = int(tensor_bound // (ele_in_block * ROW_LIMIT) / ele_in_block * ele_in_block)
                l_bound, u_bound = util.get_bound(item[0])
                l_bound_align = (l_bound + ele_in_block - 1) // ele_in_block * ele_in_block
                if l_bound > last_dim_threshold and l_bound_align > u_bound:
                    return False
            lower_bound *= cur_bound
        if not self._tensor_space // self._max_dtype_bytes >= lower_bound:
            return False
        return True

    def __calc_tensor_space(self):
        # minus tmp size
        self._correct_factor = 2 if self._enable_db else 1
        self._ub_size -= self._correct_factor * self._tmp_ub_size
        tensor_space = self._ub_size // self._coexisting_quantity
        if self._enable_db:
            tensor_space = self._ub_size // 2 // self._coexisting_quantity
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        # adjust storage bound by tiling handle one dim (128 align)
        if self._is_one_dim and self._tensor_space > ONE_DIM_ALIGN and get_soc_spec(SHORT_SOC_VERSION) != ASCEND_910B:
            self._tensor_space = self._tensor_space // ONE_DIM_ALIGN * ONE_DIM_ALIGN

        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        min_storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING.get(self._out.dtype))
        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING.get(tensor_i.dtype))
            if storage_bound < min_storage_bound:
                min_storage_bound = storage_bound
        self._min_storage_bound = min_storage_bound

    def __dfs_sub_graph(self, out, visited_tensors: set):
        for tensor_i in out.op.input_tensors:
            util.merge_value(self._in_out_map, tensor_i, out)
            self._dtypes.add(tensor_i.dtype)

            if util.is_placeholder(tensor_i):
                self._input_tensors.add(tensor_i)
            else:
                self._middle_tensors.add(tensor_i)
                if util.is_broadcast(tensor_i):
                    self._broadcast_tensors.add(tensor_i)

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self.__dfs_sub_graph(tensor_i, visited_tensors)

    def _calc_set_value(self):
        if operation.get_context().get_current_compute().get("_is_5hd_pattern"):
            self._5hd_actions = padding.calc_padding(self._outs)

            if self._5hd_actions is not None and len(self._5hd_actions) > 0:
                operation.add_compile_info_inner(CompileInfo.CONTAINS_NEED_PAD_COMPUTE, True)

    def _do_set_value(self):
        if self._5hd_actions is not None and len(self._5hd_actions) > 0:
            for action in self._5hd_actions:
                action_type = action.get_action_type()
                tensor = action.get_tensor()
                shape = tensor.shape
                ori_shape = d_format_util.get_original(shape[1])
                conditon = action.get_condition()
                value = action.get_value()
                target_tensors = action.get_target_tensors()
                value_type = action.get_value_type()

                if value_type == ActionValueType.TENSOR:
                    value = value(self._get_ub_tensor(tensor))

                if action_type == ActionType.SET_VALUE:
                    self._schedule[self._get_ub_tensor(tensor)].set_value(conditon, value)
                elif action_type == ActionType.CACHE_READ_AND_SET_VALUE:
                    set_value_cache_read_buffer = self._schedule.cache_read(tensor, self._scope, target_tensors)
                    # add dma copy emit insn
                    self._emit_insn_map[set_value_cache_read_buffer] = \
                        [set_value_cache_read_buffer.op.axis[0], DMA_COPY]
                    self._schedule[set_value_cache_read_buffer].set_value(conditon, value)


def _fake_node(tensors):
    dtype = tensors[0].dtype
    dim_length = max(len(t.shape) for t in tensors)
    shape = [1] * dim_length
    for tensor_i in tensors:
        if DTYPE_BYTE_MAPPING.get(tensor_i.dtype) > DTYPE_BYTE_MAPPING.get(dtype):
            dtype = tensor_i.dtype
        shape_i = util.shape_to_list(tensor_i.shape)
        diff = dim_length - len(shape_i)
        shape_i = [1] * diff + shape_i
        for j in range(diff, dim_length):
            if util.equals_one(shape[j]):
                shape[j] = shape_i[j]
            elif not expr_equal(shape[j], shape_i[j]) and not util.equals_one(shape_i[j]):
                shape[j] = tvm.max(shape_i[j], shape[j])

    def _fake_compute(*indices):
        res_ = tvm.const(1, dtype)
        for tensor in tensors:
            cur_indices = []
            for idx, dim in enumerate(tensor.shape):
                if util.equals_one(dim):
                    cur_indices.append(0)
                else:
                    cur_indices.append(indices[idx])
            res_ *= tvm.expr.Cast(dtype, tensor(*cur_indices))
        return res_

    with tvm.tag_scope(FAKE_NODE_TAG):
        res = tvm.compute(shape, _fake_compute, name="fake_node")

    return res


def _copy_node(tensor):
    shape = tensor.shape
    with tvm.tag_scope("dma_copy"):
        res = tvm.compute(shape, lambda *i: tensor(*i), name="copy_node")
    return res
