#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
broadcast schedule for milan
"""
from functools import reduce
from operator import mul

from tbe import tvm
from tbe.common.platform import intrinsic_check_support
from tbe.common.platform import ASCEND_910B

from ... import util
from ...constants import BroadcastPattern
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import Pattern
from ...constants import SUPPORT_SCALAR_INSNS
from ...constants import TERNARY_INSNS
from ...schedule import Schedule
from .broadcast_schedule_base import BaseBroadcastSchedule
from .broadcast_tilingcase import TilingStrategy

# block size in D architecture
BLOCK_SIZE_BYTE = 32

VECTOR_REDUCE = "vector_reduce"

# vcmpsel constant
VCMP_INPUT_NUMBER = 2
VSEL_INPUT_NUMBER = 3
VCMPSEL_INPUT_NUMBER = 4
SPECIAL_VCMP_NODE = 3


# 'pylint: disable=R0902, R0903
class BroadcastScheduleMl(BaseBroadcastSchedule, Schedule):
    """
    ElewiseSchedule
    """

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [ASCEND_910B]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.BROADCAST]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [BroadcastPattern.B_0]

    def __init__(self, outs, tiling_case):
        super(BroadcastScheduleMl, self).__init__(outs, tiling_case)

        self._storage_align_map = {}

        self._remove_pad_tensors = set()
        self._remove_pad_cache_read_buffer = set()

    def _calc_storage_bound(self):
        def _correct_ub_size_by_cmp_sel(_tensor):
            if util.is_vcmp_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VCMP_INPUT_NUMBER - len(_tensor.op.input_tensors))
            if util.is_vsel_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))
                if VSEL_INPUT_NUMBER == len(_tensor.op.input_tensors):
                    self._tmp_ub_size += BLOCK_SIZE_BYTE
            if util.is_vcmpsel_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VCMPSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))

        def _vcmp_complex_instructions(_tensor):
            """
            check if vcmp impl by complex instructions
            """
            if not intrinsic_check_support("Intrinsic_vcmpv_eq", "int64") and \
                    util.get_dsl_insn(_tensor).startswith("elewise_binary_vcmpv_") and \
                    _tensor.op.input_tensors:
                src_dtype = _tensor.op.input_tensors[0].dtype
                dst_dtype = _tensor.dtype
                return src_dtype in ("uint64", "int64") and dst_dtype == "int8"
            return False

        def _calc_current_space(_tensor):
            # one of the input of the ternary instruction must be reused with the output
            if util.get_dsl_insn(_tensor) in TERNARY_INSNS or _tensor in dependent_map:
                current_space = len(dependent_map)
            else:
                current_space = len(dependent_map) + 1
            for tensor_i in dependent_map:
                if tensor_i in self._absorbable_broadcast_tensors and \
                        len(tensor_i.op.input_tensors) == 1 and tensor_i.op.input_tensors[0] in dependent_map:
                    current_space -= 1
            # temporary plan: use a temp node
            if util.need_extent_node(_tensor) and not self._is_last_align(_tensor):
                current_space += 1
            if util.is_unified_broadcast(_tensor) and not self._is_last_align(_tensor) and \
                    self._broadcast_axis_num.get(_tensor, 0) > 1:
                current_space += 1
            if self._5hd_actions is not None and len(self._5hd_actions) > 0:
                current_space += 1
            if _vcmp_complex_instructions(_tensor):
                current_space += SPECIAL_VCMP_NODE
            if _need_external_space(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE
            return current_space

        def _r_coexisting(_tensor):
            if _tensor in dependent_map and _tensor not in init_map:
                return len(dependent_map)
            _need_space = []
            for _tensor_i in _tensor.op.input_tensors:
                if _tensor in self._compute_root_tensors:
                    continue
                _need_space.append(_r_coexisting(_tensor_i))

            _current_space = _calc_current_space(_tensor)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)

            _need_space.append(_current_space)
            _refresh_dependent(_tensor)
            if _tensor not in dependent_map:
                if _tensor in self._remove_pad_map:
                    _tensor = self._remove_pad_map[_tensor]
                dependent_map[_tensor] = self._in_out_map[_tensor].copy()
            elif _tensor in init_map:
                init_map.remove(_tensor)
            return max(_need_space)

        def _refresh_dependent(_tensor):
            for _tensor_i in _tensor.op.input_tensors:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        def _need_external_space(_tensor):
            exist_absorbable_broadcast = any(x in self._absorbable_broadcast_tensors
                                              for x in _tensor.op.input_tensors)
            if not exist_absorbable_broadcast:
                return False

            op_tag = util.get_dsl_insn(_tensor)
            if op_tag in set(SUPPORT_SCALAR_INSNS):
                return True

        coexisting_quantities = []
        dependent_map = {}
        init_map = set()
        all_producers = self._middle_tensors.copy()
        all_producers.update(self._out_tensors | self._input_tensors)
        for tensor_i in self._broadcast_store_predicate | self._store_predicate_common_tensors:
            dependent_map[tensor_i] = all_producers.copy()
            init_map.add(tensor_i)
        for tensor_i in self._out.op.input_tensors:
            if tensor_i in self._compute_root_tensors:
                continue
            coexisting_quantities.append(_r_coexisting(tensor_i))
        if not self._out.op.tag == FAKE_NODE_TAG:
            _current_space = _calc_current_space(self._out)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(self._out)

            coexisting_quantities.append(_current_space)

        self._coexisting_quantity = max(coexisting_quantities)

        if self._coexisting_quantity == 1:
            self._tmp_ub_size += BLOCK_SIZE_BYTE

    def _calc_remove_pad(self):
        is_cut_last = self._ub_split_axis == len(self._out.shape) - 1
        if is_cut_last:
            return

        sch = self._schedule
        remove_tensors = self._broadcast_tensors - self._absorbable_broadcast_tensors - self._compute_inline_tensors
        for tensor_i in remove_tensors:
            last_axis_shape = util.shape_to_list(tensor_i.shape)[-1]
            align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[tensor_i.dtype]
            if isinstance(last_axis_shape, (tvm.expr.Var, tvm.expr.Expr)) or \
                    (isinstance(last_axis_shape, int) and last_axis_shape % align_factor != 0):
                self._remove_pad_tensors.add(tensor_i)
                use_tensors = [super(BroadcastScheduleMl, self)._get_ub_tensor(_tensor) for
                               _tensor in self._in_out_map.get(tensor_i, [])]
                if not use_tensors:
                    if tensor_i.op.input_tensors and tensor_i.op.input_tensors[0] in self._input_tensors:
                        use_tensors = list(self._in_out_map.get(tensor_i.op.input_tensors[0], []))
                remove_pad_buffer = sch.cache_read(super(BroadcastScheduleMl, self)._get_ub_tensor(tensor_i),
                                                   self._scope, use_tensors)
                util.merge_value(self._in_out_map, remove_pad_buffer, use_tensors)
                self._in_out_map[tensor_i] = {remove_pad_buffer}
                self._remove_pad_map[tensor_i] = remove_pad_buffer
                self._remove_pad_cache_read_buffer.add(remove_pad_buffer)
                self._middle_tensors.add(remove_pad_buffer)
                self._pure_middle_tensors.add(remove_pad_buffer)

    def _calc_ub_align(self):
        def _dsf_pre_tensors(_tensor):
            if _tensor in compute_align_tensors:
                return
            compute_align_tensors.add(_tensor)
            for tensor_j in _tensor.op.input_tensors:
                if tensor_j in self._input_tensors:
                    storage_align_tensors.add(tensor_j)
                else:
                    _dsf_pre_tensors(tensor_j)

        inline_tensors = self._absorbable_broadcast_tensors | self._compute_inline_tensors
        for tensor_i in inline_tensors:
            if tensor_i in self._remove_pad_map:
                self._remove_pad_tensors.remove(tensor_i)
                self._remove_pad_cache_read_buffer.remove(self._remove_pad_map.get(tensor_i))
                self._middle_tensors.remove(self._remove_pad_map.get(tensor_i))
                self._pure_middle_tensors.remove(self._remove_pad_map.get(tensor_i))
                self._compute_inline_tensors.add(self._remove_pad_map.get(tensor_i))

        compute_align_tensors = set()
        storage_align_tensors = set()
        for tensor_i in self._remove_pad_tensors:
            if not util.is_scalar_broadcast(tensor_i):
                _dsf_pre_tensors(tensor_i)

        for tensor_i in compute_align_tensors:
            tensor_ub = super(BroadcastScheduleMl, self)._get_ub_tensor(tensor_i)
            factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[tensor_i.dtype]
            axis = tensor_ub.op.axis[-1]
            self._compute_align_map[tensor_ub] = [axis, factor]

        offset = 0
        for tensor_i in storage_align_tensors:
            tensor_ub = super(BroadcastScheduleMl, self)._get_ub_tensor(tensor_i)
            factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[tensor_i.dtype]
            axis = tensor_ub.op.axis[-2]
            self._storage_align_map[tensor_ub] = [axis, factor, offset]

    def _calc_emit_insn(self):
        super(BroadcastScheduleMl, self)._calc_emit_insn()
        for tensor_i in self._remove_pad_cache_read_buffer:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], VECTOR_REDUCE]

    def _do_ub_align(self):
        sch = self._schedule
        for tensor_i, (axis, factor, offset) in self._storage_align_map.items():
            sch[tensor_i].storage_align(axis, factor, offset)

        for tensor_i, (axis, factor) in self._compute_align_map.items():
            sch[tensor_i].compute_align(axis, factor)

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        before_ub_tensor = set()
        for tensor_i in self._compute_root_tensors:
            before_ub_tensor.add(self._get_ub_tensor(tensor_i))

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING.get(tensor_i.dtype))
            if tensor_i in before_ub_tensor or tensor_i in self._compute_root_tensors:
                continue
            if self._tiling_strategy == TilingStrategy.CONST:
                dst_shape = util.shape_to_list(tensor_i.shape)
                ele_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(tensor_i.dtype)
                dst_shape[-1] = (dst_shape[-1] + ele_in_block - 1) // ele_in_block * ele_in_block
                if self._need_do_block:
                    dst_shape[self._ub_split_axis] = 1 if dst_shape[self._ub_split_axis] == 1 else self._ub_factor
                real_storage_bound = reduce(mul, dst_shape[self._ub_split_axis:], 1)
                if real_storage_bound % ele_in_block != 0:
                    real_storage_bound = (real_storage_bound + ele_in_block - 1) // ele_in_block * ele_in_block
                storage_bound = min(storage_bound, int(real_storage_bound))
            sch[tensor_i].set_buffer_size(storage_bound)
