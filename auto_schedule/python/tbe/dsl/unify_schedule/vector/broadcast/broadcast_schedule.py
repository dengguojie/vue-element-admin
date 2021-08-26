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
broadcast schedule
"""
from ... import util
from ...constants import BroadcastPattern
from ...constants import FAKE_NODE_TAG
from ...constants import Pattern
from ...constants import SUPPORT_SCALAR_INSNS
from ...constants import TERNARY_INSNS
from ...schedule import Schedule
from .broadcast_schedule_base import BaseBroadcastSchedule

DEFAULT = "default"

# block size in D architecture
BLOCK_SIZE_BYTE = 32

# temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192

# vcmpsel constant
VCMP_INPUT_NUMBER = 2
VSEL_INPUT_NUMBER = 3
VCMPSEL_INPUT_NUMBER = 4


# 'pylint: disable=R0902, R0903
class BroadcastSchedule(BaseBroadcastSchedule, Schedule):
    """
    BroadcastSchedule
    """

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.BROADCAST]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [BroadcastPattern.B_0]

    def __init__(self, outs, tiling_case):
        super(BroadcastSchedule, self).__init__(outs, tiling_case)

    def _calc_storage_bound(self):
        def _correct_ub_size_by_cmp_sel(_tensor):
            if util.is_vcmp_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VCMP_INPUT_NUMBER - len(_tensor.op.input_tensors))
            if util.is_vsel_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))
                if util.is_v200() and (VSEL_INPUT_NUMBER == len(_tensor.op.input_tensors)):
                    self._tmp_ub_size += BLOCK_SIZE_BYTE
            if util.is_vcmpsel_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VCMPSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))

        def _calc_current_space(_tensor):
            # one of the input of the ternary instruction must be reused with the output
            if util.get_dsl_insn(_tensor) in TERNARY_INSNS or _tensor in dependent_map:
                current_space = len(dependent_map)
            else:
                current_space = len(dependent_map) + 1
            for tensor_i in dependent_map.keys():
                if tensor_i in self._absorbable_broadcast_tensors and \
                        len(tensor_i.op.input_tensors) == 1 and tensor_i.op.input_tensors[0] in dependent_map:
                    current_space -= 1
            if util.need_extent_node(_tensor):
                current_space += 1
            if util.is_unified_broadcast(_tensor) and self._broadcast_axis_num.get(_tensor, 0) > 1:
                current_space += 1
            if util.need_temp_space(_tensor) or _need_external_space(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE
            return current_space

        def _r_coexisting(_tensor):
            if _tensor in dependent_map and _tensor not in init_map:
                return len(dependent_map)
            if util.is_vtranspose_broadcast(_tensor):
                self._tmp_ub_size += VTRANSPOSE_TEMP_SPACE
            _need_space = []
            for _tensor_i in _tensor.op.input_tensors:
                _need_space.append(_r_coexisting(_tensor_i))

            _current_space = _calc_current_space(_tensor)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)

            _need_space.append(_current_space)
            _refresh_dependent(_tensor)
            if _tensor not in dependent_map:
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
            exist_absorbable_broadcast = any([x in self._absorbable_broadcast_tensors
                                              for x in _tensor.op.input_tensors])
            if not exist_absorbable_broadcast:
                return False

            op_tag = util.get_dsl_insn(_tensor)
            support_vector_scalar_insns = ("elewise_binary_add", "elewise_binary_mul")
            if op_tag in set(SUPPORT_SCALAR_INSNS) - set(support_vector_scalar_insns):
                return True

            if util.is_v100() and op_tag in support_vector_scalar_insns and _tensor.dtype == "int32":
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
            coexisting_quantities.append(_r_coexisting(tensor_i))
        if not self._out.op.tag == FAKE_NODE_TAG:
            if util.is_vtranspose_broadcast(self._out):
                self._tmp_ub_size += VTRANSPOSE_TEMP_SPACE

            _current_space = _calc_current_space(self._out)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(self._out)

            coexisting_quantities.append(_current_space)

        self._coexisting_quantity = max(coexisting_quantities)

        if self._coexisting_quantity == 1:
            self._tmp_ub_size += BLOCK_SIZE_BYTE
        if len(self._broadcast_tensors - self._absorbable_broadcast_tensors) > 0:
            self._tmp_ub_size += BLOCK_SIZE_BYTE
