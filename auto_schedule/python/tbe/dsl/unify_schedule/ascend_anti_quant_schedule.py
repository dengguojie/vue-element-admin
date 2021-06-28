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
ascend_anti_quant schedule
"""
from tbe import tvm
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform import SOC_VERSION
from tbe.common.platform import ASCEND_310
from tbe.common.platform import scope_ubuf
from ..base.operation import register_schedule
from ..base.operation import var
from .constants import Pattern, DTYPE_BYTE_MAPPING
from .ascend_anti_quant_tilingcase import AntiQuantTilingCase

CAST_F16_NAME = "cast_f16_ub"
INPUT_NAME = "input_ub"
VMULS_REFORM_NAME = "reform_by_vmuls"
SQRT_NAME = "scale_sqrt_ub"
OFFSET_NAME = "offset_ub"
TMP_RES_NAME = "tmp_res"


@register_schedule(pattern=Pattern.ASCEND_ANTI_QUANT)
def schedule(outs, tiling_case):
    """
    ascend_anti_quant schedule
    """
    if not isinstance(tiling_case, AntiQuantTilingCase):
        raise RuntimeError("AntiQuantTilingCase required for AscendAntiQuant Schedule")

    return AscendAntiQuantSchedule(outs, tiling_case).do_schedule()

class AscendAntiQuantSchedule:
    def __init__(self, outs, tiling_case):
        self._scope = scope_ubuf
        self._core_dim = get_soc_spec("CORE_NUM")
        if self._scope.lower().find('.ub') != -1:
            self._total_size = get_soc_spec("UB_SIZE")
        else:
            raise RuntimeError("only support UB buffer now")

        self._schedule = None
        self._outs = outs
        self._res = outs[0]
        self._attr_dic = {}
        self._tensor_map = {}
        self._tiling_case = tiling_case
        self._max_ub_size = 1024
        self._ub_split_result = []

        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}

    def _get_tensor_map(self):
        """
        get the compute tensors
        """

        if self._res is None:
            return

        stack = [self._res]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                    self._tensor_map[in_tensor.name] = in_tensor

        if "input_x" in self._tensor_map:
            self._tensor_map.pop("input_x")

    def _get_res_attrs(self):
        """
        get the attrs carried by the tensor
        """
        self._attr_dic["scale"] = self._res.op.attrs['scale']
        self._attr_dic["sqrt_mode"] = self._res.op.attrs['sqrt_mode']
        self._attr_dic["offset"] = self._res.op.attrs['offset']

    def _set_buffer_scope(self):
        """
        set the scope for tensors
        """
        for _, value in self._tensor_map.items():
            self._schedule[value].set_scope(self._scope)

    def _do_storage_bound(self):
        ub_size = get_soc_spec("UB_SIZE")
        total_width = 2
        max_bound = total_width * 128
        self._max_ub_size = int(ub_size // max_bound * 128)

        for _, stage in self._tensor_map.items():
            storage_bound = self._max_ub_size // DTYPE_BYTE_MAPPING[stage.dtype]
            self._schedule[stage].set_storage_bound(storage_bound)

    def _do_tiling(self):
        case = self._tiling_case
        res = self._res
        sch = self._schedule

        # get tiling axis
        block_tiling_axis = case.block_tiling_axis
        ub_tiling_axis = case.ub_tiling_axis

        # get tiling params
        block_factor = case.block_factor
        ub_factor = case.ub_factor

        block_inner_factor = block_factor if block_factor is not None else var(
            "block_factor_" + str(block_tiling_axis) + "_" + str(case.is_fuse_block), (1, None))
        ub_inner_factor = ub_factor if ub_factor is not None else var(
            "ub_factor_" + str(ub_tiling_axis) + "_" + str(case.is_fuse_block), (1, None))

        # block tiling
        block_outer, block_inner = sch[res].split(res.op.axis[block_tiling_axis],
                                                  nparts=block_inner_factor)
        if case.is_split_ub:
            ub_outer, ub_inner = sch[res].split(res.op.axis[ub_tiling_axis],
                                                factor=ub_inner_factor)
        else:
            ub_outer, ub_inner = sch[res].split(block_inner, factor=ub_inner_factor)
        self._ub_split_result = [ub_outer, ub_inner]

        if case.is_fuse_block:
            fuse_axis_list = [sch[res].op.axis[i] for i in range(block_tiling_axis)]
            fuse_axis_list.append(block_outer)
            if len(fuse_axis_list) > 1:
                multi_core_bind_axis = sch[res].fuse(*fuse_axis_list)
            else:
                multi_core_bind_axis = fuse_axis_list[0]
        else:
            multi_core_bind_axis = block_outer

        if case.multi_core:
            block = tvm.thread_axis("blockIdx.x")
            sch[res].bind(multi_core_bind_axis, block)

    def _set_buffer_compute_at(self):
        """
        set the compute axis for tensors
        """
        ub_outer = self._ub_split_result[0]
        res = self._res
        sch = self._schedule

        for _, value in self._tensor_map.items():
            sch[value].compute_at(sch[res], ub_outer)

    def _set_buffer_emit_insn(self):
        """
        instruction mapping
        """
        res = self._res
        sch = self._schedule
        tensor_map = self._tensor_map
        ub_inner = self._ub_split_result[1]

        for tensor in tensor_map.values():
            if tensor.op.name == INPUT_NAME:
                sch[tensor].emit_insn(tensor.op.axis[0], 'dma_copy')
            else:
                sch[tensor].emit_insn(tensor.op.axis[0], 'vector_auto')

        sch[res].emit_insn(ub_inner, 'dma_copy')

    def do_schedule(self):
        """
        auto_schedule for cce AI-CORE
        """
        self._get_tensor_map()

        self._get_res_attrs()

        self._schedule = tvm.create_schedule(self._res.op)

        self._schedule.tiling_key = self._tiling_case.tiling_key

        self._schedule[self._tensor_map.get(INPUT_NAME)].buffer_align((1, 1),
                                                                      (1, 1),
                                                                      (1, 1),
                                                                      (32, 32))
        self._schedule[self._tensor_map.get(CAST_F16_NAME)].buffer_align((1, 1),
                                                                         (1, 1),
                                                                         (1, 1),
                                                                         (32, 32))

        self._set_buffer_scope()

        self._do_storage_bound()

        self._do_tiling()

        self._set_buffer_compute_at()

        self._set_buffer_emit_insn()

        return self._schedule

