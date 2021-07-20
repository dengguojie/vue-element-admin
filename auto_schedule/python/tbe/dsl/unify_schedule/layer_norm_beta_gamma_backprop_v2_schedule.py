# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
layer_norm_beta_gamma_backprop_v2 schedule
"""
from .constants import Pattern
from . import util
from te import tvm
from tbe.dsl.base.operation import register_schedule
import te.platform as tbe_platform
from tbe.dsl.base import operation
from tbe.dsl.base.operation import var
from tbe.dsl.base.operation import add_compile_info


@register_schedule(pattern=Pattern.LAYER_NORM_BETA_GAMMA_BACKPROP_V2)
def schedule(outs, tiling_case):
    """
    schedule for layer_norm_beta_gamma_backprop dynamic shape
    """
    return LayerNormBetaGammaSchedule(outs, tiling_case).schedule(outs, tiling_case)


class LayerNormBetaGammaSchedule:
    """
    LayerNormBetaGammaBackpropSchedule
    """
    def __init__(self, outs, tiling_case):
        self._insn_map = {"elewise_single_cast": "vector_conv",
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
                          "broadcast": "vector_dup",
                          "elewise_binary_sub": "vector_sub",
                          "tuple_reduce_sum": "vector_reduce_sum",
                          "unified_broadcast": "vector_dup",
                          "reduce_sum": "vector_reduce_sum",
                          "broadcast_for_tensor": "unified_broadcast"}

    def gen_reversed_subgraph_list(self, out_tensor, tensor_list):
        if out_tensor is None:
            return
        stack = [t for t in out_tensor]
        visited_list = []
        data_dy_out = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor.op.name == "data_dy_layernormgrad_beta_gamma":
                    data_dy_out.append(cur_tensor)
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                    tensor_list.append((in_tensor, cur_tensor))
        return data_dy_out

    def get_emit_insn_map(self, tensor):
        str_list = tensor.op.tag.split("|")
        return self._insn_map.get(str_list[0])

    def travel_res_and_cache_read(self, res_list, sch):
        emit_insn_list = []
        tensorlist = []
        data_dy_out = self.gen_reversed_subgraph_list(res_list, tensorlist)
        dy_type = None
        for tensor, tensor_out in tensorlist:
            if tensor.op.name == "data_dy_layernormgrad_beta_gamma":
                tensor_ub = sch.cache_read(tensor, tbe_platform.scope_ubuf, data_dy_out)
                emit_insn_list.append((tensor_ub, tbe_platform.DMA_COPY))
                dy_type = tensor.dtype
            elif tensor.op.name in ("data_x", "data_mean", "data_variance"):
                tensor_ub = sch.cache_read(tensor, tbe_platform.scope_ubuf, [tensor_out])
                emit_insn_list.append((tensor_ub, tbe_platform.DMA_COPY))
            else:
                sch[tensor].set_scope(tbe_platform.scope_ubuf)
                insn = self.get_emit_insn_map(tensor)
                emit_insn_list.append((tensor, insn))
        return emit_insn_list, dy_type

    def schedule_split_reduce(self, sch, outs, is_split_i=False):
        res = outs[0]
        emit_insn_list, dy_type = self.travel_res_and_cache_read([res], sch)
        dim0, dim1 = res.op.axis
        reduce_axis = res.op.reduce_axis[0]
        sch[res].reorder(reduce_axis, dim0, dim1)
        reduce_o, reduce_i = sch[res].split(reduce_axis, factor=self.reduce_factor)
        if is_split_i:
            reduce_i_o, reduce_i_i = sch[res].split(reduce_i, factor=50)
        rf_ub = sch.rfactor(res, reduce_o)[0]
        sch[rf_ub].set_scope(tbe_platform.scope_ubuf)
        last_dim = int(res.shape[-1])
        sch[rf_ub].set_storage_bound(last_dim)
        res_gm_list = sch.cache_write(outs, "")
        res_gm = res_gm_list[0]
        sch[res_gm].remove_init()
        if is_split_i:
            sch[rf_ub].reorder(rf_ub.op.axis[0], rf_ub.op.reduce_axis[0], rf_ub.op.reduce_axis[1], rf_ub.op.axis[1],
                               rf_ub.op.axis[2])
        else:
            sch[rf_ub].reorder(rf_ub.op.axis[0], rf_ub.op.reduce_axis[0], rf_ub.op.axis[1], rf_ub.op.axis[2])
        sch[res_gm].reorder(res_gm.op.reduce_axis[0], res_gm.op.axis[0], res_gm.op.axis[1])
        sch[res_gm].bind(res_gm.op.reduce_axis[0], self.block)

        for tensor, insn in emit_insn_list:
            if is_split_i:
                sch[tensor].compute_at(sch[rf_ub], rf_ub.op.reduce_axis[0])
            else:
                sch[tensor].compute_at(sch[rf_ub], rf_ub.op.axis[0])
        sch[rf_ub].compute_at(sch[res_gm], res_gm.op.reduce_axis[0])
        for tensor, insn in emit_insn_list:
            sch[tensor].emit_insn(tensor.op.axis[0], insn)
        if is_split_i:
            sch[rf_ub].emit_insn(rf_ub.op.reduce_axis[1], tbe_platform.REDUCE_SUM)
        else:
            sch[rf_ub].emit_insn(rf_ub.op.reduce_axis[0], tbe_platform.REDUCE_SUM)
        sch[res_gm].emit_insn(res_gm.op.axis[0], tbe_platform.DMA_COPY)
        sch[res].emit_insn(sch[res].op.axis[0], tbe_platform.PHONY_INSN)
        return res_gm_list, dy_type

    def schedule_no_split_reduce(self, sch, outs):
        res = outs[0]
        emit_insn_list, dy_type = self.travel_res_and_cache_read([res], sch)
        dim0, dim1 = res.op.axis
        reduce_axis = res.op.reduce_axis[0]
        sch[res].reorder(reduce_axis, dim0, dim1)
        sch[res].bind(reduce_axis, self.block)
        for tensor, insn in emit_insn_list:
            sch[tensor].compute_at(sch[res], res.op.reduce_axis[0])
        for tensor, insn in emit_insn_list:
            sch[tensor].emit_insn(tensor.op.axis[0], insn)
        sch[res].emit_insn(res.op.axis[0], tbe_platform.DMA_COPY)
        return dy_type

    def get_max_reduce_factor(self, outs, dy_type):
        res = outs[0]
        last_dim = int(res.shape[-1])
        ub_bytes = util.get_ub_size()
        bytes_fp32 = 4
        # 4 fp32 tensor
        bytes_independent = last_dim * 4 * bytes_fp32
        if dy_type == "float32":
            # fp32 situation, 3 tensor dependent on factor
            factor = (ub_bytes - bytes_independent) // (3 * bytes_fp32 * last_dim)
        else:
            # fp15 situation, 4 fp32 tensor and 2 fp16 tensor dependent on factor, 2 fp16 same as 1 fp32
            factor = (ub_bytes - bytes_independent) // (5 * bytes_fp32 * last_dim)
        return factor

    def schedule(self, outs, tiling_case):
        self.block = tvm.thread_axis("blockIdx.x")
        sch = tvm.create_schedule([outs[0].op, outs[1].op])
        self.fused_axis = operation.get_context().get_current_compute().get("fused_axis")
        key_name, sch.tiling_key = tiling_case
        core_num = util.get_core_num()
        max_reduce_factor = 1
        if key_name == "no_split_reduce":
            dy_type = self.schedule_no_split_reduce(sch, outs)
            sch.set_var_range(self.fused_axis, 1, core_num)
        else:
            self.reduce_factor = var("reduce_factor")
            if key_name == "split_reduce":
                [res_gm_0, res_gm_1], dy_type = self.schedule_split_reduce(sch, outs, is_split_i=False)
                max_reduce_factor = self.get_max_reduce_factor(outs, dy_type)
                sch.set_var_range(self.fused_axis, core_num + 1, core_num * max_reduce_factor)
                sch.set_var_range(self.reduce_factor, 1, max_reduce_factor)
            elif key_name == "split_reduce_i":
                [res_gm_0, res_gm_1], dy_type = self.schedule_split_reduce(sch, outs, is_split_i=True)
                max_reduce_factor = self.get_max_reduce_factor(outs, dy_type)
                sch.set_var_range(self.fused_axis, core_num * max_reduce_factor + 1 , None)
            else:
                raise RuntimeError("LayerNormBetaGammaBackpropV2 tiling key error.")
            outs.pop()
            outs.pop()
            outs.append(res_gm_0)
            outs.append(res_gm_1)
        add_compile_info("core_num", core_num)
        add_compile_info("max_reduce_factor", max_reduce_factor)
        return sch

