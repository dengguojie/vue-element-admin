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
layer_norm_beta_gamma_backprop schedule
"""
from te.lang.dynamic.schedule.constants import Pattern
from te.tvm import schedule as tvm
from tbe.dsl.base.operation import register_schedule
import te.platform as tbe_platform

@register_schedule(pattern=Pattern.LAYER_NORM_BETA_GAMMA_BACKPROP)
def schedule(outs, tiling_case):
    """
    schedule for layer_norm_beta_gamma_backprop dynamic shape
    """
    return LayerNormBetaGammaSchedule(outs, tiling_case).schedule()

class LayerNormBetaGammaSchedule:
    """
    LayerNormBetaGammaBackpropSchedule
    """
    def __init__(self, outs, tiling_case):
        self._outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        self._schedule = None
        self._tiling_case = tiling_case

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

    def _get_emit_insn_map(self, tensor):
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
        if tensor.op.tag.find("|") != -1:
            str_list = tensor.op.tag.split("|")
            insn = insn_map.get(str_list[0])
        else:
            insn = insn_map.get(tensor.op.tag)
        return insn

    def schedule_gamma(self, sch, res):
        emit_insn_list = []
        tensorlist = []
        data_dy_out = self.gen_reversed_subgraph_list([res], tensorlist)
        for tensor, tensor_out in tensorlist:
            #if tensor.op.name in ("data_x", "data_dy_layernormgrad_beta_gamma", "data_mean", "data_variance"):
            if tensor.op.name == "data_dy_layernormgrad_beta_gamma":
                tensor_ub = sch.cache_read(tensor, tbe_platform.scope_ubuf, data_dy_out)
                emit_insn_list.append((tensor_ub, tbe_platform.DMA_COPY))
            elif tensor.op.name in ("data_x", "data_mean", "data_variance"):
                tensor_ub = sch.cache_read(tensor, tbe_platform.scope_ubuf, [tensor_out])
                emit_insn_list.append((tensor_ub, tbe_platform.DMA_COPY))
            else:
                sch[tensor].set_scope(tbe_platform.scope_ubuf)
                insn = self._get_emit_insn_map(tensor)
                emit_insn_list.append((tensor, insn))
        res_ub = sch.cache_write(res, tbe_platform.scope_ubuf)

        dim_0 = res_ub.op.axis[0]
        dim_1 = res_ub.op.axis[1]
        dim_2 = res_ub.op.axis[2]
        reduce_dim_0 = res_ub.op.reduce_axis[0]
        reduce_dim_1 = res_ub.op.reduce_axis[1]

        sch[res_ub].reorder(reduce_dim_0, reduce_dim_1, dim_0, dim_1, dim_2)

        for tensor,insn in emit_insn_list:
            sch[tensor].compute_at(sch[res_ub], dim_0)
        sch[res_ub].compute_at(sch[res], res.op.axis[0])

        for tensor,insn in emit_insn_list:
            sch[tensor].emit_insn(tensor.op.axis[0], insn)
        sch[res_ub].emit_insn(dim_1, self._get_emit_insn_map(res))
        sch[res].emit_insn(res.op.axis[1], tbe_platform.DMA_COPY)

    def schedule_beta(self, sch, res):
        emit_insn_list = []
        tensorlist = []
        data_dy_out = self.gen_reversed_subgraph_list([res], tensorlist)
        for tensor, tensor_out in tensorlist:
            if tensor.op.name == "data_dy_layernormgrad_beta_gamma":
                tensor_ub = sch.cache_read(tensor, tbe_platform.scope_ubuf, data_dy_out)
                emit_insn_list.append((tensor_ub, tbe_platform.DMA_COPY))
            else:
                sch[tensor].set_scope(tbe_platform.scope_ubuf)
                insn = self._get_emit_insn_map(tensor)
                emit_insn_list.append((tensor, insn))
        res_ub = sch.cache_write(res, tbe_platform.scope_ubuf)

        dim_0 = res_ub.op.axis[0]
        dim_1 = res_ub.op.axis[1]
        dim_2 = res_ub.op.axis[2]
        reduce_dim_0 = res_ub.op.reduce_axis[0]
        reduce_dim_1 = res_ub.op.reduce_axis[1]

        sch[res_ub].reorder(reduce_dim_0, reduce_dim_1, dim_0, dim_1, dim_2)

        for tensor,insn in emit_insn_list:
            sch[tensor].compute_at(sch[res_ub], dim_0)
        sch[res_ub].compute_at(sch[res], res.op.axis[0])
       
        for tensor,insn in emit_insn_list:
            sch[tensor].emit_insn(tensor.op.axis[0], insn)
        sch[res_ub].emit_insn(dim_1, self._get_emit_insn_map(res))
        sch[res].emit_insn(res.op.axis[1], tbe_platform.DMA_COPY)

    def schedule(self):
        sch = tvm.create_schedule([self._outs[0].op, self._outs[1].op])
        sch.tiling_key = 1
        self.schedule_gamma(sch, self._outs[0])
        self.schedule_beta(sch, self._outs[1])
        return sch
