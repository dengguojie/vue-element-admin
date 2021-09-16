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
from tbe.dsl.base.operation import var
from tbe.dsl.base.operation import get_compile_info
from tbe import dsl
from ..base import operation


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
        self._dy_type = None
        self._max_reduce_factor = 1

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
        for tensor, tensor_out in tensorlist:
            if tensor.op.name == "data_dy_layernormgrad_beta_gamma":
                tensor_ub = sch.cache_read(tensor, tbe_platform.scope_ubuf, data_dy_out)
                emit_insn_list.append((tensor_ub, tbe_platform.DMA_COPY))
            elif tensor.op.name in ("data_x", "data_mean", "data_variance"):
                tensor_ub = sch.cache_read(tensor, tbe_platform.scope_ubuf, [tensor_out])
                emit_insn_list.append((tensor_ub, tbe_platform.DMA_COPY))
            else:
                sch[tensor].set_scope(tbe_platform.scope_ubuf)
                insn = self.get_emit_insn_map(tensor)
                emit_insn_list.append((tensor, insn))
        return emit_insn_list

    def schedule_split_reduce(self, sch, outs, is_split_i=False, is_split_normal=False):
        res = outs[0]
        normal_dim = res.shape[-1]
        reduce_dim = res.shape[0]
        emit_insn_list = self.travel_res_and_cache_read([res], sch)
        dim0, dim1 = res.op.axis
        reduce_axis = res.op.reduce_axis[0]
        sch[res].reorder(reduce_axis, dim0, dim1)
        reduce_o, reduce_i = sch[res].split(reduce_axis, factor=self.reduce_factor)
        if is_split_i:
            # for this situation, double buffer is enabled, so split factor must divided by 2
            if isinstance(normal_dim, tvm.expr.Var) and isinstance(reduce_dim, tvm.expr.Var):
                self.reduce_i_factor = var("reduce_i_factor")
                reduce_i_o, reduce_i_i = sch[res].split(reduce_i, factor=self.reduce_i_factor)
            elif isinstance(normal_dim, tvm.expr.Var):
                reduce_i_o, reduce_i_i = sch[res].split(reduce_i, factor=2)
            else:
                reduce_i_o, reduce_i_i = sch[res].split(reduce_i, factor=self._max_reduce_factor // 2)
        rf_ub = sch.rfactor(res, reduce_o)[0]
        sch[rf_ub].set_scope(tbe_platform.scope_ubuf)
        if isinstance(normal_dim, tvm.expr.Var):
            unknown_normal = True
            sch[rf_ub].set_storage_bound(self._max_last_factor)
        else:
            unknown_normal = False
            normal_dim = int(res.shape[-1])
            if normal_dim <= self._max_last_factor:
                sch[rf_ub].set_storage_bound(normal_dim)
            else:
                sch[rf_ub].set_storage_bound(self._max_last_factor)
        res_gm_list = sch.cache_write(outs, "")
        res_gm = res_gm_list[0]
        sch[res_gm].remove_init()
        res_, res_gm_normal = res_gm.op.axis
        res_gm_reduce_o = res_gm.op.reduce_axis[0]
        if is_split_i:
            # (reduce_o, [reduce_i_o], [reduce_i_i], 1, normal)
            rf_reduce_i_o, rf_reduce_i_i = rf_ub.op.reduce_axis
            rf_reduce_o, _, rf_normal = rf_ub.op.axis
            sch[rf_ub].reorder(rf_reduce_o, rf_reduce_i_o, rf_reduce_i_i, _, rf_normal)
            sch[res_gm].reorder(res_gm_reduce_o, res_, res_gm_normal)
            axis_compute_at, axis_rf_compute_at = rf_reduce_i_o, res_gm_reduce_o
            axis_emit_insn, axis_res_emit_insn = rf_reduce_i_i, res_
            if unknown_normal:
                sch.set_var_range(normal_dim, 1, self._max_last_factor)
                bound = self.calc_storage_bound()
                for tensor, _ in emit_insn_list:
                    sch[tensor].set_storage_bound(bound)
        else:
            # (reduce_o, [reduce_i,] 1, normal)
            rf_reduce_i = rf_ub.op.reduce_axis[0]
            rf_reduce_o, _, rf_normal = rf_ub.op.axis
            if unknown_normal == False:
                if normal_dim > self._max_last_factor:
                    rf_normal_o, rf_normal_i = sch[rf_ub].split(rf_normal, factor=self._max_last_factor)
                    sch[rf_ub].reorder(rf_reduce_o, rf_reduce_i, _, rf_normal_o, rf_normal_i)
                    res_gm_normal_o, res_gm_normal_i = sch[res_gm].split(res_gm_normal, factor=self._max_last_factor)
                    sch[res_gm].reorder(res_gm_reduce_o, res_, res_gm_normal_o, res_gm_normal_i)
                    axis_compute_at, axis_rf_compute_at = rf_normal_o, res_gm_normal_o
                    axis_emit_insn, axis_res_emit_insn = rf_normal_i, res_gm_normal_i
                else:
                    sch[rf_ub].reorder(rf_reduce_o, rf_reduce_i, _, rf_normal)
                    sch[res_gm].reorder(res_gm_reduce_o, res_, res_gm_normal)
                    axis_compute_at, axis_rf_compute_at = rf_reduce_o, res_gm_reduce_o
                    axis_emit_insn, axis_res_emit_insn = rf_reduce_i, res_
            else:
                if is_split_normal:
                    rf_normal_o, rf_normal_i = sch[rf_ub].split(rf_normal, factor=self._max_last_factor)
                    sch[rf_ub].reorder(rf_reduce_o, rf_reduce_i, _, rf_normal_o, rf_normal_i)
                    res_gm_normal_o, res_gm_normal_i = sch[res_gm].split(res_gm_normal, factor=self._max_last_factor)
                    sch[res_gm].reorder(res_gm_reduce_o, res_, res_gm_normal_o, res_gm_normal_i)
                    axis_compute_at, axis_rf_compute_at = rf_normal_o, res_gm_normal_o
                    axis_emit_insn, axis_res_emit_insn = rf_normal_i, res_gm_normal_i
                    sch.set_var_range(normal_dim, self._max_last_factor + 1, None)
                    bound = self.calc_storage_bound()
                    for tensor, _ in emit_insn_list:
                        sch[tensor].set_storage_bound(bound)
                else:
                    sch[rf_ub].reorder(rf_reduce_o, rf_reduce_i, _, rf_normal)
                    sch[res_gm].reorder(res_gm_reduce_o, res_, res_gm_normal)
                    axis_compute_at, axis_rf_compute_at = rf_reduce_o, res_gm_reduce_o
                    axis_emit_insn, axis_res_emit_insn = rf_reduce_i, res_
                    sch.set_var_range(normal_dim, 1, self._max_last_factor)
                    bound = self.calc_storage_bound()
                    for tensor, _ in emit_insn_list:
                        sch[tensor].set_storage_bound(bound)

        sch[res_gm].bind(res_gm.op.reduce_axis[0], self.block)

        for tensor, insn in emit_insn_list:
            sch[tensor].compute_at(sch[rf_ub], axis_compute_at)
        sch[rf_ub].compute_at(sch[res_gm], axis_rf_compute_at)
        for tensor, insn in emit_insn_list:
            sch[tensor].emit_insn(tensor.op.axis[0], insn)
            if is_split_i:
                sch[tensor].double_buffer()
        sch[rf_ub].emit_insn(axis_emit_insn, tbe_platform.REDUCE_SUM)
        sch[res_gm].emit_insn(axis_res_emit_insn, tbe_platform.DMA_COPY)
        sch[res].emit_insn(sch[res].op.axis[0], tbe_platform.PHONY_INSN)
        return res_gm_list

    def schedule_no_split_reduce(self, sch, outs, is_split_normal=False):
        res = outs[0]
        emit_insn_list = self.travel_res_and_cache_read([res], sch)
        dim0, dim1 = res.op.axis
        reduce_axis = res.op.reduce_axis[0]
        normal_dim = res.shape[-1]
        unknown_normal = False
        if isinstance(normal_dim, tvm.expr.Var):
            unknown_normal = True
        else:
            normal_dim = int(res.shape[-1])
        factor = normal_dim
        if unknown_normal == False and normal_dim > self._max_last_factor:
            factor = self._max_last_factor
        if not unknown_normal:
            dim1_o, dim1_i = sch[res].split(dim1, factor=factor)
            sch[res].reorder(reduce_axis, dim0, dim1_o, dim1_i)
            sch[res].bind(reduce_axis, self.block)
            for tensor, insn in emit_insn_list:
                sch[tensor].compute_at(sch[res], dim1_o)
            for tensor, insn in emit_insn_list:
                sch[tensor].emit_insn(tensor.op.axis[0], insn)
            sch[res].emit_insn(dim1_i, tbe_platform.DMA_COPY)
        else:
            if is_split_normal:
                sch.set_var_range(normal_dim, self._max_last_factor + 1, None)
                dim1_o, dim1_i = sch[res].split(dim1, factor=self._max_last_factor)
                sch[res].reorder(reduce_axis, dim0, dim1_o, dim1_i)
                axis_compute_at = dim1_o
                axis_emit_insn = dim1_i
            else:
                sch.set_var_range(normal_dim, 1, self._max_last_factor)
                sch[res].reorder(reduce_axis, dim0, dim1)
                axis_compute_at = dim0
                axis_emit_insn = dim1
            sch[res].bind(reduce_axis, self.block)
            for tensor, insn in emit_insn_list:
                sch[tensor].compute_at(sch[res], axis_compute_at)
            for tensor, insn in emit_insn_list:
                sch[tensor].emit_insn(tensor.op.axis[0], insn)
            sch[res].emit_insn(axis_emit_insn, tbe_platform.DMA_COPY)

    def schedule_no_reduce(self, outs):
        gamma, beta = outs

        if self._dy_type == "float32":
            input_for_gamma, input_dy = gamma.op.input_tensors
        else:
            input_for_gamma, dy_cast = gamma.op.input_tensors
            input_dy = dy_cast.op.input_tensors[0]

        fake_out = dsl.vadd(gamma, beta)
        sch = tvm.create_schedule([fake_out.op])

        input_for_gamma_ub = sch.cache_read(input_for_gamma, tbe_platform.scope_ubuf, [gamma])
        if self._dy_type == "float32":
            input_dy_ub = sch.cache_read(input_dy, tbe_platform.scope_ubuf, [gamma, beta])
            gamma_ub = sch.cache_write(gamma, tbe_platform.scope_ubuf)
            beta_ub = sch.cache_write(beta, tbe_platform.scope_ubuf)
            sch[beta_ub].reused_by(input_dy_ub)
        else:
            sch[dy_cast].set_scope(tbe_platform.scope_ubuf)
            input_dy_ub = sch.cache_read(input_dy, tbe_platform.scope_ubuf, [dy_cast])
            gamma_ub = sch.cache_write(gamma, tbe_platform.scope_ubuf)
            beta_ub = sch.cache_write(beta, tbe_platform.scope_ubuf)
            sch[beta_ub].reused_by(dy_cast)

        axis = fake_out.op.axis[0]
        self.block_factor = var("block_factor")
        self.ub_factor = var("ub_factor")
        sch.set_var_range(self.ub_factor, 1, 100)
        axis_block, axis_ub = sch[fake_out].split(axis, factor=self.block_factor)
        ub_out, ub_in = sch[fake_out].split(axis_ub, factor=self.ub_factor)
        sch[fake_out].bind(axis_block, self.block)

        sch[input_dy_ub].compute_at(sch[fake_out], ub_out)
        sch[input_for_gamma_ub].compute_at(sch[fake_out], ub_out)
        sch[gamma_ub].compute_at(sch[fake_out], ub_out)
        sch[beta_ub].compute_at(sch[fake_out], ub_out)
        sch[gamma].compute_at(sch[fake_out], ub_out)
        sch[beta].compute_at(sch[fake_out], ub_out)

        if self._dy_type == "float16":
            sch[dy_cast].compute_at(sch[fake_out], ub_out)
            sch[dy_cast].emit_insn(dy_cast.op.axis[0], tbe_platform.CAST)

        sch[input_dy_ub].emit_insn(input_dy_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[input_for_gamma_ub].emit_insn(input_for_gamma_ub.op.axis[0], tbe_platform.DMA_COPY)
        sch[gamma_ub].emit_insn(gamma_ub.op.axis[0], tbe_platform.MUL)
        sch[beta_ub].emit_insn(beta_ub.op.axis[0], tbe_platform.PHONY_INSN)
        sch[gamma].emit_insn(gamma.op.axis[0], tbe_platform.DMA_COPY)
        sch[beta].emit_insn(beta.op.axis[0], tbe_platform.DMA_COPY)
        sch[fake_out].emit_insn(ub_in, tbe_platform.PHONY_INSN)
        return sch, gamma, beta

    def calc_storage_bound(self):
        ub_bytes = util.get_ub_size()
        bytes_fp32 = 4
        bytes_fp16 = 2

        # 4 fp32 tensor
        bytes_independent = self._max_last_factor * 2 * bytes_fp32 + 1024 # 1024 for tmp ub
        bytes_remain = ub_bytes - bytes_independent
        if self._dy_type == "float32":
            # 3 tensor ub but one resued so devided by 2
            bound = bytes_remain // 2 // bytes_fp32
        else:
            # 2 fp32 and 1 fp16, fp16 regard as fp32 is safe
            bound = bytes_remain // 3 // bytes_fp32
        return bound

    def schedule(self, outs, tiling_case):
        self.block = tvm.thread_axis("blockIdx.x")
        key_name, tiling_key = tiling_case
        compile_info = get_compile_info()
        current_compute = operation.get_context().get_current_compute()
        self._dy_type = current_compute.get("dy_type")
        if key_name == "no_reduce":
            # (-1, )
            sch, gamma, beta = self.schedule_no_reduce(outs)
            sch.tiling_key = tiling_key
            outs.pop()
            outs.pop()
            outs.append(gamma)
            outs.append(beta)
            return sch

        sch = tvm.create_schedule([outs[0].op, outs[1].op])
        sch.tiling_key = tiling_key
        self._max_reduce_factor = compile_info.get("max_reduce_factor")
        self._max_last_factor = compile_info.get("max_last_factor")
        core_num = compile_info.get("core_num")
        self.fused_axis = current_compute.get("reduce_dim")

        if key_name in ("dynamic_reduce_no_split", "all_dynamic_no_split", "dynamic_normal_no_split"):
            self.schedule_no_split_reduce(sch, outs)
            if isinstance(self.fused_axis, tvm.expr.Var):
                sch.set_var_range(self.fused_axis, 1, core_num)
        elif key_name in ("all_dynamic_split_normal", "dynamic_normal_split_normal"):
            self.schedule_no_split_reduce(sch, outs, is_split_normal=True)
            if isinstance(self.fused_axis, tvm.expr.Var):
                sch.set_var_range(self.fused_axis, 1, core_num)
        elif key_name in ("dynamic_reduce_split_reduce", "all_dynamic_split_reduce", "dynamic_normal_split_reduce"):
            if isinstance(self.fused_axis, tvm.expr.Var):
                self.reduce_factor = var("reduce_factor")
            else:
                self.reduce_factor = (self.fused_axis + core_num - 1) // core_num
            [res_gm_0, res_gm_1] = self.schedule_split_reduce(sch, outs, is_split_i=False)
            if key_name == "dynamic_reduce_split_reduce":
                sch.set_var_range(self.fused_axis, core_num + 1, core_num * self._max_reduce_factor)
                sch.set_var_range(self.reduce_factor, 1, self._max_reduce_factor)
            outs.pop()
            outs.pop()
            outs.append(res_gm_0)
            outs.append(res_gm_1)
        elif key_name in ("all_dynamic_split_reduce_split_normal", "dynamic_normal_split_reduce_split_normal"):
            if isinstance(self.fused_axis, tvm.expr.Var):
                self.reduce_factor = var("reduce_factor")
            else:
                self.reduce_factor = (self.fused_axis + core_num - 1) // core_num
            [res_gm_0, res_gm_1] = self.schedule_split_reduce(sch, outs, is_split_i=False, is_split_normal=True)
            outs.pop()
            outs.pop()
            outs.append(res_gm_0)
            outs.append(res_gm_1)
        elif key_name in ("all_dynamic_split_reduce_i", "dynamic_reduce_split_reduce_i",
                          "dynamic_normal_split_reduce_i"):
            if isinstance(self.fused_axis, tvm.expr.Var):
                self.reduce_factor = var("reduce_factor")
            else:
                self.reduce_factor = (self.fused_axis + core_num - 1) // core_num
            [res_gm_0, res_gm_1] = self.schedule_split_reduce(sch, outs, is_split_i=True)

            if key_name == "dynamic_reduce_split_reduce_i":
                sch.set_var_range(self.fused_axis, core_num * self._max_reduce_factor + 1, None)
            outs.pop()
            outs.pop()
            outs.append(res_gm_0)
            outs.append(res_gm_1)
        else:
            raise RuntimeError("LayerNormBetaGammaBackpropV2 tiling key error.")
        return sch

