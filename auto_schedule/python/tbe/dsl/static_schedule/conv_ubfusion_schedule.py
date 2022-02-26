#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
Schedule of conv2d ub fusion in v220/v300.
"""
from collections import deque
from te.platform import cce_params
from tbe import tvm
from tbe.dsl.static_schedule.conv_schedule_util import get_src_tensor
from tbe.dsl.static_schedule.conv_schedule_util import is_elewise
from tbe.dsl.static_schedule.conv_schedule_util import is_placeholder
from tbe.dsl.static_schedule.conv_schedule_util import is_shape_equal
from tbe.dsl.static_schedule.conv_schedule_util import is_support_fixpipe_op


UB_COEFF_CONVERT = {
    "int4": 0.25,
    "int8": 0.5,
    "bfloat16": 1,
    "float16": 1,
    "float32": 2,
    "int32": 2,
    } # ub space is calculated in fp16 uniformly.


class EltwiseUBFusion:
    """
    Class of common eltwise op ub fusion.
    process cub and common eltwise ub tensors.
    conv + fixpipe + ub to be supported.
    """
    def __init__(self, res, op_graph, conv_param):
        self.cub_tag_list = ["fixpipe_reform",
                             "convolution_res_fp32_conv2d",
                             "convolution_res_conv2d"
                            ]  # The order cannot be changed.
        self._conv_param = conv_param
        self.flag = self.check_ub_fusion_flag(res)
        self.ub_body_tensors = set()
        self.ub_input_placeholders = set()
        self.ub_input_broadcast_tensors = set()
        self.cache_read_tensors = []
        self.cache_write_tensors = []
        self.inline_tensors = []
        self.emit_insn_dict = {
            "elewise_single_relu": "vector_auto",
            "elewise_single_round_d": "vector_conv_round",
            "elewise_single_VS_max": "vector_maxs",
            "elewise_single_VS_min": "vector_mins",
            "elewise_binary_div": "vector_div",
            "elewise_binary_vcmpv_gt": "vector_gt",
            "elewise_binary_vcmpv_ge": "vector_ge",
            "elewise_binary_vcmpv_lt": "vector_lt",
            "elewise_binary_vcmpv_le": "vector_le",
            "elewise_binary_vcmpv_eq": "vector_eq",
            "elewise_binary_vcmpv_ne": "vector_ne",
            "elewise_binary_cmpsel_gt": "vector_select_gt",
            "elewise_binary_cmpsel_ge": "vector_select_ge",
            "elewise_binary_cmpsel_lt": "vector_select_lt",
            "elewise_binary_cmpsel_le": "vector_select_le",
            "elewise_binary_cmpsel_eq": "vector_select_eq",
            "elewise_binary_cmpsel_ne": "vector_select_ne",
            "elewise_binary_cmpsel": "vector_cmpsel",
            "elewise_binary_add": "vector_add",
            "elewise_binary_sub": "vector_sub",
            "elewise_binary_mul": "vector_mul",
            "elewise_binary_min": "vector_min",
            "elewise_binary_max": "vector_max",
            "elewise_binary_or": "vector_or",
            "elewise_binary_and": "vector_and",
            "elewise_single_lrelu": "vector_auto",
            "elewise_binary_addrelu": "vector_addrelu",
            "elewise_binary_subrelu": "vector_subrelu",
            "elewise_multiple_sel": "vector_select_bool",
            "elewise_single_rec": "vector_rec",
            "emit_insn_elewise_binary_cmp": "vector_gt",
            "elewise_single_VS_mul": "vector_auto",
            "elewise_single_VS_add": "vector_auto",
            "elewise_single_cast": "vector_auto",
            "elewise_single_exp": "vector_auto",
            "elewise_single_log": "vector_auto"
        }
        self.pre_op_list = []
        self.next_op_list = []
        self.cub = None # cub is the first tensor in ub. which is also fixpipe_res.
        self.cache_write_flag = self.flag and is_elewise(res)
        self.cache_read_blacklist = [
            "max_pooling_pad_top",
            "max_pooling_pad_bottom",
            "max_pooling_pad_left",
            "max_pooling_pad_right",
        ]
        if self.flag:
            self.parse_ub_tensors(res, op_graph)
            self.parse_next_op(op_graph)

    def check_ub_fusion_flag(self, res):
        """
        Check for ub fusion situations.
        Special fusion reset cub_tag_list to get the real cub
        """
        if self._conv_param.convbn1_flag:
            self.cub_tag_list = ["convolution_c_ub"]
            return True

        # to be deleted when fixpipe ready
        if not is_support_fixpipe_op():
            return res.op.tag == "elewise_single_VS_min"

        res_tag = get_src_tensor(res).op.tag if res.op.tag == "strided_write" else res.op.tag
        return res_tag not in self.cub_tag_list

    def parse_ub_tensors(self, res, op_graph):
        """
        Parse the body tensors and input placeholders in eltwise ub fusion.
        """
        def parse_cub():
            """
            Parse cub tensor.
            """
            for tag in self.cub_tag_list:
                for lop in op_graph.body_ops:
                    if lop["op"] == tag:
                        return lop["dst_buffer"]
            return None

        self.cub = parse_cub()

        tensor_queue = deque()
        tensor_queue.append(res)

        while tensor_queue:
            src_tensor = tensor_queue.popleft()

            if is_placeholder(src_tensor):
                if src_tensor.op.tag == "broadcast":
                    self.ub_input_broadcast_tensors.add(src_tensor)
                elif src_tensor.op.name not in self.cache_read_blacklist:
                    self.ub_input_placeholders.add(src_tensor)
            elif src_tensor.op.tag == "broadcast_for_tensor":
                self.inline_tensors.append(src_tensor)
            elif is_elewise(src_tensor) and src_tensor != res:  # exclude placeholders
                self.ub_body_tensors.add(src_tensor)

            if src_tensor.op.input_tensors:
                append_list = list(i for i in src_tensor.op.input_tensors if i != self.cub)
                tensor_queue.extend(append_list)

    def parse_next_op(self, op_graph):
        """
        Parse the dependencies between nearby tensors.
        The elements at the same index in pre_op_list and post_op_list are in pair.
        """
        for ops in op_graph.input_ops + op_graph.body_ops:
            pre_op = ops["dst_buffer"]
            for next_ops in ops["next_op"]:
                if pre_op not in self.pre_op_list:
                    self.pre_op_list.append(pre_op)
                    self.next_op_list.append([next_ops["dst_buffer"]])
                else:
                    self.next_op_list[self.pre_op_list.index(pre_op)].append(next_ops["dst_buffer"])

    def coeff_eltwise_cal(self, res):
        """
        Calculate the ub space coefficient.
        """
        eltwise_coeff = 0
        channelwise_coeff = 0
        scalar_num = 0

        def is_memory_unique(tensor):
            """
            Check if cannot reuse the memory of pre tensor.
            """
            if self._conv_param.dynamic_flag:
                return False

            pre_tensor_list = []
            for pre_tensor in tensor.op.input_tensors:
                if pre_tensor in self.inline_tensors:
                    append_list = list(i for i in pre_tensor.op.input_tensors)
                    pre_tensor_list.extend(append_list)
                else:
                    pre_tensor_list.append(pre_tensor)

            for pre_tensor in pre_tensor_list:
                if len(self.next_op_list[self.pre_op_list.index(pre_tensor)]) == 1 and \
                        is_shape_equal(pre_tensor, tensor) and \
                        pre_tensor.dtype == tensor.dtype:
                    return False
            return True

        if self.cache_write_flag and is_memory_unique(res):

            eltwise_coeff += UB_COEFF_CONVERT[res.dtype] # res.local.UB

        if self.flag:
            eltwise_coeff += UB_COEFF_CONVERT[self.cub.dtype]  # C_UB

            #=================================body tensor======================================
            for ub_tensor in self.ub_body_tensors:
                if is_memory_unique(ub_tensor):
                    eltwise_coeff += UB_COEFF_CONVERT[ub_tensor.dtype]

            #=================================input tensor======================================
            for input_tensor in list(self.ub_input_placeholders) + list(self.ub_input_broadcast_tensors):
                if len(input_tensor.shape) == 1:
                    if not isinstance(input_tensor, tvm.expr.Var) and input_tensor.shape[0].value == 1:
                        scalar_num += 1
                    else:
                        channelwise_coeff += UB_COEFF_CONVERT[input_tensor.dtype]
                else:
                    eltwise_coeff += UB_COEFF_CONVERT[input_tensor.dtype]

        return eltwise_coeff, channelwise_coeff, scalar_num

    def cub_set_scope(self, sch):
        """
        Set scope for cub body tensors.
        """
        if self.flag:
            sch[self.cub].set_scope(cce_params.scope_ubuf)
            for tensor in list(self.ub_body_tensors) + list(self.ub_input_broadcast_tensors):
                sch[tensor].set_scope(cce_params.scope_ubuf)

    def inputs_cache_read(self, sch, op_graph):
        """
        Cache read for ub input placeholders.
        """
        next_op_map = {}

        for input_op in op_graph.input_ops:
            next_op_map[input_op["dst_buffer"]] = input_op["next_op"][0]["dst_buffer"]

        for tensor in self.ub_input_placeholders:
            input_ub = sch.cache_read(tensor, cce_params.scope_ubuf, next_op_map[tensor])
            self.cache_read_tensors.append(input_ub)

    def res_cache_write(self, sch, res):
        """
        Cache write for the res tensor of eltwise operation in ub fusion.
        """
        if self.cache_write_flag:
            res_cache_write = sch.cache_write(res, cce_params.scope_ubuf)
            self.cache_write_tensors.append(res_cache_write)

    def ub_tensors_inline(self, sch):
        """
        Compute inline for certain tensors in ub.
        """
        if self.flag:
            for tensor in self.inline_tensors:
                sch[tensor].compute_inline()

    def ub_tensors_attach(self, sch, res, cub_slice_axis):
        """
        Attach for ub tensors.
        """
        if self.flag:
            sch[self.cub].compute_at(sch[res], cub_slice_axis)

            for tensor in self.cache_read_tensors + self.cache_write_tensors + \
                    list(self.ub_body_tensors) + \
                    list(self.ub_input_broadcast_tensors):
                sch[tensor].compute_at(sch[res], cub_slice_axis)

    def ub_tensors_emit_insn(self, sch, res):
        """
        Emit insn for ub tensors.
        """
        if self.flag:
            for tensor in self.cache_read_tensors:
                sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")
            for tensor in self.cache_write_tensors:
                sch[tensor].emit_insn(tensor.op.axis[0], self.emit_insn_dict.get(res.op.tag, res.op.tag))
            for tensor in self.ub_body_tensors:
                sch[tensor].emit_insn(tensor.op.axis[0], self.emit_insn_dict.get(tensor.op.tag, tensor.op.tag))
            for tensor in self.ub_input_broadcast_tensors:
                # broadcast an immediate operand to tensor in ub by vector dup.
                sch[tensor].emit_insn(tensor.op.axis[0], "vector_dup")


class QuantFusion:
    """
    Class of Ascend_quant op fusion.
    Ascend_quant is always the last op when it appeared in int8/int4 conv2d ub fusion dataflow
    if there is no strided_write.
    """
    def __init__(self, res, op_graph):
        res_tag = get_src_tensor(res).op.tag if res.op.tag == "strided_write" else res.op.tag
        self.flag = res_tag == "quant"
        self.fusion_tensors = {}
        self.quant_tensor_dict = {}
        self.emit_insn_dict = {
            "input_ub": "dma_padding",
            "reform_by_vadds": "vector_auto",
            "reform_by_vmuls": "vector_auto",
            "offset_ub": "vector_auto",
            "cast_i8_ub": "vector_conv",
            "cast_i4_ub": "vector_conv",
        }
        self.quant_padding_flag = False
        self.reform_emit_insn_axis = None
        self.parse_quant_tensors(op_graph)

    def parse_quant_tensors(self, op_graph):
        """
        Parse the tensors in Ascend_quant fusion compute.
        """
        if self.flag:
            for lop in op_graph.body_ops:
                if lop["op"] in self.emit_insn_dict:
                    self.quant_tensor_dict[lop["op"]] = lop["dst_buffer"]

            input_ub = self.quant_tensor_dict["input_ub"]
            c_out = input_ub.op.attrs["c_out"].value
            c1_transform = input_ub.op.attrs["c1_transform"].value
            self.quant_padding_flag = c_out % c1_transform != 0

    def cal_quant_coeff(self):
        """
        Calculate ub space coefficient of the tensors in Ascend_quant fusion compute.
        """
        quant_coeff = 0
        if self.flag:
            for tensor_name, tensor in self.quant_tensor_dict.items():
                if tensor_name == "input_ub" and not self.quant_padding_flag:
                    continue
                if tensor_name == "offset_ub":
                    continue
                quant_coeff += UB_COEFF_CONVERT[tensor.dtype]

        return quant_coeff

    def inline_input_ub(self, sch):
        """
        Compute inline for input_ub when output channel is 32 aligned in Ascend_quant.
        """
        if self.flag and not self.quant_padding_flag:
            sch[self.quant_tensor_dict["input_ub"]].compute_inline()
            del self.quant_tensor_dict["input_ub"]

    def quant_tensors_set_scope(self, sch):
        """
        Set scope for the tensors in Ascend_quant fusion compute.
        """
        if self.flag:
            for _, tensor in self.quant_tensor_dict.items():
                sch[tensor].set_scope(cce_params.scope_ubuf)

    def split_reform_axis(self, sch):
        """
        split the c0=32 axis of reform_by_vadds/reform_by_vmuls into 2 and c0=16 axis for emit insn.
        """
        for tensor_name, tensor in self.quant_tensor_dict.items():
            if "reform" in tensor_name:
                axis_list = sch[tensor].op.axis[: - 1]
                reform_c0_outer_axis, reform_c0_axis = sch[tensor].split(tensor.op.axis[-1], 16)
                sch[tensor].reorder(reform_c0_outer_axis, *axis_list)
                self.reform_emit_insn_axis = tensor.op.axis[2]
                # optimization to be added.

    def quant_tensors_attach(self, sch, res, cub_slice_axis):
        """
        Attach for the tensors in Ascend_quant fusion compute.
        """
        if self.flag:
            for _, tensor in self.quant_tensor_dict.items():
                sch[tensor].compute_at(sch[res], cub_slice_axis)

    def quant_tensors_emit_insn(self, sch):
        """
        Emit insn for the tensors in Ascend_quant fusion compute.
        """
        if self.flag:
            for tensor_name, tensor in self.quant_tensor_dict.items():
                if "reform" in tensor_name:
                    sch[tensor].emit_insn(self.reform_emit_insn_axis, self.emit_insn_dict[tensor_name])
                else:
                    sch[tensor].emit_insn(tensor.op.axis[0], self.emit_insn_dict[tensor_name])
