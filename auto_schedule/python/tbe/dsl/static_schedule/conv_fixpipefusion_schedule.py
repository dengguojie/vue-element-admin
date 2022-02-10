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
Schedule of conv2d fixpipefusion in v220/v300.
"""
from collections import deque
from tbe.common.utils import log
from tbe.dsl.static_schedule.conv_schedule_util import is_placeholder
from te.platform import cce_params


FIXPIPE_REFORM_TAG = "fixpipe_reform"
QUANT_SCALE_0_STR = "quant_scale_0"
QUANT_SCALE_1_STR = "quant_scale_1"
RELU_WEIGHT_0_STR = "relu_weight_0"
RELU_WEIGHT_1_STR = "relu_weight_1"
ELTWISE_SRC_STR = "eltwise_src"

INTRINSIC_FIXPIPE_UNIT_LIST = "Intrinsic_fix_pipe_unit_list"
UNIT_POST_ELTWISE = "post_eltwise"
FIXPIPE_SCOPE_MAP = {
    QUANT_SCALE_0_STR: cce_params.scope_fb0,
    QUANT_SCALE_1_STR: cce_params.scope_fb3,
    RELU_WEIGHT_0_STR: cce_params.scope_fb1,
    RELU_WEIGHT_1_STR: cce_params.scope_fb2,
    ELTWISE_SRC_STR: cce_params.scope_cbuf
}


class FixpipeFusionNew:
    """
    Class of v300 fixpipe op fusion.
    """
    def __init__(self, fixpipe_res):
        """
        Class FixpipeFusionNew init func.
        """
        self.fixpipe_flag = False
        self.fixpipe_res = fixpipe_res
        self.inline_tensors = []
        self.fixpipe_params = []
        self.fixpipe_tensors = [] # param tensors
        self.eltwise_src = None
        self.eltwise_dtype = "float16"
        self.eltwise_flag = False
        self.quant_pre_flag = False
        self.relu_pre_flag = False
        self.quant_post_flag = False
        self.relu_post_flag = False
        self.anti_quant_flag = False
        self.nz2nd_flag = False
        self.cache_read_tensors = []
        self.cache_read_tensors_elewise = []

    def fetch_quant_relu_flag(self):
        """
        Fetch the quant_pre_flag and relu_pre_flag for tiling info dict.
        """
        return self.quant_pre_flag, self.relu_pre_flag, self.quant_post_flag, self.relu_post_flag, self.anti_quant_flag

    def fetch_eltwise_info(self):
        """
        Return eltwise src info.
        """
        return self.eltwise_flag, self.eltwise_dtype

    def get_eltwise_info(self):
        """
        Get eltwise src info.
        """
        for idx, tensor_param in enumerate(self.fixpipe_params):
            if tensor_param.value == ELTWISE_SRC_STR:
                self.eltwise_src = self.fixpipe_tensors[idx]
                self.eltwise_dtype = self.eltwise_src.dtype
                self.eltwise_flag = True

    def parse_fusion_pattern(self):
        """
        Parse fixpipe fusion.
        """
        tensor_queue = deque()
        tensor_queue.append(self.fixpipe_res)
        find_fixpipe = False
        while tensor_queue:
            src_tensor = tensor_queue.popleft()
            tag = src_tensor.op.tag

            if tag in ("convolution_c_col", "convolution_c_col_bias"):
                break

            if find_fixpipe:
                if not is_placeholder(src_tensor):
                    self.inline_tensors.append(src_tensor)

            if tag == FIXPIPE_REFORM_TAG:
                find_fixpipe = True
                self.fixpipe_flag = True
                self.fixpipe_params = src_tensor.op.attrs["vector_params"]
                self.fixpipe_tensors = src_tensor.op.attrs["vector_tensors"]
                self.nz2nd_flag = bool(src_tensor.op.attrs["nz2nd_flag"].value)
                self.anti_quant_flag = bool(src_tensor.op.attrs["anti_quant_flag"].value)
                self.get_eltwise_info()

                tensor_queue.clear()
            if src_tensor.op.input_tensors:
                append_list = list(i for i in src_tensor.op.input_tensors)
                append_list.reverse()
                tensor_queue.extend(append_list)
        log.debug("fixpipe inline tensors:{}".format(self.inline_tensors))

    def fixpipe_inputs_set_scope(self, sch, op_graph):
        """
        Set scope for fixpipe vector input tensors.
        """
        next_op_map = {}

        for input_op in op_graph.input_ops:
            next_op_map[input_op["dst_buffer"]] = input_op["next_op"][0]["dst_buffer"]

        for idx, tensor_param in enumerate(self.fixpipe_params):
            if tensor_param.value not in FIXPIPE_SCOPE_MAP.keys():
                raise RuntimeError("tensor {} cannot set scope to fb".format(tensor_param))

            tensor = self.fixpipe_tensors[idx]
            scope = FIXPIPE_SCOPE_MAP.get(tensor_param.value)
            if tensor_param.value == ELTWISE_SRC_STR:
                input_l1 = sch.cache_read(tensor, scope, next_op_map[tensor])
                self.cache_read_tensors_elewise.extend([input_l1])
                continue

            input_fb = sch.cache_read(tensor, scope, next_op_map[tensor])
            input_l1 = sch.cache_read(tensor, cce_params.scope_cbuf, input_fb)
            self.cache_read_tensors.extend([input_fb, input_l1])

    def fixpipe_inputs_emit_insn(self, sch):
        """
        Dma for the inputs of fixpipe fusion ops.
        """
        for tensor in self.cache_read_tensors:
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")

        for tensor in self.cache_read_tensors_elewise:
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")

    def inline_fixpipe_tensor(self, sch):
        """
        Inline the body tensors in fixpipe fusion compute.
        """
        for tensor in self.inline_tensors:
            sch[tensor].compute_inline()

    def fixpipe_inputs_compute_at(self, sch, res, fixpipe_slice_axis, cub_slice_axis):
        """
        Attach the inputs of fixpipe fusion ops to res tensor.
        """
        for tensor in self.cache_read_tensors:
            sch[tensor].compute_at(sch[res], fixpipe_slice_axis)

        for tensor in self.cache_read_tensors_elewise:
            sch[tensor].compute_at(sch[res], cub_slice_axis)


class FixpipeFusion:
    """
    Class of fixpipe on-the-fly fusion.
    """
    def __init__(self, fixpipe_res):
        self.fixpipe_res = fixpipe_res
        self.quant_pre_flag = False
        self.relu_pre_flag = False
        self.quant_post_flag = False
        self.relu_post_flag = False
        self.nz2nd_flag = False
        self.anti_quant_flag = False
        self.weight_input = None
        self.inline_tensors = []
        self.fixpipe_inputs = [] # scale of dequant/requant, weight_input of prelu
        self.cache_read_tensors = []

    def parse_fusion_pattern(self):
        """
        Parse the fixpipe fusion type.
        find out the tensors to be inlined and the inputs to be cache readed.
        """
        tensor_queue = deque()
        tensor_queue.append(self.fixpipe_res)
        while tensor_queue:
            src_tensor = tensor_queue.popleft()
            tag = src_tensor.op.tag

            if tag in ("convolution_c_col", "convolution_c_col_bias"):
                break
            if is_placeholder(src_tensor):
                self.fixpipe_inputs.append(src_tensor)
            else: # exclude placeholders
                self.inline_tensors.append(src_tensor)

            if tag == "elewise_binary_add" and "weight_input" in src_tensor.op.attrs:
                self.weight_input = src_tensor.op.attrs["weight_input"].op.input_tensors[0]
                self.relu_pre_flag = True
            if tag in ("dequant_remove_pad", "requant_remove_pad"):
                self.quant_pre_flag = True

            if src_tensor.op.input_tensors:
                append_list = list(i for i in src_tensor.op.input_tensors)
                append_list.reverse()
                tensor_queue.extend(append_list)

        self.inline_tensors = self.inline_tensors[1: ] # fixpipe_res cannot be inlined
        self.inline_tensors = list(set(self.inline_tensors))
        self.fixpipe_inputs = list(set(self.fixpipe_inputs))

    def fetch_quant_relu_flag(self):
        """
        fetch the quant_pre_flag and relu_pre_flag for tiling info dict.
        """
        return self.quant_pre_flag, self.relu_pre_flag, self.quant_post_flag, self.relu_post_flag, self.anti_quant_flag

    def fixpipe_inputs_set_scope(self, sch, op_graph):
        """
        Cache read fixpipe params into L1 and fixpipe.
        """
        next_op_map = {} # save the next tensor of fixpipe inputs

        for input_op in op_graph.input_ops:
            next_op_map[input_op["dst_buffer"]] = input_op["next_op"][0]["dst_buffer"]

        for tensor in self.fixpipe_inputs:
            if tensor == self.weight_input:
                scope_inputs = cce_params.scope_fb1
            elif next_op_map[tensor].op.tag in ("dequant_vector", "requant_vector"):
                scope_inputs = cce_params.scope_fb0

            input_fb = sch.cache_read(tensor, scope_inputs, next_op_map[tensor]) # fb0: QUANT_PRE, fb1: RELU_PRE
            input_l1 = sch.cache_read(tensor, cce_params.scope_cbuf, input_fb)
            self.cache_read_tensors.extend([input_fb, input_l1])

    def fixpipe_inputs_compute_at(self, sch, res, fixpipe_slice_axis, cl0_at_res_axis):
        """
        Attach the inputs of fixpipe fusion ops to res tensor.
        """
        _ = cl0_at_res_axis
        for tensor in self.cache_read_tensors:
            sch[tensor].compute_at(sch[res], fixpipe_slice_axis)

    def inline_fixpipe_tensor(self, sch):
        """
        Inline the body tensors in fixpipe fusion compute.
        """
        for tensor in self.inline_tensors:
            sch[tensor].compute_inline()

    def fixpipe_inputs_emit_insn(self, sch):
        """
        Dma for the inputs of fixpipe fusion ops.
        """
        for tensor in self.cache_read_tensors:
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")
