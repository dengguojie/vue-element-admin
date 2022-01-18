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
Schedule of conv2d in v220.
"""
from collections import deque
from functools import reduce
import tbe
from tbe import tvm
from tbe.common.utils import log
from tbe.common.platform import platform_info as cce
from te.platform.cce_params import scope_fb0, scope_fb1, scope_fb2, scope_fb3, scope_bt, scope_cbuf
from tbe.common.platform import CUBE_MKN
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.tiling import tiling_api
from tbe.common.utils.errormgr import error_manager_cube as err_man
from tbe.dsl.static_schedule import util
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
    QUANT_SCALE_0_STR: scope_fb0,
    QUANT_SCALE_1_STR: scope_fb3,
    RELU_WEIGHT_0_STR: scope_fb1,
    RELU_WEIGHT_1_STR: scope_fb2,
    ELTWISE_SRC_STR: scope_cbuf
}

NON_L1_FUSION = -1
DEPTH_L1_FUSION = 0
BREADTH_L1_FUSION = 1

DDR_SCOPE = 0
L1_SCOPE = 1
L2_SCOPE = 2


def ceil_div(num_a, num_b):
    """
    Do upper division.
    """
    if num_b == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (num_a + num_b - 1) // num_b


def ceil(num_a, num_b):
    """
    Do upper align.
    """
    if num_b == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (num_a + num_b - 1) // num_b*num_b


def get_src_tensor(tensor):
    """
    Get the source tensor of input tensor.
    """
    src_tensor = tensor.op.input_tensors[0]
    return src_tensor


def is_placeholder(tensor):
    """
    Check whether the input tensor is a placeholder.
    """
    if tensor.op.input_tensors:
        return False
    return True


def is_support_fixpipe_op():
    if tbe.common.platform.platform_info.intrinsic_check_support(INTRINSIC_FIXPIPE_UNIT_LIST):
        return tbe.common.platform.platform_info.intrinsic_check_support(
            INTRINSIC_FIXPIPE_UNIT_LIST, UNIT_POST_ELTWISE)

    return False

class FixpipeFusionNew(object):
    """
    """
    def __init__(self):
        """
        class FixpipeFusionNew init func
        """
        self.fixpipe_flag = False
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
        self.cache_read_tensors_channelwise = []

    def fetch_quant_relu_flag(self):
        """
        fetch the quant_pre_flag and relu_pre_flag for tiling info dict.
        """
        return self.quant_pre_flag, self.relu_pre_flag, self.quant_post_flag, self.relu_post_flag, self.anti_quant_flag

    def fetch_eltwise_info(self):
        """
        return eltwise src info
        """
        return self.eltwise_flag, self.eltwise_dtype

    def get_eltwise_info(self):
        """
        get eltwise src info
        """
        for idx, tensor_param in enumerate(self.fixpipe_params):
            if tensor_param.value == ELTWISE_SRC_STR:
                self.eltwise_src = self.fixpipe_tensors[idx]
                self.eltwise_dtype = self.eltwise_src.dtype
                self.eltwise_flag = True

    def parse_fusion_pattern(self, res):
        """
        parse fixpipe fusion
        """
        tensor_queue = deque()
        tensor_queue.append(res)
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
        set scope for fixpipe vector input tensors
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
                self.cache_read_tensors_channelwise.extend([input_l1])
                continue

            input_fb = sch.cache_read(tensor, scope, next_op_map[tensor])
            input_l1 = sch.cache_read(tensor, cce.scope_cbuf, input_fb)
            self.cache_read_tensors.extend([input_fb, input_l1])

    def fixpipe_inputs_emit_insn(self,sch):
        """
        Dma for the inputs of fixpipe fusion ops.
        """
        for tensor in self.cache_read_tensors:
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")

        for tensor in self.cache_read_tensors_channelwise:
            sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")

    def inline_fixpipe_tensor(self, sch):
        """
        Inline the body tensors in fixpipe fusion compute.
        """
        for tensor in self.inline_tensors:
            sch[tensor].compute_inline()

    def fixpipe_inputs_compute_at(self, sch, res, fixpipe_slice_axis, cl0_at_res_axis):
        """
        Attach the inputs of fixpipe fusion ops to res tensor.
        """
        for tensor in self.cache_read_tensors:
            sch[tensor].compute_at(sch[res], fixpipe_slice_axis)

        for tensor in self.cache_read_tensors_channelwise:
            sch[tensor].compute_at(sch[res], cl0_at_res_axis)


class FixpipeFusion:
    """
    Class of fixpipe on-the-fly fusion.
    """
    def __init__(self):
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

    def parse_fusion_pattern(self, res):
        """
        Parse the fixpipe fusion type.
        find out the tensors to be inlined and the inputs to be cache readed.
        """
        tensor_queue = deque()
        tensor_queue.append(res)
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

        self.inline_tensors = self.inline_tensors[1: ] # res cannot be inlined
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
            input_l1 = sch.cache_read(tensor, cce.scope_cbuf, input_fb)
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


class InputNd2Nz:
    """
    Class of input nd2nz.
    Transform the fmap of NHWC format in ddr to the fmap_l1 of 5hd format in L1.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.input_nd_flag

    def inline_input_nd_dynamic(self, sch, tensor_map, dynamic_flag):
        """
        inline al1 in nd2nz dynamic situation (group = 1).
        """
        if self.flag and dynamic_flag:
            input_nd_dynamic = tensor_map["fmap"]
            sch[input_nd_dynamic].compute_inline()

    def al1_nd2nz_emit_insn(self, sch, al1):
        """
        Dma for the al1 tensor in nd2nz situation.
        """
        sch[al1].emit_insn(al1.op.axis[1],
                           "dma_copy",
                           {"layout_transform": "nd2nz"})


class WeightNd2Nz:
    """
    Class of weight nd2nz.
    Transform the weight of NHWC format in ddr to the weight_l1 of fracZ format in L1.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.weight_nd_flag

    def check_bl1_nd2nz_tiling(self, tiling):
        """
        Check whether the bl1 tiling is None in weight nd2nz situation.
        """
        if self.flag and tiling["BL1_shape"] is None:
            err_man.raise_err_specific(
                "conv2d", "BL1 tiling cannot be None when weight nd2nz.")

    def bl1_nd2nz_emit_insn(self, sch, bl1):
        """
        Dma for the bl1 tensor in weight nd2nz situation.
        """
        sch[bl1].emit_insn(bl1.op.axis[0],
                           "dma_copy",
                           {"layout_transform": "nd2nz"})


class OutputNz2Nd:
    """
    Class of output nz2nd.
    Transform the output of 5hd format in L0C to the output of NHWC format in DDR.
    """
    def __init__(self, res):
        self.flag = res.op.tag == "5HD_trans_NHWC"

    def res_nz2nd_emit_insn(self, sch, res, res_pragma_axis):
        """
        Emit insn for res in output nz2nd situation.
        """
        sch[res].emit_insn(res_pragma_axis,
                           "dma_copy",
                           attrs={"layout_transform": "nz2nd"})


class LxFusion:
    """
    Class of L1 fusion and L2 fusion.
    """
    def __init__(self, conv_param):
        # lxfusion params
        self.fusion_para = conv_param.fusion_para
        self.l1_fusion_type = self.fusion_para.get("l1_fusion_type")
        self.fmap_l1_addr_flag = self.fusion_para.get("fmap_l1_addr_flag")
        self.fmap_l1_valid_size = self.fusion_para.get("fmap_l1_valid_size")
        self.input_memory_type = self.fusion_para.get("input_memory_type")
        self.output_memory_type = self.fusion_para.get("output_memory_type")

        # combined parameters
        self.l1_fusion_flag = self.l1_fusion_type in (DEPTH_L1_FUSION, BREADTH_L1_FUSION) # l1fusion enable

    def config_default_tiling(self, default_tiling):
        """
        Config default tiling in l1fusion.
        """
        if self.l1_fusion_flag:
            default_tiling["block_dim"] = [1, 1, 1, 1]
        if self.l1_fusion_type == BREADTH_L1_FUSION:
            default_tiling["AL1_shape"] = []
        return default_tiling

    def check_l1fusion_tiling(self, tiling):
        """
        Check the tiling of l1fusion.
        """
        if self.l1_fusion_type == 1 and tiling["AL1_shape"] != []:
            err_man.raise_err_value_or_format_invalid(
                "conv2d", "tiling[\"AL1_shape\"]", "[]",
                "when l1_fusion_type is breadth fusion.")

        if self.l1_fusion_flag:
            if tiling["A_overhead_opt_flag"]:
                err_man.raise_err_value_or_format_invalid(
                    "conv2d", 'tiling["A_overhead_opt_flag"]', "False", "when l1_fusion.")

            if tiling["B_overhead_opt_flag"]:
                err_man.raise_err_value_or_format_invalid(
                    "conv2d", 'tiling["B_overhead_opt_flag"]', "False", "when l1_fusion.")

            if tiling["block_dim"] != [1, 1, 1, 1]:
                err_man.raise_err_specific(
                    "conv2d", "only support one core tiling in L1 Fusion.")

    def config_al1_scope(self):
        """
        Config L1 scope to scope_cbuf_fusion when l1fusion is enabled.
        """
        if self.l1_fusion_flag:
            return cce.scope_cbuf_fusion
        return cce.scope_cbuf

    def align_al1_lxfusion(self, sch, al1):
        """
        AL1 buffer_align in l1fusion breadth fusion.
        """
        sch[al1].buffer_align(
            (1, 1),
            (1, 1),
            (al1.shape[2], al1.shape[2]),
            (al1.shape[3], al1.shape[3]),
            (1, 1))

    def config_l1_tensormap(self, sch, fmap, al1, op_graph):
        """
        Append L1 tensor in cce kernel function.
        """
        l1_tensor_map = {}
        if self.fmap_l1_addr_flag == "nothing":
            l1_tensor_map = None
        else:
            if self.l1_fusion_flag and self.input_memory_type[0] in (DDR_SCOPE, L2_SCOPE):
                for input_op in op_graph.input_ops:
                    l1_tensor_map[input_op["dst_buffer"]] = tvm.var("dummy")
                l1_tensor_map[fmap] = al1
                if self.fmap_l1_valid_size > 0:
                    sch[al1].set_buffer_size(self.fmap_l1_valid_size)
            else:
                l1_tensor_map = None

        util.L1CommonParam.l1_fusion_tensors_map = l1_tensor_map

    def al1_l1fusion_pragma(self, sch, al1):
        """
        Pragma on AL1 when l1fusion and fmap ddr in.
        """
        if self.input_memory_type[0] != L1_SCOPE and self.l1_fusion_flag:
            sch[al1].pragma(al1.op.axis[0], 'jump_data', 1)


class AippFusion:
    """
    Class of Aipp + conv2d fusion.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.aipp_fuse_flag

    def al1_aipp_emit_insn(self, sch, al1):
        """
        Emit insn for al1 in aipp fusion.
        """
        aipp_map = al1.op.attrs
        aipp_map['spr_0'] = al1.op.axis[0]
        aipp_map_res = {"spr_0": al1.op.axis[0],
                        "spr_1": aipp_map["spr_1"],
                        "spr_2": aipp_map["spr_2"],
                        "spr_3": aipp_map["spr_3"],
                        "spr_4": aipp_map["spr_4"],
                        "spr_5": aipp_map["spr_5"],
                        "spr_6": aipp_map["spr_6"],
                        "spr_7": aipp_map["spr_7"],
                        "spr_8": aipp_map["spr_8"],
                        "spr_9": aipp_map["spr_9"],
                        "src_image_h": aipp_map["src_image_h"],
                        "src_image_w": aipp_map["src_image_w"],
                        "input_format": aipp_map["input_format"],
                        "load_start_pos_h": aipp_map["load_start_pos_h"],
                        "load_start_pos_w": aipp_map["load_start_pos_w"],
                        "crop_size_h": aipp_map["crop_size_h"],
                        "crop_size_w": aipp_map["crop_size_w"]}
        sch[al1].emit_insn(al1.op.axis[1], "load_image_to_cbuf", aipp_map_res)


class DynamicShape:
    """
    Class of dynamic shape.
    """
    def __init__(self, conv_param, tiling_dict_flag, tiling_case, var_range):
        self.flag = conv_param.dynamic_flag
        self.var_map = conv_param.dyn_var_map
        self.tiling_dict_flag = tiling_dict_flag
        self.tiling_case = tiling_case
        self.var_range = var_range

        #==========combined parameters==================
        self.h_dynamic = "fmap_h" in self.var_map
        self.w_dynamic = "fmap_w" in self.var_map
        self.hw_dynamic = self.h_dynamic or self.w_dynamic
        self.n_dynamic = "batch_n" in self.var_map

    def fetch_tiling_case(self):
        """
        Fetch tiling case in dynamic shape.
        """
        return self.tiling_case

    def handle_var_range(self, sch):
        """
        Set var range for hi, ho, wi, wo, batch.
        """
        var_range = self.var_range
        var_map = self.var_map

        if self.h_dynamic:
            fmap_h_range = var_range['fmap_h']
            ho_range = var_range['ho']
            sch.set_var_range(var_map['fmap_h'], fmap_h_range[0], fmap_h_range[1])
            sch.set_var_range(var_map['ho'], ho_range[0], ho_range[1])

        if self.w_dynamic:
            fmap_w_range = var_range['fmap_w']
            wo_range = var_range['wo']
            sch.set_var_range(var_map['fmap_w'], fmap_w_range[0], fmap_w_range[1])
            sch.set_var_range(var_map['wo'], wo_range[0], wo_range[1])

        if self.n_dynamic:
            batch_range = var_range['batch_n']
            sch.set_var_range(var_map['batch_n'], batch_range[0], batch_range[1])

    def check_dynamic_overhead_opt_flag(self, tiling):
        """
        Fmap overhead opti is not supported when hi or wi is dynamic.
        """
        if self.hw_dynamic and tiling["A_overhead_opt_flag"]:
            err_man.raise_err_value_or_format_invalid(
                "conv2d", 'tiling["A_overhead_opt_flag"]', "False", "when dynamic shape.")

    def disable_memory_reuse(self, sch, tensor_param):
        """
        Disable memory reuse in dynamic situation.
        """
        if self.flag:
            al1 = tensor_param["al1"]
            bl1 = tensor_param["bl1"]
            al0 = tensor_param["al0"]
            bl0 = tensor_param["bl0"]
            cl0 = tensor_param["cl0"]

            # sequential_malloc
            sch.sequential_malloc(cce.scope_cbuf)
            sch.sequential_malloc(cce.scope_ca)
            sch.sequential_malloc(cce.scope_cb)
            sch.sequential_malloc(cce.scope_cc)

            # mem_unique
            sch[al1].mem_unique()
            sch[al0].mem_unique()
            if bl1 is not None:
                sch[bl1].mem_unique()
            sch[bl0].mem_unique()
            sch[cl0].mem_unique()

    def set_al1_bound(self, sch, al1, conv_param, tiling_param, l0a_load2d_flag, strideh_opti_flag):
        """
        Set al1 bound for dynamic shape.
        """
        def modify_m_for_load3d():
            ho_len = tvm.floordiv(m_al1, out_width) + additional_rows
            hi_max = kernel_h + (ho_len - 1)*stride_h_update
            return hi_max*in_width

        if self.flag:
            _, in_c1, in_height, in_width, in_c0 = tiling_param["fmap_5hd_shape"]
            al1_tiling = tiling_param["al1_tiling"]
            stride_h_update = tiling_param["stride_h_update"]
            m_cl0 = tiling_param["m_cl0"]

            stride_h = conv_param.stride_h
            kernel_h = conv_param.filter_h
            out_width = conv_param.w_out

            if al1_tiling:
                multi_m_al1 = tiling_param["multi_m_al1"]
                k1_al1 = tiling_param["k1_al1"]
                m_al1 = multi_m_al1 * m_cl0
                if l0a_load2d_flag:
                    pass
                else:
                    if self.hw_dynamic:
                        additional_rows = tvm.select(
                            tvm.floormod(m_al1, out_width) == 0,
                            0,
                            tvm.select(tvm.floormod(m_al1 * 2, out_width) == 0, 1, 2))
                    elif self.n_dynamic:
                        if m_al1 % out_width == 0:
                            additional_rows = 0
                        elif m_al1 * 2 % out_width == 0:
                            additional_rows = 1
                        else:
                            additional_rows = 2
                    m_al1 = modify_m_for_load3d()
                al1_bound = m_al1 * k1_al1 * in_c0
            else:
                if strideh_opti_flag:
                    in_height = (in_height - 1) // stride_h + 1
                m_al1 = ceil(in_height * in_width, in_c0)
                al1_bound = m_al1 * in_c1 * in_c0

            sch[al1].set_buffer_size(al1_bound)

    def set_cl0_bound(self, sch, cl0, cl0_tiling):
        """
        Set storage bound for CL0 in dynamic shape
        to solve the memory allocate problem of using storage_align for CL0.
        """
        if self.flag:
            sch[cl0].set_buffer_size(reduce((lambda x, y: x*y), cl0_tiling))

    def res_hw_dynamic_pragma(self, sch, res, res_pragma_axis):
        """
        Pragma for res when hw dynamic.
        """
        if self.hw_dynamic:
            sch[res].pragma(res_pragma_axis, "gm_no_sync", 1)

    def dynamic_mode_im2col_v2(self, sch, conv_param, tensor_param, tiling_param,
                               emit_insn_dict, input_nd_flag, l0a_load2d_flag):
        """
        Use im2col_v2 in dynamic shape situation.
        """
        stride_h_update = tiling_param["stride_h_update"]
        fmap = tensor_param["fmap"]
        al1 = tensor_param["al1"]
        al0 = tensor_param["al0"]
        dynamic_al0_pragma_axis = emit_insn_dict["dynamic_al0_pragma_axis"]

        im2col_attr = {
            'set_fmatrix': 1,
            'conv_kernel_h': conv_param.filter_h,
            'conv_kernel_w': conv_param.filter_w,
            'conv_padding_top': conv_param.padding[0],
            'conv_padding_bottom': conv_param.padding[1],
            'conv_padding_left': conv_param.padding[2],
            'conv_padding_right': conv_param.padding[3],
            'conv_stride_h': stride_h_update,
            'conv_stride_w': conv_param.stride_w,
            'conv_fm_c': fmap.shape[4]*fmap.shape[1],
            'conv_fm_c1': fmap.shape[1],
            'conv_fm_h': fmap.shape[2],
            'conv_fm_w': fmap.shape[3],
            'conv_fm_c0': fmap.shape[4],
        }
        im2col_attr_0 = {
            'set_fmatrix': 0,
            'conv_kernel_h': conv_param.filter_h,
            'conv_kernel_w': conv_param.filter_w,
            'conv_padding_top': conv_param.padding[0],
            'conv_padding_bottom': conv_param.padding[1],
            'conv_padding_left': conv_param.padding[2],
            'conv_padding_right': conv_param.padding[3],
            'conv_stride_h': stride_h_update,
            'conv_stride_w': conv_param.stride_w,
            'conv_fm_c': fmap.shape[4]*fmap.shape[1],
            'conv_fm_c1': fmap.shape[1],
            'conv_fm_h': fmap.shape[2],
            'conv_fm_w': fmap.shape[3],
            'conv_fm_c0': fmap.shape[4],
            'group_flag': 1,
            'l1_group_flag': 1
        }

        if l0a_load2d_flag:
            sch[al1].emit_insn(al1.op.axis[0], "dma_copy")
        elif input_nd_flag:
            im2col_attr.update({"layout_transform": "nd2nz"})
            sch[al1].emit_insn(al1.op.axis[2], "dma_copy", im2col_attr)
        else:
            sch[al1].emit_insn(al1.op.axis[0], "dma_copy", im2col_attr)

        if l0a_load2d_flag:
            sch[al0].emit_insn(dynamic_al0_pragma_axis, "dma_copy")
        else:
            sch[al0].emit_insn(dynamic_al0_pragma_axis, 'im2col_v2', im2col_attr_0)


class StridedRead:
    """
    Class of StridedRead + Conv2d fusion.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.strided_read_flag

    def process_strided_read(self, sch, al1, strideh_opti_flag, l0a_load2d_flag):
        """
        Inline the output tensor of strided read when strideh_opti or l0a_load2d is enabled.
        """
        def inline_fmap_strided_read(fusion_flag, al1):
            if fusion_flag and self.flag:
                fmap_strided_read = get_src_tensor(al1)
                sch[fmap_strided_read].compute_inline()

        inline_fmap_strided_read(strideh_opti_flag, al1)
        inline_fmap_strided_read(l0a_load2d_flag, al1)


class StridedWrite:
    """
    Class of Conv2d + StridedWrite fusion.
    """
    def __init__(self, res):
        self.flag = res.op.tag == "strided_write"

    def process_strided_write(self, sch, res):
        """
        Inline the output tensor of strided write.
        """
        if self.flag:
            strided_write_src = get_src_tensor(res)
            res_hw, res_c0 = res.shape[-2:]
            sch[strided_write_src].compute_inline()
            sch[res].bind_buffer(res.op.axis[0], res_hw * res_c0 * res.op.attrs['stride'], 0)


class Im2colDma:
    """
    Class of Im2col dma when load3d exceeds L1 size.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.l0a_dma_flag

    def config_al1_im2col(self, sch, tensor_map):
        """
        Get the im2col_fractal tensor in L1.
        """
        if self.flag:
            al1_im2col = tensor_map["fmap_im2col"]
            sch[al1_im2col].set_scope(cce.scope_cbuf)
        else:
            al1_im2col = None
        return al1_im2col

    def config_al0_im2coldma(self, sch, al1_im2col, cl0):
        """
        Cache read al1_im2col into L0.
        """
        al0 = sch.cache_read(al1_im2col, cce.scope_ca, [cl0])
        return al0

    def align_al1_im2col(self, sch, al1_im2col, block_k0):
        """
        Buffer align al1_im2col.
        """
        if self.flag:
            sch[al1_im2col].buffer_align(
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, block_k0))

    def inline_al1_im2coldma(self, sch, al1, fmap_row_major):
        """
        Inline al1 and row major tensor.
        """
        if self.flag:
            sch[al1].compute_inline()
            sch[fmap_row_major].compute_inline()

    def im2col_dma_emit_insn(self, sch, al1_im2col, al0, al0_axis_list):
        """
        Emit insn for al1_im2col and al0.
        """
        sch[al1_im2col].emit_insn(al1_im2col.op.axis[5], "dma_copy")
        sch[al0].emit_insn(al0_axis_list[0], "dma_copy")


class InnerBatch:
    """
    Class of l0b innerbatch.
    """
    def __init__(self):
        self.flag = False # inner_batch of sharing L0B weight

    def set_l0b_innerbatch_flag(self, tiling, batch_cl0):
        """
        Set l0b innerbatch flag by tiling.
        """
        if batch_cl0 > 1 and tiling["BL1_shape"] is None:
            self.flag = True
            log.debug("enable innerbatch")

    def config_innerbatch_axis(self, batch_al1, batch_cl0):
        """
        Config innerbatch case batch inner split axis.
        """
        if self.flag:
            return batch_cl0
        return batch_al1

    def check_innerbatch_tiling(self, tiling, batch, out_hw):
        """
        Check the tiling of l0b innerbatch.
        """
        if self.flag:
            _, mc_cl0, m0_cl0, _, batch_cl0, _ = tiling["CL0_matrix"]
            batch_dim, _, _, _ = tiling["block_dim"]
            if mc_cl0 * m0_cl0 != ceil(out_hw, m0_cl0):
                err_man.raise_err_specific("conv2d", "innerbatch case must full load M.")
            if batch_cl0 * batch_dim > batch:
                err_man.raise_err_specific("conv2d", "innerbatch batch_cl0*batch_dim cannot be greater than batch.")
class L0aLoad2d:
    """
    class of fmap load2d optimization.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.l0a_load2d_flag

    def align_al1_load2d(self, sch, al1):
        """
        al1 M align.
        """
        if self.flag:
            sch[al1].storage_align(al1.op.axis[1], 256, 0)

    def load2d_emit_insn(self, sch, al1, al0):
        """
        Emit insn for al1 and al0 when al1_load2d is enabled.
        """
        sch[al1].emit_insn(al1.op.axis[0], "dma_copy")
        sch[al0].emit_insn(al0.op.axis[0], "dma_copy")


class StridehOpti:
    """
    class of stride_h optimization when kernel_h = 1 and stride_h > 1.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.strideh_opti_flag
        self.stride_h_update = 1 if self.flag else conv_param.stride_h


class C04Opti:
    """
    Class of C04 optimization.
    """
    def __init__(self, conv_param):
        self.mode = conv_param.v220_c04_mode
        self.flag = conv_param.v220_c04_mode != "disabled"


class Conv1dSplitw:
    """
    Class of Conv1d.
    """
    def __init__(self, conv_param):
        self.flag = conv_param.conv1d_split_w_flag

    def align_row_major_conv1d(self, sch, fmap_row_major, block_k0):
        """
        Buffer align row major in Conv1d.
        """
        sch[fmap_row_major].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, block_k0))


class Conv2dSchedule:
    """
    Class of Conv2d Schedule.
    """
    def __init__(self, sch, res, conv_param, op_graph, tiling_dict_flag, tiling_case, var_range):
        self._sch = sch
        self._res = res
        self._conv_param = conv_param
        self._op_graph = op_graph

        self._tensor_map = conv_param.tensor_map
        self._dim_map = conv_param.dim_map
        self._para_dict = conv_param.para_dict

        # dtype
        self._fmap_dtype = self._tensor_map["fmap"].dtype
        self._weight_dtype = self._tensor_map["filter"].dtype
        self._res_dtype = res.dtype

        # frac unit size
        self._block_m0, self._block_k0, self._block_n0 = CUBE_MKN[self._weight_dtype]["mac"]

        # conv2d params
        self._kernel_h = conv_param.filter_h
        self._kernel_w = conv_param.filter_w
        self._stride_h = conv_param.stride_h
        self._stride_w = conv_param.stride_w
        self._dilate_h = conv_param.dilate_h
        self._dilate_w = conv_param.dilate_w

        self._out_height = conv_param.h_out
        self._out_width = conv_param.w_out
        self._out_hw = self._out_height*self._out_width

        self._group_opt = self._para_dict["group_opt"]
        self._ci1_opt = self._para_dict["c1_opt"]
        self._co1_opt = self._para_dict["cout1_opt"]
        self._batch, self._in_c1, self._in_height, self._in_width, self._in_c0 = self._para_dict["a_shape"]
        self._quant_fusion_muti_groups_in_cl0 = self._co1_opt % 2 == 1 and self._group_opt > 1 and res.dtype == "int8"
        self._bias_tensor = self._para_dict["bias_tensor"]
        self._bias_flag = self._bias_tensor is not None

        # self._tiling params
        self._tiling_query_param = conv_param.tiling_query_param
        self._tiling = {}
        self._fusion_type = op_graph.fusion_type

        # device params
        self._core_num = get_soc_spec("CORE_NUM")

        #====================create feature instance=============================
        self._fixpipe_fusion = FixpipeFusionNew() if is_support_fixpipe_op() else FixpipeFusion()
        self._input_nd2nz = InputNd2Nz(conv_param)
        self._weight_nd2nz = WeightNd2Nz(conv_param)
        self._output_nz2nd = OutputNz2Nd(res)
        self._lx_fusion = LxFusion(conv_param)
        self._aipp_fusion = AippFusion(conv_param)
        self._dynamic_shape = DynamicShape(conv_param, tiling_dict_flag, tiling_case, var_range)
        self._strided_read = StridedRead(conv_param)
        self._strided_write = StridedWrite(res)
        self._im2col_dma = Im2colDma(conv_param)
        self._inner_batch = InnerBatch()
        self._l0a_load2d = L0aLoad2d(conv_param)
        self._strideh_opti = StridehOpti(conv_param)
        self._c04 = C04Opti(conv_param)
        self._conv1d = Conv1dSplitw(conv_param)

        #===================parse fusion pattern========================
        self._fixpipe_fusion.parse_fusion_pattern(res)

        if self._aipp_fusion.flag:
            self._fmap_dtype = "float16"

    def fetch_info_dict(self, tiling_case):
        """
        Fetch the info_dict to get tiling.
        """
        if self._dynamic_shape.flag and tiling_case: # pass when tiling_case
            return None

        tiling_query_param = self._tiling_query_param
        conv_param = self._conv_param

        fmap_shape_nc1hwc0 = list(tiling_query_param["fmap_shape_nc1hwc0"])
        shape_w_nc1hwc0 = list(tiling_query_param["shape_w_nc1hwc0"])
        c_shape = tiling_query_param["c_shape"]
        mad_dtype = tiling_query_param["mad_dtype"]
        bias_flag = tiling_query_param["bias_flag"]
        quant_pre_flag, relu_pre_flag, quant_post_flag, relu_post_flag, anti_quant_flag = self._fixpipe_fusion.fetch_quant_relu_flag()
        eltwise_flag = False
        eltwise_dtype = "float16"
        if is_support_fixpipe_op():
            eltwise_flag, eltwise_dtype = self._fixpipe_fusion.fetch_eltwise_info()

        # group conv, send one group_opt a, b, c shape to tiling
        info_dict = {"op_type": 'conv2d',
                     "a_shape": fmap_shape_nc1hwc0,
                     "b_shape": shape_w_nc1hwc0,
                     "c_shape": c_shape,
                     "a_dtype": self._fmap_dtype,
                     "b_dtype": self._weight_dtype,
                     "c_dtype": self._res_dtype,
                     "mad_dtype": mad_dtype,
                     "pad": [conv_param.pad_w[0], conv_param.pad_w[1],
                             conv_param.pad_h[0], conv_param.pad_h[1]],
                     "stride": [self._stride_h, self._stride_w],
                     "dilation": [self._dilate_h, self._dilate_w],
                     "group": self._group_opt,
                     "bias_flag": bias_flag,
                     "fixpipe_fusion_flag_dict": {"quant_pre_flag": quant_pre_flag,
                                                  "relu_pre_flag": relu_pre_flag,
                                                  "quant_post_flag": quant_post_flag,
                                                  "relu_post_flag": relu_post_flag,
                                                  "anti_quant_flag": anti_quant_flag},
                     "eltwise_dict": {
                         "eltwise_flag": eltwise_flag,
                         "eltwise_dtype": eltwise_dtype
                     },
                     "in_fm_memory_type": self._lx_fusion.input_memory_type,
                     "out_fm_memory_type": self._lx_fusion.output_memory_type,
                     "l1_fusion_type": self._lx_fusion.l1_fusion_type,
                     "fm_l1_valid_size": self._lx_fusion.fmap_l1_valid_size,
                     "fusion_type": self._fusion_type,
                     "kernel_name": conv_param.kernel_name,
                     "special_mode": {"use_c04_mode": 2 if self._c04.flag else 0, # 3 for v220 c04
                                      # disable strideh opti when input nd2nz.
                                      "input_nd_flag": self._input_nd2nz.flag},
                     "placeholder_fmap_5hd_shape": list(self._dim_map["fmap_5hd_shape"]),
                     #=============to be deleted====================
                     "fused_coefficient": [0, 0, 0],
                     "fused_channel_wise": [0, 0, 0],
                     "pooling_shape": [0, 0],
                     "pooling_stride": [0, 0],
                     }
        return info_dict

    def fetch_tiling(self, info_dict, tiling_case):
        """
        Fetch tiling info.
        """
        def get_default_tiling():
            """
            Set default tiling when fetch tiling failed.
            """
            default_tiling = {}
            tiling_m = 1
            tiling_k = 1
            if self._quant_fusion_muti_groups_in_cl0:
                tiling_n = self._co1_opt
                group_cl0 = 2
            else:
                tiling_n = 2
                group_cl0 = 1

            default_tiling["AL1_shape"] = [1, 1, 1, 1]
            default_tiling["BL1_shape"] = None

            default_tiling["AL0_matrix"] = [tiling_m, tiling_k, 16, self._in_c0, 1, 1]
            default_tiling["BL0_matrix"] = [tiling_k, tiling_n, 16, self._in_c0, 1, 1]
            default_tiling["CL0_matrix"] = [tiling_n, tiling_m, 16, 16, 1, group_cl0]
            default_tiling["manual_pingpong_buffer"] = {'AL1_pbuffer': 1,
                                                        'BL1_pbuffer': 1,
                                                        'AL0_pbuffer': 1,
                                                        'BL0_pbuffer': 1,
                                                        'CL0_pbuffer': 1,
                                                        'AUB_pbuffer': 1,
                                                        'BUB_pbuffer': 1,
                                                        'CUB_pbuffer': 1,
                                                        'UBG_pbuffer': 1}
            default_tiling["n_bef_batch_flag"] = False
            default_tiling["A_overhead_opt_flag"] = False
            default_tiling["B_overhead_opt_flag"] = False

            default_tiling["block_dim"] = [1, 1, 1, 1]
            if self._batch > 1 and self._core_num > 1:
                if self._batch <= self._core_num:
                    default_tiling["block_dim"][0] = self._batch
                else:
                    for i in range(self._core_num, 0, -1):
                        if self._batch % i == 0:
                            break
                    default_tiling["block_dim"][0] = i
            else:
                default_tiling["block_dim"][0] = 1

            default_tiling = self._lx_fusion.config_default_tiling(default_tiling)

            return default_tiling

        def get_v220_tiling(info_dict):
            """
            Get tiling in v220 situation.
            """
            tiling = tiling_api.get_tiling(info_dict)
            tiling["CUB_matrix"] = tiling["CL0_matrix"]
            if tiling is None or tiling["AL0_matrix"][2] == 32:
                log.warn("get invalid tiling, default tiling will be used")
                tiling = get_default_tiling()

            tiling["CUB_matrix"] = tiling["CL0_matrix"]
            return tiling

        if self._dynamic_shape.flag and tiling_case:
            self._tiling = self._dynamic_shape.fetch_tiling_case()
        else:
            self._tiling = get_v220_tiling(info_dict)

    def verify_tiling(self):
        """
        Verify whether the tiling returned is legal.
        """
        def check_l0_tiling():
            """
            Check al0 tiling and bl0 tiling.
            """
            if ma_al0 != mc_cl0:
                err_man.raise_err_equal_invalid("conv2d", "ma", "mc")

            if tiling["BL0_matrix"]:
                kb_bl0, nb_bl0, _, _, _, _ = tiling["BL0_matrix"]
                if ka_al0 != kb_bl0:
                    err_man.raise_err_equal_invalid("conv2d", "ka", "kb")
                if nb_bl0 != nc_cl0:
                    err_man.raise_err_equal_invalid("conv2d", "nb", "nc")

        def check_overhead_opt_flag():
            """
            Check the overhead_opt_flag.
            """
            if group_cl0 > 1 and tiling["A_overhead_opt_flag"]:
                err_man.raise_err_value_or_format_invalid(
                    "conv2d", 'tiling["A_overhead_opt_flag"]', "False", "when multi_cl0_group.")

        def modify_bl0_tiling():
            """
            Modify bl0 tiling in certain circumstances.
            """
            filter_matrix = list(self._dim_map["filter_matrix_dim"]) # [k1, n1, n0, k0]
            filter_matrix[1] = filter_matrix[1] // n_dim

            if self._tiling["BL0_matrix"]:
                if self._tiling["BL0_matrix"][0: 4] == filter_matrix and group_bl0 == 1:
                    self._tiling["BL0_matrix"] = []

        def modify_bl1_tiling():
            """
            Modify bl1 tiling in certain circumstances.
            """
            if self._tiling["BL0_matrix"] == []:
                if ceil_div(self._group_opt, group_dim) == 1:
                    self._tiling["BL1_shape"] = None
                elif ceil_div(self._group_opt, group_dim) > 1 and self._tiling["BL1_shape"] is not None:
                    self._tiling["BL1_shape"] = []

        tiling = self._tiling
        bl0_tiling = tiling["BL0_matrix"]
        ma_al0, ka_al0, _, _, _, _ = tiling["AL0_matrix"]
        nc_cl0, mc_cl0, _, _, batch_cl0, group_cl0 = tiling["CL0_matrix"]
        if bl0_tiling:
            _, _, _, _, _, group_bl0 = bl0_tiling

        _, n_dim, _, group_dim = tiling["block_dim"]

        self._lx_fusion.check_l1fusion_tiling(tiling)

        self._weight_nd2nz.check_bl1_nd2nz_tiling(tiling)

        self._dynamic_shape.check_dynamic_overhead_opt_flag(tiling)

        self._inner_batch.set_l0b_innerbatch_flag(tiling, batch_cl0)
        self._inner_batch.check_innerbatch_tiling(tiling, self._batch, self._out_hw)

        check_l0_tiling()
        check_overhead_opt_flag()

        #==================modify tiling to be deleted=========================
        modify_bl0_tiling()
        modify_bl1_tiling()

    def config_scope(self):
        """
        Config tensor scope.
        """
        def config_cl0():
            """
            Config cl0 scope.
            """
            cl0 = tensor_map["cl0"]
            sch[cl0].set_scope(cce.scope_cc)
            return cl0

        def config_fmap_row_major():
            """
            Config row major scope.
            """
            if self._dynamic_shape.flag or self._l0a_load2d.flag:
                return None
            fmap_row_major = tensor_map["fmap_row_major"]
            sch[fmap_row_major].set_scope(cce.scope_cbuf)
            return fmap_row_major

        def config_al1():
            """
            Config al1 scope.
            """
            scope_al1 = self._lx_fusion.config_al1_scope()
            al1_already_exist_flags = (self._dynamic_shape.flag,
                                       self._l0a_load2d.flag,
                                       self._strideh_opti.flag,
                                       self._input_nd2nz.flag,
                                       self._strided_read.flag,
                                       self._aipp_fusion.flag,
                                       self._c04.mode == "not_first_layer_c04")
            for flag in al1_already_exist_flags:
                if flag:
                    al1 = tensor_map["fmap_l1"]
                    sch[al1].set_scope(scope_al1)
                    return al1

            al1 = sch.cache_read(fmap, scope_al1, [fmap_row_major])
            return al1

        def config_al0():
            """
            Config al0 scope.
            """
            if self._im2col_dma.flag:
                return self._im2col_dma.config_al0_im2coldma(sch, al1_im2col, cl0)

            al0 = tensor_map["fmap_im2col"]
            sch[al0].set_scope(cce.scope_ca)
            return al0

        def config_bl1():
            """
            Config bl1 scope.
            """
            if self._tiling["BL1_shape"] is None:
                bl1 = None
            elif self._weight_nd2nz.flag:
                bl1 = weight
                sch[bl1].set_scope(cce.scope_cbuf)
            else:
                bl1 = sch.cache_read(weight, cce.scope_cbuf, [cl0])
            return bl1

        def config_bl0():
            """
            Config bl0 scope.
            """
            if self._tiling["BL1_shape"] is None:
                bl0 = sch.cache_read(weight, cce.scope_cb, [cl0])
            else:
                bl0 = sch.cache_read(bl1, cce.scope_cb, [cl0])
            return bl0

        def config_bias():
            """
            Config bias scope.
            """
            if self._bias_flag:
                bias_l1 = sch.cache_read(self._bias_tensor, cce.scope_cbuf, [cl0])
                bias_bt = sch.cache_read(bias_l1, cce_params.scope_bt, [cl0])
                return bias_l1, bias_bt

            return None, None

        #========set scope && cache_read && cache_write==========
        tensor_map = self._tensor_map
        sch = self._sch

        fmap = tensor_map["fmap"]
        weight = tensor_map["filter"]
        fmap_row_major_reshape = tensor_map.get("fmap_row_major_reshape", None)
        al1_im2col = self._im2col_dma.config_al1_im2col(sch, tensor_map)

        cl0 = config_cl0()
        fmap_row_major = config_fmap_row_major()
        al1 = config_al1()
        al0 = config_al0()
        bl1 = config_bl1()
        bl0 = config_bl0()
        bias_l1, bias_bt = config_bias()

        self._fixpipe_fusion.fixpipe_inputs_set_scope(sch, self._op_graph)

        tensor_param = {"al1": al1, "bl1": bl1,
                        "fmap": fmap, "weight": weight,
                        "fmap_row_major": fmap_row_major, "fmap_row_major_reshape": fmap_row_major_reshape,
                        "al1_im2col": al1_im2col,
                        "al0": al0, "bl0": bl0, "cl0": cl0,
                        "bias_l1": bias_l1, "bias_bt": bias_bt}
        return tensor_param

    def special_process_pre(self, res, tensor_param):
        """
        Special process before tiling is parsed.
        """
        def align_al1():
            """
            Align al1 in various situation.
            """
            if self._lx_fusion.l1_fusion_type == BREADTH_L1_FUSION:
                return self._lx_fusion.align_al1_lxfusion(sch, al1)
            if self._l0a_load2d.flag:
                return self._l0a_load2d.align_al1_load2d(sch, al1)
            return None

        def align_row_major():
            """
            Align row major in various situation.
            """
            if self._dynamic_shape.flag or self._l0a_load2d.flag:
                return None
            if self._conv1d.flag:
                return self._conv1d.align_row_major_conv1d(sch, fmap_row_major, self._block_k0)

            sch[fmap_row_major].buffer_align(
                (1, 1),
                (1, 1),
                (self._out_width, self._out_width),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 4 if self._c04.flag else self._block_k0))
            return None

        sch = self._sch
        al1 = tensor_param["al1"]
        fmap_row_major = tensor_param["fmap_row_major"]
        fmap_row_major_reshape = tensor_param["fmap_row_major_reshape"]
        al1_im2col = tensor_param["al1_im2col"]

        align_al1()
        align_row_major()

        self._fixpipe_fusion.inline_fixpipe_tensor(sch)
        if res.op.name == "res_fp32_conv2d" and is_support_fixpipe_op():
            # for single conv2d inline res_conv2d
            sch[res.op.input_tensors[0]].compute_inline()

        # inline row_major_reshape
        if fmap_row_major_reshape is not None:
            sch[fmap_row_major_reshape].compute_inline()

        self._im2col_dma.inline_al1_im2coldma(sch, al1, fmap_row_major)
        self._im2col_dma.align_al1_im2col(sch, al1_im2col, self._block_k0)
        self._strided_read.process_strided_read(sch, al1, self._strideh_opti.flag, self._l0a_load2d.flag)

        self._strided_write.process_strided_write(sch, self._res)
        # inline input_nd
        self._input_nd2nz.inline_input_nd_dynamic(sch, self._tensor_map, self._dynamic_shape.flag)

        # dynamic shape
        self._dynamic_shape.handle_var_range(sch)
        self._dynamic_shape.disable_memory_reuse(sch, tensor_param)

    def tile_attach_tensor(self, res, tensor_param):
        """
        Split tensor axis and attach tensors.
        """
        def bl0_compute_at():
            """
            Handle bl0 attach.
            """
            if bl0_tiling or (bl0_tiling == [] and multi_cl0_group):
                if multi_bl0_group and group_cl0 == 1:
                    sch[bl0].compute_at(sch[res], res_gio)
                else:
                    sch[bl0].compute_at(sch[cl0], cl0_co)
            else:
                sch[bl0].compute_at(sch[res], res_gio)

        def al1_compute_at():
            """
            Handle al1 attach.
            """
            def get_al1_attach_info():
                """
                Get the consumer tensor of al1 and the target axis to be attached.
                """
                if multi_cl0_group:
                    return cl0, al1_at_cl0_axis

                if al1_tiling:
                    if al1_nparts[0] != 1:
                        return cl0, al1_at_cl0_axis
                    return res, al1_at_res_axis

                # al1 full load
                if self._conv1d.flag:
                    return res, al1_at_res_axis

                if self._lx_fusion.l1_fusion_type == DEPTH_L1_FUSION:
                    return res, res_gio

                return res, res_nioo

            consumer, target_axis = get_al1_attach_info()
            sch[al1].compute_at(sch[consumer], target_axis)
            if fmap_row_major is not None:
                sch[fmap_row_major].compute_at(sch[consumer], target_axis)

        def bl1_compute_at():
            """
            Handle bl1 attach.
            """
            def get_bl1_attach_info():
                """
                Get the consumer tensor of bl1 and the target axis to be attached.
                """
                if self._inner_batch.flag:
                    return res, res_ciooi

                if bl1_tiling:
                    if bl1_nparts[0] != 1:
                        # cl0_gi is under cl0_ko
                        return cl0, bl1_at_cl0_axis
                    if bl1_nparts[0] == 1 and multi_cl0_group:
                        return cl0, cl0_co
                    if group_cl0 == 1 and multi_bl0_group:
                        return res, res_gio
                    return res, bl1_at_res_axis

                if bl1_tiling == [] and multi_cl0_group:
                    return cl0, cl0_co

                return res, res_gio

            if bl1_tiling is not None:
                consumer, target_axis = get_bl1_attach_info()
                sch[bl1].compute_at(sch[consumer], target_axis)

        def bias_compute_at():
            """
            Handle bias attach.
            """
            if self._bias_flag:
                sch[bias_l1].compute_at(sch[res], fixpipe_slice_axis)
                sch[bias_bt].compute_at(sch[res], fixpipe_slice_axis)

        def anti_quant_spilt_flag(res):
            if res.dtype == "float16" and not self._fixpipe_fusion.nz2nd_flag and \
                    self._fixpipe_fusion.anti_quant_flag:
                return True

            return False

        #==========================parse tiling==================================
        al1_tiling = self._tiling["AL1_shape"]
        bl1_tiling = self._tiling["BL1_shape"]
        al0_tiling = self._tiling["AL0_matrix"]
        bl0_tiling = self._tiling["BL0_matrix"]
        cl0_tiling = self._tiling["CL0_matrix"]
        pingpong_buffer = self._tiling["manual_pingpong_buffer"]
        batch = self._batch
        kernel_h = self._kernel_h
        kernel_w = self._kernel_w
        dilate_h = self._dilate_h
        dilate_w = self._dilate_w
        block_k0 = self._block_k0
        ci1_opt = self._ci1_opt
        co1_opt = self._co1_opt
        out_hw = self._out_hw

        # only bl1_tiling can be None, only al0_tiling cannot be []

        if al0_tiling:
            ma_al0, ka_al0, _, _, _, _ = al0_tiling

        if bl0_tiling:
            _, nb_bl0, _, _, _, group_bl0 = bl0_tiling

        if cl0_tiling:
            nc_cl0, mc_cl0, m0_cl0, _, batch_cl0, group_cl0 = cl0_tiling

        batch_al1 = 1

        if al1_tiling:
            k_al1, multi_m_al1, batch_al1, _ = al1_tiling
            k1_al1 = k_al1 // (((kernel_h - 1)*dilate_h + 1)*((kernel_w - 1)*dilate_w + 1)*block_k0)
            if k1_al1 == 0:
                k1_al1 = 1

        if bl1_tiling:
            k_bl1, multi_n_bl1, _, _ = bl1_tiling
            k1_bl1 = k_bl1 // (kernel_h*kernel_w*block_k0)

        batch_dim, n_dim, m_dim, group_dim = self._tiling["block_dim"]

        #==========================calculate various coefficients==================================
        multi_bl0_group = bl0_tiling and group_bl0 > 1
        multi_cl0_group = group_cl0 > 1

        if multi_bl0_group or multi_cl0_group:
            cl0_factor = [1, ceil_div(out_hw, mc_cl0*m0_cl0)]
        else:
            cl0_factor = [ceil_div(co1_opt, nc_cl0), ceil_div(out_hw, mc_cl0*m0_cl0)]

        if al1_tiling:
            al1_nparts = [ci1_opt // k1_al1, ceil_div(cl0_factor[1], multi_m_al1)]
        else: # al1 full load
            al1_nparts = [1, 1]

        if bl1_tiling:
            if cl0_factor[0] % multi_n_bl1 != 0:
                err_man.raise_err_specific("conv2d", "second value of BL1_shape should be factor of n block num")
            if multi_n_bl1 > 1 and multi_n_bl1 % 2 != 0:
                err_man.raise_err_specific("conv2d", "second value of BL1_shape better to be even number")

            bl1_nparts = [(ci1_opt + k1_bl1 - 1) // k1_bl1,
                          (cl0_factor[0] + multi_n_bl1 - 1) // multi_n_bl1]
        else:
            bl1_nparts = [1, n_dim]

        #===========================split and compute at=========================================
        sch = self._sch
        al1 = tensor_param["al1"]
        bl1 = tensor_param["bl1"]
        al0 = tensor_param["al0"]
        bl0 = tensor_param["bl0"]
        cl0 = tensor_param["cl0"]
        fmap_row_major = tensor_param["fmap_row_major"]
        bias_l1 = tensor_param["bias_l1"]
        bias_bt = tensor_param["bias_bt"]

        # tile
        #===================================tile al0============================================
        # tile al0 for load3d
        al0_mo, al0_mi = sch[al0].split(al0.op.axis[2], ma_al0)
        al0_ko, al0_ki = sch[al0].split(al0.op.axis[3], ka_al0)
        al0_no, al0_ni = sch[al0].split(al0.op.axis[1], 1)

        sch[al0].reorder(al0.op.axis[0], # group
                         al0_no, # batch.outer
                         al0_mo, # m_1.outer
                         al0_ko, # k_1.outer
                         al0_ni, # batch.inner = 1
                         al0_mi, # m_1.inner
                         al0_ki, # k_1.inner
                         al0.op.axis[4], # m_0
                         al0.op.axis[5]) # k_0
        al0_axis_list = [al0_no, al0_mo, al0_ko,
                         al0_ni, al0_mi, al0_ki, al0.op.axis[4], al0.op.axis[5]] # axis for im2col

        #===================================tile res============================================
        def set_reorder_mn_flag():
            """
            Reorder axis m and n to achieve better performance.
            """
            def reorder_res_mn_axis():
                """
                Reorder axis m and n according to various flags.
                """
                if self._inner_batch.flag:
                    if reorder_mn_flag:
                        sch[res].reorder(res_nioo, res_mooi, res_nioi, res_ciooi)
                    else:
                        sch[res].reorder(res_nioo, res_ciooi, res_mooi, res_nioi)
                else:
                    # res_ciooi means nparts of co1 axis loading into L1 in single core. (N axis)
                    # res_mooi means nparts of howo axis loading into L1 in single core. (M axis)
                    if reorder_mn_flag:
                        sch[res].reorder(res_nioo, res_mooi, res_nioi, res_ciooi, res_cioi, res_nii)
                        # False ori version is (res_nioo, res_mooi, res_nioi, res_cioi, res_ciooi, res_nii)
                    else:
                        sch[res].reorder(res_nioo, res_ciooi, res_mooi, res_cioi, res_nioi, res_nii)

            reorder_mn_flag = False
            if not bl1_tiling:
                reorder_mn_flag = True
            elif pingpong_buffer["AL1_pbuffer"] == pingpong_buffer["BL1_pbuffer"]:
                if not self._dynamic_shape.flag and bl1_nparts[1] >= al1_nparts[1]:
                    reorder_mn_flag = True
            elif pingpong_buffer["BL1_pbuffer"] == 2:
                reorder_mn_flag = True

            reorder_res_mn_axis()

            return reorder_mn_flag

        if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
            res_n_axis, res_hw_axis, res_c_axis = res.op.axis # [n, howo, co]
            # split c axis into c1 and c0 to avoid nonlinear ir
            res_c1_axis, res_c0_axis = sch[res].split(res_c_axis, 16) # [n, howo, co1, co0]
        else:
            res_n_axis, res_c1_axis, res_hw_axis, res_c0_axis = res.op.axis # [n, co1, howo, co0]

        if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
            res_g, res_co1_ori = sch[res].split(res_c1_axis, factor=co1_opt)
            res_cio, res_cii = sch[res].split(res_co1_ori, nc_cl0)
        elif res.dtype == "int8":
            _, _ = sch[res].split(res_c0_axis, factor=16) # split c0=16 in channel merging to avoid nonlinear ir
            if multi_cl0_group:
                res_g, res_co1_ori = sch[res].split(res_c1_axis, factor=co1_opt*group_cl0 // 2)
                res_cio, res_cii = sch[res].split(res_co1_ori, factor=co1_opt*group_cl0 // 2)
            else:
                res_g, res_co1_ori = sch[res].split(res_c1_axis, factor=ceil_div(co1_opt, 2))
                res_cio, res_cii = sch[res].split(res_co1_ori, nc_cl0 // 2)
        elif res.dtype == "float32":
            res_g, res_co1_ori = sch[res].split(res_c1_axis, factor=co1_opt*2)
            res_co1_ori_o, res_co1_ori_i = sch[res].split(res_co1_ori, 2) # res_co1_ori_i = 2
            res_cio, res_cii = sch[res].split(res_co1_ori_o, nc_cl0)
        else:
            res_g, res_co1_ori = sch[res].split(res_c1_axis, factor=co1_opt)
            res_cio, res_cii = sch[res].split(res_co1_ori, nc_cl0)
            # fp16 and not nd2nz and anti_quant, split 2 for pass claculate c0 stride
            if anti_quant_spilt_flag(res):
                res_cii_ori = res_cii
                res_cii, res_ciii = sch[res].split(res_cii_ori, 2)

        res_mo, res_mi = sch[res].split(res_hw_axis, mc_cl0*m0_cl0)

        if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
            sch[res].reorder(res_g,   # group_opt
                             res_cio, # co1_opt // nc
                             res_mo,  # howo // (mc*m0)
                             res_mi,  # mc*m0
                             res_cii) # nc
            # [n, group_opt, co1_opt // nc, howo // (mc*m0), ||| mc*m0, nc, co0]
        elif res.dtype == "float32":
            sch[res].reorder(res_g,
                             res_cio,
                             res_mo,
                             res_cii,       # nc
                             res_co1_ori_i, # 2
                             res_mi)        # mc*m0
            # [n, group_opt, co1_opt // 2*nc, howo // (mc*m0), ||| nc, 2, mc*m0, co0]
        else:
            if res.dtype == "float16":
                if anti_quant_spilt_flag(res):
                    sch[res].reorder(res_cio,   # co1_opt // nc
                                     res_mo,    # howo // (mc*m0)
                                     res_cii,  # nc // 2
                                     res_ciii,  # 2
                                     res_mi)    # mc*m0
                else:
                    sch[res].reorder(res_cio,  # co1_opt // nc
                                     res_mo,  # howo // (mc*m0)
                                     res_cii,  # nc
                                     res_mi)  # mc*m0
            else:
                sch[res].reorder(res_cio, # co1_opt // nc
                                 res_mo,  # howo // (mc*m0)
                                 res_cii, # nc
                                 res_mi)  # mc*m0
            # [n, group_opt, co1_opt // nc, howo // (mc*m0), ||| nc, mc*m0, co0]

        res_moo, res_moi = sch[res].split(res_mo, nparts=al1_nparts[1])
        # when multi_cl0_group, cl0_factor[0] is 1 and bl1_nparts[1] is 1
        res_cioo, res_cioi = sch[res].split(res_cio, nparts=bl1_nparts[1])

        # split batch of res
        if self._dynamic_shape.n_dynamic:
            batch_dim_factor = tvm.max(1, ceil_div(batch, batch_dim))
            res_no, res_ni = sch[res].split(res_n_axis, batch_dim_factor)
        else:
            res_no, res_ni = sch[res].split(res_n_axis, nparts=batch_dim)

        res_go, res_gi = sch[res].split(res_g, nparts=group_dim)

        res_nio, res_nii = sch[res].split(res_ni, self._inner_batch.config_innerbatch_axis(batch_cl0, batch_al1))

        if group_cl0 == 1 and multi_bl0_group:
            res_gio, res_gii = sch[res].split(res_gi, factor=group_bl0)
        else:
            res_gio, res_gii = sch[res].split(res_gi, 1)

        # split cout of res
        res_ciooo, res_ciooi = sch[res].split(res_cioo, nparts=n_dim)
        res_mooo, res_mooi = sch[res].split(res_moo, nparts=m_dim)

        bl1_at_res_axis = res_ciooi
        al1_at_res_axis = res_mooi

        if self._inner_batch.flag:
            if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
                pass
            elif res.dtype == "float32":
                sch[res].reorder(res_no,
                                 res_go,
                                 res_ciooo,
                                 res_mooo,
                                 res_gio,
                                 res_gii,
                                 res_nio,
                                 res_ciooi,
                                 res_mooi,
                                 res_cioi,
                                 res_moi,
                                 res_cii,
                                 res_nii,
                                 res_co1_ori_i, # third axis from bottom must be 2 for fp32
                                 res_mi)
            else:
                if anti_quant_spilt_flag(res):
                    sch[res].reorder(res_no,
                                     res_go,
                                     res_ciooo,
                                     res_mooo,
                                     res_gio,
                                     res_gii,
                                     res_nio,
                                     res_ciooi,
                                     res_mooi,
                                     res_cioi,
                                     res_moi,
                                     res_cii,
                                     res_ciii,
                                     res_nii,
                                     res_mi)
                else:
                    sch[res].reorder(res_no,
                                     res_go,
                                     res_ciooo,
                                     res_mooo,
                                     res_gio,
                                     res_gii,
                                     res_nio,
                                     res_ciooi,
                                     res_mooi,
                                     res_cioi,
                                     res_moi,
                                     res_cii,
                                     res_nii,
                                     res_mi)
        else:
            sch[res].reorder(res_no,
                             res_go,
                             res_ciooo,
                             res_mooo,
                             res_gio,
                             res_gii,
                             res_nio,
                             res_ciooi,
                             res_mooi,
                             res_nii,
                             res_cioi)

        cl0_at_res_axis = res_moi # c_slice_axis

        if self._output_nz2nd.flag or self._fixpipe_fusion.nz2nd_flag:
            res_pragma_axis = res_mi
        else:
            res_pragma_axis = res_cii

        blocks = batch_dim*n_dim*m_dim*group_dim

        if blocks != 1:
            multicore_axis = sch[res].fuse(
                res_no,
                res_go,
                res_ciooo,
                res_mooo)
            if self._dynamic_shape.flag:
                multicore_axis_o, _ = sch[res].split(multicore_axis, factor=1)
            else:
                multicore_axis_o, _ = sch[res].split(multicore_axis, nparts=blocks)

            bindcore_axis, multicore_axis_oi = sch[res].split(multicore_axis_o, 1)
            sch[res].bind(bindcore_axis, tvm.thread_axis("blockIdx.x"))

            if blocks == batch_dim:
                sch[res].pragma(multicore_axis_oi, 'json_info_batchBindOnly', 1)
        else:
            bindcore_axis = res_no

        res_nioo, res_nioi = sch[res].split(res_nio, factor=1)

        reorder_mn_flag = set_reorder_mn_flag()

        if self._tiling["n_bef_batch_flag"] and not reorder_mn_flag and not self._inner_batch.flag:
            sch[res].reorder(res_ciooi, res_nioo)

        #=====================================tile cl0==============================================
        cl0_k1, cl0_k0 = cl0.op.reduce_axis

        cl0_mo, cl0_mi = sch[cl0].split(sch[cl0].op.axis[3], ma_al0*self._block_m0)

        al0_at_cl0_axis = cl0_mo

        if bl0_tiling == []:
            cl0_co, cl0_ci = sch[cl0].split(sch[cl0].op.axis[2], nparts=1)
        else:
            cl0_co, cl0_ci = sch[cl0].split(sch[cl0].op.axis[2], nb_bl0)

        if multi_cl0_group and multi_bl0_group:
            cl0_go, cl0_gi = sch[cl0].split(cl0.op.axis[0], factor=group_bl0)

        # for reduce axis, al0 and bl0 should be the same
        cl0_ko, cl0_ki = sch[cl0].split(cl0_k1, ka_al0)
        cl0_no, cl0_ni = sch[cl0].split(cl0.op.axis[1], 1)
        cl0_co0 = cl0.op.axis[4]

        if self._inner_batch.flag:
            sch[cl0].reorder(cl0_ko,
                             cl0_co,
                             cl0_no,
                             cl0_mo,
                             cl0_ni,
                             cl0_ci,
                             cl0_mi,
                             cl0_co0,
                             cl0_ki,
                             cl0_k0)
        else:
            sch[cl0].reorder(cl0_ko,
                             cl0_co,
                             cl0_mo,
                             #=======L0C tiling=========
                             cl0_ni, # 1
                             cl0_ci, # nb
                             cl0_mi, # ma*m0
                             cl0_co0, # co0
                             cl0_ki, # ka = kb
                             cl0_k0) # k0

        if multi_cl0_group and multi_bl0_group:
            if self._inner_batch.flag:
                sch[cl0].reorder(cl0_go,
                                 cl0_co,
                                 cl0_gi,
                                 cl0_ko,
                                 cl0_no,
                                 cl0_mo)
            else:
                sch[cl0].reorder(cl0_go,
                                 cl0_no,
                                 cl0_co,
                                 cl0_gi,
                                 cl0_ko,
                                 cl0_mo)

        outer_factor = max(al1_nparts[0], bl1_nparts[0])
        inner_factor = min(al1_nparts[0], bl1_nparts[0])

        if outer_factor % inner_factor != 0:
            err_man.raise_err_specific("conv2d", "illegal value of AL1_shape & BL1_shape")

        if al1_nparts[0] > bl1_nparts[0]:
            cl0_koo, cl0_koi = sch[cl0].split(cl0_ko, nparts=al1_nparts[0])
            cl0_kooo, cl0_kooi = sch[cl0].split(cl0_koo, nparts=bl1_nparts[0])
            al1_at_cl0_axis = cl0_kooi
            bl1_at_cl0_axis = cl0_kooo
        else:
            cl0_koo, cl0_koi = sch[cl0].split(cl0_ko, nparts=bl1_nparts[0])
            cl0_kooo, cl0_kooi = sch[cl0].split(cl0_koo, nparts=al1_nparts[0])
            al1_at_cl0_axis = cl0_kooo
            bl1_at_cl0_axis = cl0_kooi

        #===============================attach=======================================
        fixpipe_slice_axis = cl0_at_res_axis if self._group_opt*self._co1_opt > 16 else bindcore_axis
        self._fixpipe_fusion.fixpipe_inputs_compute_at(sch, res, fixpipe_slice_axis, cl0_at_res_axis)

        sch[cl0].compute_at(sch[res], cl0_at_res_axis)
        sch[al0].compute_at(sch[cl0], al0_at_cl0_axis)
        bl0_compute_at()
        al1_compute_at()
        bl1_compute_at()
        bias_compute_at()

        tiling_param = {"al1_tiling": al1_tiling,
                        "al0_tiling": al0_tiling,
                        "bl1_tiling": bl1_tiling,
                        "bl0_tiling": bl0_tiling,
                        "cl0_tiling": cl0_tiling,
                        "al1_nparts": al1_nparts,
                        "stride_h_update": self._strideh_opti.stride_h_update,
                        "fmap_5hd_shape": self._para_dict["a_shape"],
                        "m_cl0": mc_cl0*m0_cl0,
                        "out_hw": self._out_hw}
        if al1_tiling:
            tiling_param.update({"multi_m_al1": multi_m_al1,
                                 "k_al1": k_al1,
                                 "k1_al1": k1_al1})

        emit_insn_dict = {"al0_axis_list": al0_axis_list,
                          "bindcore_axis": bindcore_axis,
                          "k_outer": [cl0_kooo, cl0_kooi, cl0_koi],
                          "cl0_pragma_axis": cl0_ni,
                          "res_pragma_axis": res_pragma_axis,
                          #================dynamic shape======================
                          "dynamic_al0_pragma_axis": al0_ni,
                          }

        return tiling_param, emit_insn_dict

    def special_process_post(self, res, conv_param, tensor_param, tiling_param, emit_insn_dict):
        """
        Special process before tiling is parsed.
        """
        #===========================prepare params================================================
        fmap = tensor_param["fmap"]
        al1 = tensor_param["al1"]
        cl0 = tensor_param["cl0"]
        pingpong_buffer = self._tiling["manual_pingpong_buffer"]
        cl0_tiling = tiling_param["cl0_tiling"]
        res_pragma_axis = emit_insn_dict["res_pragma_axis"]
        sch = self._sch

        # parse the tbe compile parameter
        sch.tbe_compile_para, preload_flag = util.parse_tbe_compile_para(self._tiling.get("compile_para"))
        if pingpong_buffer["CL0_pbuffer"] == 2 and preload_flag:
            sch[cl0].preload()

        #=================================CL0 buffer align======================================
        _, _, m0_cl0, n0_cl0, _, _ = cl0_tiling

        # CL0 shape: [group_opt, batch, co1_opt, howo, co0]
        sch[cl0].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (self._block_m0, self._block_m0),
            (self._block_n0, self._block_n0),
            (1, 1), # reduce_axis_k1
            (self._block_k0, self._block_k0) # reduce_axis_k0
            )

        # align cl0 memory allocation when channel merging
        if res.dtype in ("int4", "int8"):
            sch[cl0].set_buffer_size(reduce((lambda x, y: x*y), cl0_tiling))

        #=================================lxfusion======================================
        self._lx_fusion.config_l1_tensormap(sch, fmap, al1, self._op_graph)

        #=================================dynamic shape process=====================================
        self._dynamic_shape.set_al1_bound(sch, al1, conv_param, tiling_param,
                                          self._l0a_load2d.flag, self._strideh_opti.flag)
        self._dynamic_shape.set_cl0_bound(sch, cl0, cl0_tiling)
        self._dynamic_shape.res_hw_dynamic_pragma(sch, res, res_pragma_axis)

    def double_buffer(self, tensor_param):
        """
        Enable pingpong buffer.
        """
        pingpong_buffer = self._tiling["manual_pingpong_buffer"]

        del pingpong_buffer["AUB_pbuffer"]
        if "BUB_pbuffer" in pingpong_buffer:
            del pingpong_buffer["BUB_pbuffer"]
        del pingpong_buffer["CUB_pbuffer"]
        del pingpong_buffer["UBG_pbuffer"]

        al1 = tensor_param["al1"]
        bl1 = tensor_param["bl1"]
        al0 = tensor_param["al0"]
        bl0 = tensor_param["bl0"]
        cl0 = tensor_param["cl0"]

        pingpong_map = {"AL1_pbuffer": al1,
                        "BL1_pbuffer": bl1,
                        "AL0_pbuffer": al0,
                        "BL0_pbuffer": bl0,
                        "CL0_pbuffer": cl0}

        for key, value in pingpong_buffer.items():
            if value == 2 and pingpong_map[key] is not None:
                self._sch[pingpong_map[key]].double_buffer()

    def map_insn(self, res, tensor_param, tiling_param, emit_insn_dict):
        """
        Emit insn for each tensor.
        """
        def im2col_emit_insn():
            """
            Emit insn for AL1/Row_major/AL0.
            """
            def config_setfmatrix():
                """
                Emit insn for row_major tensor.
                """
                if self._im2col_dma.flag:
                    return None

                setfmatrix_dict = {
                    "conv_kernel_h": conv_param.filter_h,
                    "conv_kernel_w": conv_param.filter_w,
                    "conv_padding_top": conv_param.padding[0],
                    "conv_padding_bottom": conv_param.padding[1],
                    "conv_padding_left": conv_param.padding[2],
                    "conv_padding_right": conv_param.padding[3],
                    "conv_stride_h": stride_h_update,
                    "conv_stride_w": conv_param.stride_w,
                    "conv_dilation_h": conv_param.dilate_h,
                    "conv_dilation_w": conv_param.dilate_w}

                setfmatrix_dict["conv_fm_c"] = al1.shape[1]*al1.shape[4]
                setfmatrix_dict["conv_fm_h"] = al1.shape[2]
                setfmatrix_dict["conv_fm_w"] = al1.shape[3]

                return setfmatrix_dict

            def al1_emit_insn():
                """
                Emit insn for al1.
                """
                def al1_common_emit_insn(sch, al1):
                    """
                    Emit insn for al1 in common usage.
                    """
                    sch[al1].emit_insn(al1.op.axis[0],
                                       "dma_copy",
                                       {"mem_align": 1})

                if self._im2col_dma.flag:
                    return None
                if self._input_nd2nz.flag:
                    return self._input_nd2nz.al1_nd2nz_emit_insn(sch, al1)
                if self._aipp_fusion.flag:
                    return self._aipp_fusion.al1_aipp_emit_insn(sch, al1)

                return al1_common_emit_insn(sch, al1)

            def al0_emit_insn():
                """
                Emit insn for al0.
                """
                def al0_common_emit_insn(sch, fmap_row_major, al0, setfmatrix_dict, al0_axis_list):
                    """
                    Emit insn for al0 and row major tensor in common usage.
                    """
                    sch[fmap_row_major].emit_insn(fmap_row_major.op.axis[1], 'set_fmatrix', setfmatrix_dict)
                    sch[al0].emit_insn(al0_axis_list[3], 'im2col')

                if self._im2col_dma.flag:
                    return self._im2col_dma.im2col_dma_emit_insn(sch, al1_im2col, al0, al0_axis_list)

                return al0_common_emit_insn(sch, fmap_row_major, al0, setfmatrix_dict, al0_axis_list)

            if self._dynamic_shape.flag:
                self._dynamic_shape.dynamic_mode_im2col_v2(
                    sch, conv_param, tensor_param, tiling_param,
                    emit_insn_dict, self._input_nd2nz.flag, self._l0a_load2d.flag)
            elif self._l0a_load2d.flag:
                self._l0a_load2d.load2d_emit_insn(sch, al1, al0)
            else:
                setfmatrix_dict = config_setfmatrix()
                al1_emit_insn()
                self._lx_fusion.al1_l1fusion_pragma(sch, al1)
                al0_emit_insn()
                get_weight_repeat_number()

        def bl1_emit_insn():
            """
            Emit insn for bl1.
            """
            def bl1_common_emit_insn():
                """
                Emit insn for bl1 in common usage.
                """
                if bl1_tiling is not None:
                    sch[bl1].emit_insn(bl1.op.axis[0], "dma_copy")

            if self._weight_nd2nz.flag:
                return self._weight_nd2nz.bl1_nd2nz_emit_insn(sch, bl1)

            return bl1_common_emit_insn()

        def cl0_emit_insn():
            """
            Emit insn for cl0.
            """
            mad_dict = {"mad_pattern": 2, "k_outer": k_outer}

            if self._fmap_dtype == "float32" and conv_param.impl_mode == "high_performance":
                if tbe.common.platform.platform_info.intrinsic_check_support("Intrinsic_mmad", "h322f32"):
                    mad_dict["hf32"] = 1
                    log.debug("enable HF32 mode")

            sch[cl0].emit_insn(cl0_pragma_axis, 'mad', mad_dict)

        def get_weight_repeat_number():
            """
            Get load repeat number of weight tensor.
            """
            if bl1_tiling is None:
                if bl0_tiling == []:
                    weight_repeat_load_num = 1
                else:
                    weight_repeat_load_num = al1_nparts[1]
            elif bl1_tiling == []:
                weight_repeat_load_num = 1
            else:
                weight_repeat_load_num = al1_nparts[1]
            sch[res].pragma(bindcore_axis, "json_info_weight_repeat", weight_repeat_load_num)

        def res_emit_insn():
            """
            Emit insn for res tensor.
            """
            layout_transform_dict = {
                "int4": "channel_merge",
                "int8": "channel_merge",
                "float32": "channel_split"
            }

            def get_res_insn_str():
                if is_support_fixpipe_op():
                    return "fixpipe_op"
                return "dma_copy"

            def res_channel_merge_split_emit_insn():
                """
                Emit insn for res tensor in channel merge/split situation.
                """
                sch[res].emit_insn(res_pragma_axis, "dma_copy",
                                   attrs={"layout_transform": layout_transform_dict[res.dtype]})

            def res_common_emit_insn():
                """
                Emit insn for res tensor in common usage.
                """
                sch[res].emit_insn(res_pragma_axis, get_res_insn_str())

            if is_support_fixpipe_op():
                return res_common_emit_insn()

            if self._output_nz2nd.flag:
                return self._output_nz2nd.res_nz2nd_emit_insn(sch, res, res_pragma_axis)
            if res.dtype in ("int4", "int8", "float32"):
                return res_channel_merge_split_emit_insn()

            return res_common_emit_insn()

        def bias_emit_insn():
            """
            Emit insn for bias.
            """
            if self._bias_flag:
                sch[bias_l1].emit_insn(bias_l1.op.axis[0], "dma_copy")
                sch[bias_bt].emit_insn(bias_bt.op.axis[0], "dma_copy")

        #=============================prepare params=========================================
        conv_param = self._conv_param
        sch = self._sch
        al1 = tensor_param["al1"]
        bl1 = tensor_param["bl1"]
        al0 = tensor_param["al0"]
        bl0 = tensor_param["bl0"]
        cl0 = tensor_param["cl0"]
        al1_im2col = tensor_param["al1_im2col"]
        fmap_row_major = tensor_param["fmap_row_major"]
        bias_l1 = tensor_param["bias_l1"]
        bias_bt = tensor_param["bias_bt"]

        bl1_tiling = tiling_param["bl1_tiling"]
        bl0_tiling = tiling_param["bl0_tiling"]
        al1_nparts = tiling_param["al1_nparts"]
        stride_h_update = tiling_param["stride_h_update"]

        al0_axis_list = emit_insn_dict["al0_axis_list"]
        bindcore_axis = emit_insn_dict["bindcore_axis"]
        k_outer = emit_insn_dict["k_outer"]
        cl0_pragma_axis = emit_insn_dict["cl0_pragma_axis"]
        res_pragma_axis = emit_insn_dict["res_pragma_axis"]

        #=============================emit insn=========================================
        im2col_emit_insn()

        bl1_emit_insn()

        sch[bl0].emit_insn(bl0.op.axis[0], "dma_copy")

        cl0_emit_insn()

        res_emit_insn()

        bias_emit_insn()

        self._fixpipe_fusion.fixpipe_inputs_emit_insn(sch)


def conv_v220_schedule(sch, res, conv_param, op_graph, tiling_dict_flag, tiling_case=None, var_range=None):
    """
    Schedule for Conv2d v220.
    """
    schedule = Conv2dSchedule(sch, res, conv_param, op_graph, tiling_dict_flag, tiling_case, var_range)

    info_dict = schedule.fetch_info_dict(tiling_case)

    if conv_param.dynamic_flag and tiling_dict_flag:
        return info_dict

    schedule.fetch_tiling(info_dict, tiling_case)

    schedule.verify_tiling()

    tensor_param = schedule.config_scope()

    schedule.special_process_pre(res, tensor_param)

    tiling_param, emit_insn_dict = schedule.tile_attach_tensor(res, tensor_param)

    schedule.special_process_post(res, conv_param, tensor_param, tiling_param, emit_insn_dict)

    schedule.double_buffer(tensor_param)

    schedule.map_insn(res, tensor_param, tiling_param, emit_insn_dict)

    return None
