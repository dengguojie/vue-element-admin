#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
common function for cube schedule
"""
from functools import reduce
import math

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.context import op_context
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.base.operation import in_dynamic
from tbe.dsl.compute import cube_util
from tbe.dsl.static_schedule.util import get_fixpipe_emit_str


def get_all_tensors(res):
    """
    get all tensor
    :param res: tensor
    :return: list
    """

    all_tensor = {}
    leaf_tensor = {}

    def get(tensor):
        """
        find all tensor
        :param tensor: c_gm
        :return: all tensor
        """
        tensor_list = tensor.op.input_tensors
        for one_tensor in tensor_list:
            if not one_tensor.op.input_tensors:
                leaf_tensor[one_tensor.op.name] = tensor
            # check which tensor has not been checked
            if one_tensor.op.name not in all_tensor:
                all_tensor[one_tensor.op.name] = one_tensor
                if one_tensor.op.tag == "conv2d_backprop_input":
                    continue
                get(one_tensor)

    get(res)
    return all_tensor, leaf_tensor


def set_intrinsic_support(tensor_attr):
    """
    get intrinsic support to tensor_attr
    """
    tensor_attr["support_l0c_to_out"] = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    tensor_attr["support_l1_to_bt"] = tbe_platform.intrinsic_check_support("Intrinsic_data_move_l12bt")
    tensor_attr["support_ub_to_l1"] = tbe_platform.intrinsic_check_support("Intrinsic_data_move_ub2l1")
    tensor_attr["support_fixpipe"] = (tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out") or
                                      tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2ub"))
    return tensor_attr


def check_need_5hd_trans_nhwc(res):
    """
    check if output tensor need to be converted 5HD to NHWC with fixpipe
    """
    # if output tensor format is NHWC, the tensor shape is (n, hw, c)
    output_shape = cube_util.shape_to_list(res.shape)
    if len(output_shape) == 3:
        return True
    return False


def fetch_fixpipe_tensor(sch, all_tensor, tensor_map):
    """
    get fixpipe tensor to tensor map
    """
    fixpipe_tensor = None
    for tensor in all_tensor.values():
        if tensor.op.tag == "fixpipe":
            fixpipe_tensor = tensor
            tensor_map["fixpipe_tensor"] = fixpipe_tensor
        elif tensor.op.tag == "fixpipe_reform":
            tensor_map["fixpipe_output"] = tensor
        elif tensor.op.tag == "conv2d_backprop_input":
            tensor_dx_gm = tensor
            tensor_map["tensor_dx_gm"] = tensor_dx_gm
    fixpipe_inputs = []
    if fixpipe_tensor is not None:
        for fixpipe_input_tensor in fixpipe_tensor.op.input_tensors:
            if not fixpipe_input_tensor.op.input_tensors:
                fixpipe_inputs.append(fixpipe_input_tensor)
        sch[fixpipe_tensor].compute_inline()
        sch[tensor_dx_gm].compute_inline()
    return sch, tensor_map


def fetch_requant_fusion_ub_info(sch, tensor_map, tensor_attr):
    """
    get ub info with dx + requant fusion
    """
    tensor_attr["n0_32_flag"] = True
    tensor_attr["quant_fuse"] = True
    tensor_map["data_transfer"] = tensor_map.get("deconv_res").op.input_tensors[0]
    tensor_map["c_ub"] = tensor_map.get("data_transfer").op.input_tensors[0]
    tensor_map["deq"] = tensor_map.get("c_ub").op.input_tensors[1]
    c_ub_ddr = tensor_map.get("c_ub").op.input_tensors[0]
    c_ub = c_ub_ddr.op.input_tensors[0]
    tensor_map["c_ub_cut"] = c_ub_ddr
    sch[c_ub_ddr].compute_inline()
    sch[c_ub].compute_inline()
    sch[c_ub_ddr].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
    return sch, tensor_map, tensor_attr


def fetch_elewise_fusion_ub_info(sch, tensor_map, tensor_attr):
    """
    get ub info with dx + relugrad fusion
    """
    deconv_res = tensor_map.get("deconv_res")
    all_tensor, leaf_tensor = get_all_tensors(deconv_res)
    ub_list = []
    input_tensor_list = []
    c_ub_res = sch.cache_write(deconv_res, tbe_platform_info.scope_ubuf)
    for key, value in all_tensor.items():
        if value.op.input_tensors:
            ub_list.append(value)
        else:
            if leaf_tensor.get(key).op.tag == deconv_res.op.tag:
                input_tensor_list.append([value, c_ub_res])
            else:
                input_tensor_list.append([value, leaf_tensor.get(key)])
    ub_list.append(c_ub_res)
    tensor_attr["elewise_fuse"] = True
    tensor_map["ub_list"] = ub_list
    tensor_map["input_tensor_list"] = input_tensor_list
    c_ub_cut = tensor_map.get("tensor_dx_gm")
    if tensor_attr.get("support_l0c_to_out"):
        c_ub = c_ub_cut
    else:
        c_ub = c_ub_cut.op.input_tensors[0]
    if c_ub.op.name == "bias_add_vector":
        tensor_map["bias_add_vector"] = c_ub
        c_ub = c_ub.op.input_tensors[0]
    tensor_map["c_ub_cut"] = c_ub_cut
    tensor_map["c_ub"] = c_ub
    return sch, tensor_map, tensor_attr


def get_c_add_bias_tensor(tensor_map):
    """
    get c_add_bias tensor
    """
    tensor_dx_gm = tensor_map.get("tensor_dx_gm")
    all_tensor, _ = get_all_tensors(tensor_dx_gm.op.input_tensors[0])
    for tensor in all_tensor.values():
        if tensor.name == "c_add_bias":
            return tensor
    return None
