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
common function for gemm_schedule
"""
from functools import reduce
import math

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.context import op_context
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.base.operation import in_dynamic

BATCH_MATMUL_LEN_ND = 3
BATCH_MATMUL_LEN_NZ = 5
MATMUL_LEN_ND = 2
MATMUL_LEN_NZ = 4
MULTI_FACTOR_BY_DTYPE = 2

DATA_SIZE = {
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "float32": 4,
    "int32": 4
}
FIXPIPE_SCOPE_MAP = {
    "quant_scale_0": "local.FB0",
    "relu_weight_0": "local.FB1",
    "relu_weight_1": "local.FB2",
    "quant_scale_1": "local.FB3"
}

INTRINSIC_FIXPIPE_UNIT_LIST = "Intrinsic_fix_pipe_unit_list"
UNIT_POST_ELTWISE = "post_eltwise"


def _is_support_fixpipe_op():
    if tbe_platform_info.intrinsic_check_support(INTRINSIC_FIXPIPE_UNIT_LIST):
        return tbe_platform_info.intrinsic_check_support(
            INTRINSIC_FIXPIPE_UNIT_LIST, UNIT_POST_ELTWISE)

    return False


# common math funtion
def int_ceil_div(divisor_a, divisor_b):
    """
    round up function
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    if divisor_b == 0:
        args_dict = {
            "errCode": "E60114",
            "reason": "division by zero",
            "value": "divisor_b = {}".format(divisor_b)
        }
        raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
    return (divisor_a + divisor_b - 1) // divisor_b


def int_ceil_align(value, align_factor):
    """
    ceil align
    :param value: int
    :param align_factor: int
    :return: int
    """
    return int_ceil_div(value, align_factor) * align_factor


def _get_precision_mode():
    """
    get calculation mode, high_performance or high_precision
    :return: str, precision_mode
    """
    context = op_context.get_context()
    op_infos = context.get_op_info() if context else {}
    if not op_infos:
        op_infos = {}
    op_type_list =  ["MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2", "FullyConnection"]
    for op_info in op_infos:
        if op_info.op_type in op_type_list:
            return op_info.precision_mode
    return ""


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        if isinstance(i, tvm.expr.Var):
            tmp.append(i)
        else:
            tmp.append(i.value)
    return tmp


# get all tensor for compute map
def get_all_tensors(res):
    """
    get all tensor
    :param res: tensor
    :return: list
    """
    all_tensor = {}
    leaf_tensor = {}
    all_tensor["res"] = res

    def get(tensor):
        """
        find all tensor
        :param tensor: c_gm
        :return: all tensor
        """
        tensor_list = tensor.op.input_tensors
        for one_tensor in tensor_list:
            # check which tensor has not been checked
            if not one_tensor.op.input_tensors:
                if leaf_tensor.get(one_tensor.op.name) is not None:
                    leaf_tensor[one_tensor.op.name].append(tensor)
                else:
                    leaf_tensor[one_tensor.op.name] = [tensor]
            if one_tensor.op.name not in all_tensor:
                all_tensor[one_tensor.op.name] = one_tensor
                get(one_tensor)

    get(res)
    return all_tensor, leaf_tensor


# set scope for matmul tensor
def set_matmul_scope(all_tensor, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: tensor of matmul which after setscope
    :return: dict
    """
    # set scopr for l0 scope and bias
    tensor_map["a_l0a"] = all_tensor.get("tensor_a_matrix")
    tensor_map["b_l0b"] = all_tensor.get("tensor_b_matrix")
    tensor_map["c_l0c"] = all_tensor.get("tensor_c_matrix")
    sch[tensor_map["a_l0a"]].set_scope(tbe_platform_info.scope_ca)
    sch[tensor_map["b_l0b"]].set_scope(tbe_platform_info.scope_cb)
    sch[tensor_map["c_l0c"]].set_scope(tbe_platform_info.scope_cc)
    if len(all_tensor["tensor_c_matrix"].op.input_tensors) == 3:
        input_bias = all_tensor["tensor_c_matrix"].op.input_tensors[2]
        tensor_map["bias_l1"] = sch.cache_read(input_bias,
                                               tbe_platform_info.scope_cbuf,
                                               [all_tensor["tensor_c_matrix"]])
        tensor_map["bias_bt"] = sch.cache_read(tensor_map["bias_l1"],
                                               "local.BT",
                                               [all_tensor["tensor_c_matrix"]])

    al1 = all_tensor.get("tensor_a_matrix").op.input_tensors[0]
    bl1 = all_tensor.get("tensor_b_matrix").op.input_tensors[0]
    if not al1.op.input_tensors:
        tensor_map["a_l1"] = sch.cache_read(al1, tbe_platform_info.scope_cbuf, [all_tensor.get("tensor_a_matrix")])
        tensor_map["a_placehold"] = al1
    elif al1.op.tag == "ND_trans_NZ":
        sch[al1].set_scope(tbe_platform_info.scope_cbuf)
        tensor_map["a_l1"] = al1
        tensor_map["a_placehold"] = al1.op.input_tensors[0]

    if not bl1.op.input_tensors:
        tensor_map["b_l1"] = sch.cache_read(bl1, tbe_platform_info.scope_cbuf, [all_tensor.get("tensor_b_matrix")])
        tensor_map["b_placehold"] = bl1
    elif bl1.op.tag == "ND_trans_NZ":
        sch[bl1].set_scope(tbe_platform_info.scope_cbuf)
        tensor_map["b_l1"] = bl1
        tensor_map["b_placehold"] = bl1.op.input_tensors[0]

    return tensor_map


def set_out_scope(all_tensor, leaf_tensor, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all output tensor of matmul which before setscope
    :param leaf_tensor: all input tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: output tensor of matmul which after setscope
    :return: dict
    """
    res = all_tensor.get("res")
    tensor_map["c_gm"] = res
    if res.op.tag != "gemm":
        if res.op.tag not in ("fixpipe_reform", "dequant_NZ", "requant_NZ", "NZ_trans_ND"):
            tensor_map = set_matmul_ub_scope(res, all_tensor, leaf_tensor, sch, tensor_map)
        else:
            tensor_map = set_matmul_fixpipe_scope(res, sch, tensor_map)

    return tensor_map


def _handle_fixpipe_tensor(sch, fixpipe_tensor, tensor_map, fixpipe_fb_dict, fixpipe_l1_list):
    """
    handle l1 and fb scope in fixpipe tensor
    """
    vector_params = fixpipe_tensor.op.attrs["vector_params"]
    vector_tensors = fixpipe_tensor.op.attrs["vector_tensors"]
    for idx, params_mem in enumerate(vector_params):
        fixpipe_input = vector_tensors[idx]
        fixpipe_input_l1 = sch.cache_read(fixpipe_input, tbe_platform_info.scope_cbuf, [fixpipe_tensor])
        fixpipe_scope_name = FIXPIPE_SCOPE_MAP.get(params_mem.value)
        if fixpipe_scope_name:
            fixpipe_fb_dict[fixpipe_scope_name] = sch.cache_read(
                fixpipe_input_l1, fixpipe_scope_name, [fixpipe_tensor])
            fixpipe_l1_list.append(fixpipe_input_l1)
        else:
            tensor_map["fixpipe_l1_eltwise"] = fixpipe_input_l1

    return tensor_map, fixpipe_fb_dict, fixpipe_l1_list


def set_matmul_fixpipe_scope(res, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all  tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: output fixpipe tensor of matmul which after setscope
    :return: dict
    """
    fixpipe_input_tensor = res.op.input_tensors[0]
    fixpipe_fb_dict = {}
    fixpipe_l1_list = []
    while fixpipe_input_tensor.op.name != "tensor_c_matrix":
        if fixpipe_input_tensor.op.tag in ("dequant_vector", "requant_vector"):
            deq_input = fixpipe_input_tensor.op.input_tensors[1]
            deq_l1 = sch.cache_read(deq_input, tbe_platform_info.scope_cbuf, [fixpipe_input_tensor])
            fixpipe_fb_dict["local.FB0"] = sch.cache_read(deq_l1, "local.FB0", [fixpipe_input_tensor])
            fixpipe_l1_list.append(deq_l1)
        if fixpipe_input_tensor.op.tag == "fixpipe":
            tensor_map, fixpipe_fb_dict, fixpipe_l1_list = _handle_fixpipe_tensor(
                sch, fixpipe_input_tensor, tensor_map, fixpipe_fb_dict, fixpipe_l1_list)
        sch[fixpipe_input_tensor].compute_inline()
        fixpipe_input_tensor = fixpipe_input_tensor.op.input_tensors[0]
    tensor_map["fixpipe_fb"] = fixpipe_fb_dict
    tensor_map["fixpipe_l1"] = fixpipe_l1_list

    return tensor_map


def _handle_ub_input_tensor(all_tensor, leaf_tensor, sch, tensor_map):
    """
    handle the input_tensor in ub
    """
    ub_eltwise_input = []
    for tensor_mem_input, next_tensor_list in leaf_tensor.items():
        eltwise_input_flag = False
        input_tensor = all_tensor[tensor_mem_input]
        for next_tensor in next_tensor_list:
            if "elewise" in next_tensor.op.tag:
                eltwise_input_flag = True
                break
        if eltwise_input_flag:
            ub_eltwise_input.append(sch.cache_read(input_tensor, tbe_platform_info.scope_ubuf, next_tensor_list))
    tensor_map["ub_eltwise_input"] = ub_eltwise_input
    return tensor_map


def set_matmul_ub_scope(res, all_tensor, leaf_tensor, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all output tensor of matmul which before setscope
    :param leaf_tensor: all input tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: output fixpipe tensor of matmul which after setscope
    :return: dict
    """
    ub_eltwise = []
    tensor_map["fixpipe_to_ub"] = all_tensor.get("tensor_c_gm")

    # handle the input ub tensor
    tensor_map = _handle_ub_input_tensor(all_tensor, leaf_tensor, sch, tensor_map)

    for tensor_mem in all_tensor.values():
        if "elewise" in tensor_mem.op.tag:
            if tensor_mem == res:
                # hanle the last eltwise tensor
                ub_write_tensor = sch.cache_write(tensor_mem, tbe_platform_info.scope_ubuf)
            else:
                # the eltwise between fixpipe and last tensor
                sch[tensor_mem].set_scope(tbe_platform_info.scope_ubuf)
                ub_write_tensor = tensor_mem
            ub_eltwise.append(ub_write_tensor)
            if "broadcast" in tensor_mem.op.tag:
                sch[tensor_mem].compute_inline()
        if tensor_mem.op.tag in ("fixpipe_reform", "dequant_NZ", "requant_NZ", "NZ_trans_ND"):
            tensor_map["fixpipe_to_ub"] = tensor_mem
            set_matmul_fixpipe_scope(tensor_mem, sch, tensor_map)

    sch[tensor_map["fixpipe_to_ub"]].set_scope(tbe_platform_info.scope_ubuf)
    tensor_map["ub_eltwise"] = ub_eltwise

    return tensor_map


# hannle tiling
def get_fixpipe_flag(tensor_map):
    """
    code the fixpipe
    :param tensor_map: tensor of matmul which after setscope
    :return: int, the flag of fixpipe
    """
    fixpipe_flag = 1
    for fixpipe_scope in tensor_map.get("fixpipe_fb", {}).keys():
        fixpipe_flag += int(math.pow(2, int(fixpipe_scope[-1])))

    return fixpipe_flag


def get_fused_num(tensor_map):
    """
    get num of ub parts
    :param tensor_map: tensor of matmul which after setscope
    :return: int, the num of ub parts
    """
    fuse_num = 0
    res_data_size = DATA_SIZE.get(tensor_map["c_gm"].dtype, 1)
    ub_data_size = res_data_size
    if tensor_map.get("ub_eltwise"):
        fuse_num += 1
        for ub_eltwise_mem in tensor_map["ub_eltwise"]:
            ub_data_size = max(ub_data_size, DATA_SIZE.get(ub_eltwise_mem, 1))
    if tensor_map.get("ub_eltwise_input"):
        fuse_num += 1
        for ub_eltwise_input_mem in tensor_map["ub_eltwise_input"]:
            ub_data_size = max(ub_data_size, DATA_SIZE.get(ub_eltwise_input_mem, 1))
    fuse_num *= int_ceil_div(ub_data_size, res_data_size)

    return fuse_num


def check_tiling_l1(tiling, tensor_map):
    """
    check tiling illgal or not
    :param tiling: the dict of tiling
    :param tensor_map: the tensor of matmul
    :return: None
    """
    al1_shape = shape_to_list(tensor_map["a_l1"].shape)
    bl1_shape = shape_to_list(tensor_map["b_l1"].shape)
    al1_dtype = tensor_map["a_l1"].dtype
    bl1_dtype = tensor_map["b_l1"].dtype

    if tiling["AL1_shape"] == []:
        al1_size = reduce(lambda x, y: x * y, al1_shape[-4:]) // tiling["block_dim"][2]
    else:
        al1_size = tiling["AL1_shape"][0] * tiling["AL1_shape"][1] * tiling["CL0_matrix"][1] * tbe_platform.BLOCK_IN
        if tiling["manual_pingpong_buffer"].get("AL1_pbuffer") == 2:
            al1_size *= 2
    if tiling["BL1_shape"] == []:
        bl1_size = reduce(lambda x, y: x * y, bl1_shape[-4:]) // tiling["block_dim"][1]
    else:
        bl1_size = tiling["BL1_shape"][0] * tiling["BL1_shape"][1] * tiling["CL0_matrix"][0] * tbe_platform.BLOCK_OUT
        if tiling["manual_pingpong_buffer"].get("BL1_pbuffer") == 2:
            bl1_size *= 2
    l1_size_max = tbe_platform_info.get_soc_spec("L1_SIZE")
    if (al1_size*DATA_SIZE.get(al1_dtype, 1) + bl1_size*DATA_SIZE.get(bl1_dtype, 1)) > l1_size_max:
        args_dict = {
            "errCode": "E60114",
            "reason": "tiling size exceed L1 Buffer",
            "value": "tiling size = {}".format(
                al1_size*DATA_SIZE.get(al1_dtype, 1) + bl1_size*DATA_SIZE.get(bl1_dtype, 1)
            )
        }
        raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))


def check_tiling_l0(tiling, tensor_map):
    """
    check tiling illgal or not
    :param tiling: the dict of tiling
    :param tensor_map: the tensor of matmul
    :return: None
    """
    al0_dtype = tensor_map["a_l0a"].dtype
    bl0_dtype = tensor_map["b_l0b"].dtype
    cl0_dtype = tensor_map["c_l0c"].dtype
    mk_dim = shape_to_list(tensor_map["a_l0a"].shape)[-3] if tiling["AL0_matrix"] == [] else tiling["AL0_matrix"][1]
    nk_dim = shape_to_list(tensor_map["b_l0b"].shape)[-4] if tiling["BL0_matrix"] == [] else tiling["BL0_matrix"][0]

    tiling["AL0_matrix"] = [
        tiling["CL0_matrix"][1],
        mk_dim,
        tbe_platform.CUBE_MKN[al0_dtype]["mac"][0],
        tbe_platform.CUBE_MKN[al0_dtype]["mac"][1]
    ]
    tiling["BL0_matrix"] = [
        nk_dim,
        tiling["CL0_matrix"][0],
        tbe_platform.CUBE_MKN[bl0_dtype]["mac"][2],
        tbe_platform.CUBE_MKN[bl0_dtype]["mac"][1]
    ]

    for buffer_name, data_dtype in zip(["A", "B", "C"], [al0_dtype, bl0_dtype, cl0_dtype]):
        l0_buffer_name = "{}{}".format(buffer_name, "L0_matrix")
        l0_size = reduce(lambda x, y: x * y, tiling[l0_buffer_name][:4]) * DATA_SIZE.get(data_dtype)
        l0_size_max = tbe_platform_info.get_soc_spec("L0" + buffer_name + "_SIZE")
        if tiling["manual_pingpong_buffer"].get(buffer_name + "L0_pbuffer") == 2:
            l0_size *= 2
        if l0_size > l0_size_max:
            args_dict = {
                "errCode": "E60114",
                "reason": "tilling size exceed L0 Buffer of " + buffer_name,
                "value": "tiling size is {} while l0_space is {}".format(l0_size, l0_size_max)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    return tiling


def check_tiling(tiling, tensor_map):
    """
    check tiling illgal or not
    :param tiling: the dict of tiling
    :param tensor_map: the tensor of matmul
    :return: None
    """
    if not tiling:
        args_dict = {"errCode": "E60114",  "reason": "tiling is None", "value": "None"}
        raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
    if tiling.get("AL0_matrix") == [1, 1, 32, 16, 1, 1]:
        args_dict = {"errCode": "E60114",  "reason": "tiling is illegal", "value": "None"}
        raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
    check_tiling_l1(tiling, tensor_map)
    tiling = check_tiling_l0(tiling, tensor_map)

    return tiling


# tiling factor or patrs cut for NZ matmul
def get_aicore_factor(tiling, tensor_map):
    """
    using tilling parameter calculate factor
    :param tiling: the dict of tiling
    :param tensor_map: tensor of matmul
    :return: tilling factor from ub to ddr
             tilling factor from l0c to ub
             tilling factor from ddr to AL1
             tilling factor from ddr to Bl1
    """
    # get l0c
    l0a_dtype = tensor_map["a_l0a"].dtype
    block_reduce = tbe_platform.CUBE_MKN.get(l0a_dtype).get("mac")[1]
    m_dim = shape_to_list(tensor_map["a_l0a"].shape)[-4]
    n_dim = shape_to_list(tensor_map["b_l0b"].shape)[-3]

    l0c_tiling_factor = tiling["CL0_matrix"][0:2]
    # parts from L0C to UB
    l0c_ub_parts = [
        int_ceil_div(l0c_tiling_factor[0], tiling["CUB_matrix"][0]),
        int_ceil_div(l0c_tiling_factor[1], tiling["CUB_matrix"][1])
    ]

    # parts for GM to L0C
    l0c_parts = [
        int_ceil_div(n_dim // tiling["block_dim"][1], l0c_tiling_factor[0]),
        int_ceil_div(m_dim // tiling["block_dim"][2], l0c_tiling_factor[1])
    ]

    # out unit is 16*32 for int8
    if tensor_map["c_gm"].dtype == "int8":
        l0c_tiling_factor[0] = max(1, l0c_tiling_factor[0] // MULTI_FACTOR_BY_DTYPE)
    # out unit is 16*8 for int32 and fp32
    if tensor_map["c_gm"].dtype ==  "float32" and tensor_map["a_l1"].dtype == "float32":
        l0c_tiling_factor[0] *= MULTI_FACTOR_BY_DTYPE

    # patrs for GM to AL1, AL1_shape = [(batch), n/16, k/16, 16, 16]
    if tiling["AL1_shape"]:
        al1_parts = [
            tiling["AL1_shape"][0] // block_reduce // tiling["AL0_matrix"][1],
            int_ceil_div(l0c_parts[1], tiling["AL1_shape"][1])
        ]
    else:
        al1_parts = [None, 1]

    if tiling["BL1_shape"]:
        bl1_parts = [
            tiling["BL1_shape"][0] // block_reduce // tiling["AL0_matrix"][1],
            int_ceil_div(l0c_parts[0], tiling["BL1_shape"][1])
        ]
    else:
        bl1_parts = [None, 1]

    return l0c_tiling_factor, l0c_ub_parts, al1_parts, bl1_parts


def split_mn_l0c_l1(c_gm, sch, l0c_factor, al1_parts, bl1_parts):
    """
    get l0c and l1 axis
    :param c_gm: final tensor
    :param sch: schedule
    :param l0c_factor: tilling factor for l0c
    :param al1_parts: tilling parts for al1
    :param bl1_parts: tilling parts for bl1
    :return: axis list after split
    """
    # split c_gm according to factor of loc and out_shape
    is_nd_flag = len(c_gm.shape) in (MATMUL_LEN_ND, BATCH_MATMUL_LEN_ND)
    if is_nd_flag:
        l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm.op.axis[-1], l0c_factor[0])
        l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[-2], l0c_factor[1])
        sch[c_gm].reorder(l0c_n_outer, l0c_m_outer, l0c_m_inner, l0c_n_inner)
    else:
        l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm.op.axis[-4], l0c_factor[0])
        l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[-3], l0c_factor[1])
        sch[c_gm].reorder(l0c_n_outer, l0c_m_outer, l0c_n_inner, l0c_m_inner)

    # split c_gm according to factor of a_l1 and b_l1
    l1_m_outer_outer, l1_m_outer_inner = sch[c_gm].split(l0c_m_outer, nparts=al1_parts[1])
    l1_n_outer_outer, l1_n_outer_inner = sch[c_gm].split(l0c_n_outer, nparts=bl1_parts[1])

    return [l1_m_outer_outer, l1_m_outer_inner, l0c_m_inner], [l1_n_outer_outer, l1_n_outer_inner, l0c_n_inner]


def split_mn_block(c_gm, sch, tiling, l1_m_axis, l1_n_axis):
    """
    get block axis and then bind
    :param c_gm: final tensor
    :param sch: schedule
    :param tiling: tilling of sch
    :param l1_m_axis: the m axis of al1
    :param l1_n_axis: the n axis of bl1
    :return: axis list after split upon block
    """
    if len(c_gm.shape) in (BATCH_MATMUL_LEN_NZ, BATCH_MATMUL_LEN_ND):
        batch_axis = c_gm.op.axis[0]
    else:
        batch_axis, l1_n_axis[0] = sch[c_gm].split(l1_n_axis[0], nparts=1)

    batch_out, batch_inner = sch[c_gm].split(batch_axis, nparts=tiling["block_dim"][0])
    block_n_out, l1_n_axis[1] = sch[c_gm].split(l1_n_axis[1], nparts=tiling["block_dim"][1])
    block_m_out, l1_m_axis[1] = sch[c_gm].split(l1_m_axis[1], nparts=tiling["block_dim"][2])

    sch[c_gm].reorder(batch_out, block_n_out, block_m_out, batch_inner,
                      l1_n_axis[0], l1_m_axis[0],
                      l1_n_axis[1], l1_m_axis[1])
    blocks = reduce(lambda x, y: x * y, tiling["block_dim"][0:3])
    block_fused_axis = sch[c_gm].fuse(batch_out, block_n_out, block_m_out)
    block_bind_axis, _ = sch[c_gm].split(block_fused_axis, nparts=blocks)
    blockidx = tvm.thread_axis("blockIdx.x")
    sch[c_gm].bind(block_bind_axis, blockidx)
    return batch_inner


def split_ub(c_gm, sch, l1_m_axis, l1_n_axis, ub_split):
    """
    get ub axis
    :param c_gm: final tensor
    :param sch: schedule
    :param l1_mn_axis: the m and n axis of l1
    :param ub_split: l0c to ub parts(NZ) or factor(ND)
    :param handle_ub: split ub or not
    """
    if ub_split is not None:
        if len(c_gm.shape) in (MATMUL_LEN_ND, BATCH_MATMUL_LEN_ND):
            l1_n_axis[2], l1_n_axis_ub_inner = sch[c_gm].split(l1_n_axis[2], ub_split[0])
            l1_m_axis[2], l1_m_axis_ub_inner = sch[c_gm].split(l1_m_axis[2], ub_split[1])
            sch[c_gm].reorder(l1_n_axis[2], l1_m_axis[2], l1_m_axis_ub_inner, l1_n_axis_ub_inner)
            c_gm_emit_axis = [l1_m_axis_ub_inner, l1_n_axis_ub_inner]
        else:
            l1_n_axis[2], l1_n_axis_ub_inner = sch[c_gm].split(l1_n_axis[2], nparts=ub_split[0])
            l1_m_axis[2], l1_m_axis_ub_inner = sch[c_gm].split(l1_m_axis[2], nparts=ub_split[1])
            sch[c_gm].reorder(l1_n_axis[2], l1_m_axis[2], l1_n_axis_ub_inner, l1_m_axis_ub_inner)
            c_gm_emit_axis = [l1_n_axis_ub_inner, l1_m_axis_ub_inner]
        fixpipe_attach_axis = l1_m_axis[2]
    else:
        if len(c_gm.shape) in (MATMUL_LEN_ND, BATCH_MATMUL_LEN_ND):
            c_gm_emit_axis = [l1_m_axis[2], l1_n_axis[2]]
        else:
            c_gm_emit_axis = [l1_n_axis[2], l1_m_axis[2]]
        fixpipe_attach_axis = l1_m_axis[1]

    return c_gm_emit_axis + [fixpipe_attach_axis]


def split_k(c_l0c, sch, l0c_k_factor, l1a_k_part, l1b_k_part):
    """
    split k dim
    :param c_l0c: the l0c tensor
    :param sch: schedule
    :param l0c_k_factor: the k factor in mmad cal
    :param l1a_k_part: the k parts from L1A to L0c
    :param l1b_k_part: the k parts from L1B to L0c
    :return: [al1_k, bl1_k, l0k]
    """
    l0c_axis = sch[c_l0c].op.axis
    k_outer_outer, k_outer_inner = sch[c_l0c].split(sch[c_l0c].op.reduce_axis[0], l0c_k_factor)
    sch[c_l0c].reorder(k_outer_outer, *l0c_axis, k_outer_inner, sch[c_l0c].op.reduce_axis[1])

    if l1a_k_part is not None and l1b_k_part is not None:
        l1_parts_inner = min(l1a_k_part, l1b_k_part)
        l1_parts_outer = max(l1a_k_part, l1b_k_part) // l1_parts_inner
        k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(k_outer_outer, l1_parts_inner)
        k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(k_outer_outer_outer, l1_parts_outer)
    elif l1a_k_part is None and l1b_k_part is None:
        k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(k_outer_outer, nparts=1)
        k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(k_outer_outer_outer, nparts=1)
    else:
        l1_parts_inner = l1a_k_part if l1a_k_part is not None else l1b_k_part
        k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(k_outer_outer, l1_parts_inner)
        k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(k_outer_outer_outer, nparts=1)

    if l1a_k_part is None or (l1b_k_part is not None and l1a_k_part > l1b_k_part):
        return [k_outer_outer_outer_outer, k_outer_outer_outer_inner, k_outer_outer_inner]
    return [k_outer_outer_outer_inner, k_outer_outer_outer_outer, k_outer_outer_inner]


def reorder_l1_mn_axis(tiling, al1_m_parts, bl1_n_parts):
    """
    reorder axis of l1
    :param tiling: the dict of tiling
    :param al1_parts: tilling parts for al1
    :param bl1_parts: tilling parts for bl1
    :return: None
    """
    if in_dynamic():
        if tiling["AL1_shape"] == []:
            return True
        if tiling["BL1_shape"] != [] and tiling["AL1_shape"][1] > tiling["BL1_shape"][1]:
            return True
        return False

    if al1_m_parts != 1 and bl1_n_parts != 1:
        l0a_size = reduce(lambda x, y: x * y, tiling["AL0_matrix"])
        l0b_size = reduce(lambda x, y: x * y, tiling["BL0_matrix"])
        l1_size_no_reorder = l0a_size * tiling["AL1_shape"][1] * al1_m_parts * bl1_n_parts \
                             + l0b_size * tiling["BL1_shape"][1] * bl1_n_parts
        l1_size_reorder = l0a_size * tiling["AL1_shape"][1] * al1_m_parts + \
                          l0b_size * tiling["BL1_shape"][1] * bl1_n_parts * al1_m_parts
        if l1_size_no_reorder > l1_size_reorder:
            return True
    if al1_m_parts == 1 and bl1_n_parts != 1:
        return True
    return False


# compute at of matmul
def attach_of_bias_table(sch, tensor_map, bl1_parts, c_slice_axis, fully_load_axis):
    """
    attach tensor of bias
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param bl1_parts: tilling parts for bl1
    :param c_slice_axis: l0c load axis tensor for tesor
    :param fully_load_axis: fully load axis for tensor
    :return: None
    """
    if tensor_map.get("bias_l1") is not None:
        bias_l1 = tensor_map["bias_l1"]
        bias_bt = tensor_map["bias_bt"]
        sch[bias_bt].compute_at(sch[tensor_map["c_gm"]], c_slice_axis)
        if bl1_parts[1] == 1:
            sch[bias_l1].compute_at(sch[tensor_map["c_gm"]], fully_load_axis)
        else:
            sch[bias_l1].compute_at(sch[tensor_map["c_gm"]], c_slice_axis)


def attach_of_fixpipe(sch, tensor_map, bl1_parts, fixpipe_axis, fully_load_axis):
    """
    attach tensor of fixpipe
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param bl1_parts: tilling parts for bl1
    :param fixpipe_axis: out load axis tensor for fixpipe tenspr
    :param fully_load_axis: fully load axis for tensor
    :return: None
    """
    for fixpipe_l1_mem in tensor_map.get("fixpipe_l1", []):
        if bl1_parts[1] == 1:
            sch[fixpipe_l1_mem].compute_at(sch[tensor_map["c_gm"]], fully_load_axis)
        else:
            sch[fixpipe_l1_mem].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis)

    if tensor_map.get("fixpipe_l1_eltwise") is not None:
        sch[tensor_map["fixpipe_l1_eltwise"]].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis)

    for fixpipe_fb_mem in tensor_map.get("fixpipe_fb", {}).values():
        sch[fixpipe_fb_mem].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis)


def attach_of_ub(sch, tensor_map, ub_axis):
    """
    attach tensor of fixpipe
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param ub_axis: out load axis tensor for ub tenspr
    :return: None
    """
    for ub_eltwise_mem in tensor_map.get("ub_eltwise", []):
        sch[ub_eltwise_mem].compute_at(sch[tensor_map["c_gm"]], ub_axis)
    for ub_eltwise_input_mem in tensor_map.get("ub_eltwise_input", []):
        sch[ub_eltwise_input_mem].compute_at(sch[tensor_map["c_gm"]], ub_axis)
    if tensor_map.get("fixpipe_to_ub") is not None:
        fixpipe_to_ub = tensor_map["fixpipe_to_ub"]
        sch[fixpipe_to_ub].compute_at(sch[tensor_map["c_gm"]], ub_axis)


def attach_of_l1_l0(sch, tensor_map, l1_attch_axis, al1_parts, bl1_parts):
    """
    attach tensor of l1a,l1b, l0a, l0b
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param l1_attch_axis: l1a_k_axis, l1b_k_axis, l0_k_aixs, l1a_m_axis, l1b_n_axis
    :param al1_parts: tilling parts for al1
    :param bl1_parts: tilling parts for bl1
    :return: None
    """
    a_l1, b_l1 = tensor_map.get("a_l1"), tensor_map.get("b_l1")
    a_l0, b_l0, c_l0c = tensor_map.get("a_l0a"), tensor_map.get("b_l0b"), tensor_map.get("c_l0c")
    c_gm = tensor_map.get("c_gm")
    sch[a_l0].compute_at(sch[c_l0c], l1_attch_axis[2])
    sch[b_l0].compute_at(sch[c_l0c], l1_attch_axis[2])
    if al1_parts[0] is None:
        sch[a_l1].compute_at(sch[c_gm], l1_attch_axis[3])
    else:
        sch[a_l1].compute_at(sch[c_l0c], l1_attch_axis[0])
    if bl1_parts[0] is None:
        sch[b_l1].compute_at(sch[c_gm], l1_attch_axis[4])
    else:
        sch[b_l1].compute_at(sch[c_l0c], l1_attch_axis[1])


# double buffer for all tensor
def double_buffer_func(sch, tensor_map, tiling):
    """
    double buffer for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param tiling: the dict of tiling
    :return: None
    """
    double_buffer_flag = tiling["manual_pingpong_buffer"]
    if double_buffer_flag["AL1_pbuffer"] == 2:
        sch[tensor_map["a_l1"]].double_buffer()
    if double_buffer_flag["BL1_pbuffer"] == 2:
        sch[tensor_map["b_l1"]].double_buffer()
    if double_buffer_flag["AL0_pbuffer"] == 2:
        sch[tensor_map["a_l0a"]].double_buffer()
    if double_buffer_flag["BL0_pbuffer"] == 2:
        sch[tensor_map["b_l0b"]].double_buffer()
    if double_buffer_flag["CL0_pbuffer"] == 2:
        sch[tensor_map["c_l0c"]].double_buffer()
        double_buffer_fp_and_bt(sch, tensor_map)
    if double_buffer_flag["CUB_pbuffer"] == 2:
        double_buffer_ub(sch, tensor_map)


def double_buffer_fp_and_bt(sch, tensor_map):
    """
    double buffer for bias table and fixpipe
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :return: None
    """
    if tensor_map.get("bias_l1") is not None:
        bias_l1 = tensor_map["bias_l1"]
        bias_bt = tensor_map["bias_bt"]
        sch[bias_l1].double_buffer()
        sch[bias_bt].double_buffer()
    for fixpipe_l1_mem in tensor_map.get("fixpipe_l1", []):
        sch[fixpipe_l1_mem].double_buffer()
    if tensor_map.get("fixpipe_l1_eltwise") is not None:
        sch[tensor_map["fixpipe_l1_eltwise"]].double_buffer()
    for fixpipe_fb_mem in tensor_map.get("fixpipe_fb", {}).values():
        sch[fixpipe_fb_mem].double_buffer()


def double_buffer_ub(sch, tensor_map):
    """
    double buffer for ub tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :return: None
    """
    for ub_eltwise_mem in tensor_map.get("ub_eltwise", []):
        sch[ub_eltwise_mem].double_buffer()
    for ub_eltwise_input_mem in tensor_map.get("ub_eltwise_input", []):
        sch[ub_eltwise_input_mem].double_buffer()
    if tensor_map.get("fixpipe_to_ub") is not None:
        fixpipe_to_ub = tensor_map["fixpipe_to_ub"]
        sch[fixpipe_to_ub].double_buffer()


# emit func of matmul
def emit_insn_func(sch, tensor_map, k_axis, c_gm_emit_axis):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param tiling: the dict of tiling
    :param k_axis: the outer axis of mmad
    :param c_gm_emit_axis: emit axis of c_gm
    :return: None
    """
    c_gm = tensor_map["c_gm"]
    is_nd_flag = len(c_gm.shape) in (MATMUL_LEN_ND, BATCH_MATMUL_LEN_ND)
    emit_insn_l1_and_l0(sch, tensor_map, k_axis)
    emit_c_gm(sch, tensor_map, c_gm_emit_axis, is_nd_flag)
    emit_insn_fp_and_bt(sch, tensor_map)
    emit_insn_ub(sch, tensor_map, is_nd_flag)


def emit_c_gm(sch, tensor_map, c_gm_emit_axis, is_nd):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param c_gm_emit_axis: emit axis of c_gm
    :param is_nd: emit nd or nz
    :return: None
    """
    c_gm = tensor_map["c_gm"]
    c_l0c = tensor_map["c_l0c"]
    emit_str = "fixpipe_op" if _is_support_fixpipe_op() else "dma_copy"
    if is_nd and tensor_map.get("fixpipe_to_ub") is None:
        sch[c_gm].split(c_gm_emit_axis[1], 16)
        dma_dict = {"layout_transform": "nz2nd"}
        sch[c_gm].emit_insn(c_gm_emit_axis[0], emit_str, dma_dict)
    else:
        if c_gm.dtype == "int8":
            sch[c_gm].split(c_gm.op.axis[-1], 16)
            n_shape = int_ceil_div(c_l0c.shape[-4].value, 2) * 2
            sch[c_l0c].storage_align(c_l0c.op.axis[-4], n_shape, 0)
        if tensor_map["a_l1"].dtype == "float32" and c_gm.dtype == "float32":
            _, chanel_split_in = sch[c_gm].split(c_gm_emit_axis[0], factor=2)
            sch[c_gm].emit_insn(chanel_split_in, emit_str,  {"layout_transform": "channel_split"})
        else:
            sch[c_gm].emit_insn(c_gm_emit_axis[0], emit_str)


def emit_insn_l1_and_l0(sch, tensor_map, k_axis):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param k_axis: the outer axis of mmad
    :return: None
    """
    a_l1 = tensor_map["a_l1"]
    b_l1 = tensor_map["b_l1"]
    dma_dict = {"layout_transform": "nd2nz"}
    if a_l1.op.tag == "ND_trans_NZ":
        sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy", dma_dict)
    else:
        sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy")
    if b_l1.op.tag == "ND_trans_NZ":
        sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy", dma_dict)
    else:
        sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy")

    a_l0a = tensor_map["a_l0a"]
    if a_l0a.dtype == "int8" and a_l0a.op.attrs["transpose_a"] == "false":
        _, a_l0a_inner = sch[a_l0a].split(a_l0a.op.axis[-4], 2) # split m1 axis
        sch[a_l0a].emit_insn(a_l0a_inner, "dma_copy")
    elif a_l0a.dtype == "float32" and a_l0a.op.attrs["transpose_a"] == "false":
        sch[a_l0a].split(a_l0a.op.axis[-2], factor=8)
        sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy", {'img2col': 1})
    else:
        sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy")
    b_l0b = tensor_map["b_l0b"]
    if b_l0b.dtype == "int8" and b_l0b.op.attrs["transpose_b"] == "true" :
        _, b_l0b_inner = sch[b_l0b].split(b_l0b.op.axis[-3], 2) # split n1 axis
        sch[b_l0b].emit_insn(b_l0b_inner, "dma_copy")
    elif b_l0b.dtype == "float32" and b_l0b.op.attrs["transpose_b"] == "true":
        sch[b_l0b].split(b_l0b.op.axis[-2], factor=8)
        sch[b_l0b].emit_insn(b_l0b.op.axis[0], "dma_copy", {'img2col': 1})
    else:
        sch[b_l0b].emit_insn(b_l0b.op.axis[0], "dma_copy")
    c_l0c = tensor_map["c_l0c"]
    mad_dict = {
        "mad_pattern": tbe_platform.GEMM_MODE,
        "k_outer": k_axis
    }
    if _get_precision_mode() == "high_performance":
        mad_dict["hf32"] = 1
    sch[c_l0c].emit_insn(c_l0c.op.axis[-4], "mad", mad_dict)


def emit_insn_fp_and_bt(sch, tensor_map):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :return: None
    """
    if tensor_map.get("bias_l1") is not None:
        bias_l1 = tensor_map["bias_l1"]
        bias_bt = tensor_map["bias_bt"]
        sch[bias_l1].emit_insn(bias_l1.op.axis[0], "dma_copy")
        sch[bias_bt].emit_insn(bias_bt.op.axis[0], "dma_copy")
    for fixpipe_l1_mem in tensor_map.get("fixpipe_l1", []):
        sch[fixpipe_l1_mem].emit_insn(fixpipe_l1_mem.op.axis[0], "dma_copy")
    if tensor_map.get("fixpipe_l1_eltwise") is not None:
        fixpipe_l1_eltwise = tensor_map["fixpipe_l1_eltwise"]
        sch[fixpipe_l1_eltwise].emit_insn(fixpipe_l1_eltwise.op.axis[0], "dma_copy")
    for fixpipe_fb_mem in tensor_map.get("fixpipe_fb", {}).values():
        sch[fixpipe_fb_mem].emit_insn(fixpipe_fb_mem.op.axis[0], "dma_copy")


def emit_insn_ub(sch, tensor_map, is_nd):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param is_nd:  :param is_nd: nz or nd format
    :return: None
    """
    for ub_eltwise_mem in tensor_map.get("ub_eltwise", []):
        align_factor = tbe_platform.CUBE_MKN[ub_eltwise_mem.dtype]["mac"][1]
        sch[ub_eltwise_mem].compute_align(ub_eltwise_mem.op.axis[-1], align_factor)
        sch[ub_eltwise_mem].storage_align(ub_eltwise_mem.op.axis[-2], align_factor, 0)
        sch[ub_eltwise_mem].emit_insn(ub_eltwise_mem.op.axis[0], "vector_auto")
    for ub_eltwise_input_mem in tensor_map.get("ub_eltwise_input", []):
        align_factor = tbe_platform.CUBE_MKN[ub_eltwise_input_mem.dtype]["mac"][1]
        if len(ub_eltwise_input_mem.op.axis) > 1:
            sch[ub_eltwise_input_mem].compute_align(ub_eltwise_input_mem.op.axis[-1], align_factor)
            sch[ub_eltwise_input_mem].storage_align(ub_eltwise_input_mem.op.axis[-2], align_factor, 0)
        sch[ub_eltwise_input_mem].emit_insn(ub_eltwise_input_mem.op.axis[0], "dma_copy")
    if tensor_map.get("fixpipe_to_ub") is not None:
        fixpipe_to_ub = tensor_map["fixpipe_to_ub"]
        if is_nd:
            align_factor = tbe_platform.CUBE_MKN[fixpipe_to_ub.dtype]["mac"][1]
            sch[fixpipe_to_ub].compute_align(fixpipe_to_ub.op.axis[-1], align_factor)
            sch[fixpipe_to_ub].storage_align(fixpipe_to_ub.op.axis[-2], align_factor, 0)
            sch[fixpipe_to_ub].split(fixpipe_to_ub.op.axis[-1], 16)
            dma_dict = {"layout_transform": "nz2nd"}
            sch[fixpipe_to_ub].emit_insn(fixpipe_to_ub.op.axis[0], "dma_copy", dma_dict)
        else:
            sch[fixpipe_to_ub].emit_insn(fixpipe_to_ub.op.axis[0], "dma_copy")


def do_buffer_align(sch, tensor_map, trans_a, trans_b):
    """
    do buffer align,
    m1 and n1 should be aligned to even number to do load2d_transpose when dtype is int8
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param trans_a: bool
    :param trans_b: bool
    """
    a_l0a, b_l0b, c_l0c = tensor_map.get("a_l0a"), tensor_map.get("b_l0b"), tensor_map.get("c_l0c")
    m_align = (1, 2) if not trans_a and a_l0a.dtype == "int8" else (1, 1)
    n_align = (1, 2) if trans_b and b_l0b.dtype == "int8" else (1, 1)
    sch[c_l0c].buffer_align(
        *([(1, 1)] * (len(c_l0c.shape) - MATMUL_LEN_NZ)),
        n_align,
        m_align,
        (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]),
        (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]),
        (1, 1),
        (1, tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][1])
    )
