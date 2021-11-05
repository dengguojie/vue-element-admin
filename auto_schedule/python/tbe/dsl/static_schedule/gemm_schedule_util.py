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
from functools import reduce # pylint: disable=C0302

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils.errormgr import error_manager_util

BATCH_MATMUL_LEN_ND = 3
BATCH_MATMUL_LEN_NZ = 5
DATA_SIZE = {
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "float32": 4,
    "int32": 4
}

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
    all_tensor = dict()
    leaf_tensor = dict()
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
                leaf_tensor[one_tensor.op.name] = tensor
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


def set_out_scope(all_tensor, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all output tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: output tensor of matmul which after setscope
    :return: dict
    """
    res = all_tensor.get("res")
    tensor_map["c_gm"] = res
    if res.op.tag != "gemm":
        matmul_res = all_tensor.get("tensor_c_gm")
        sch[matmul_res].compute_inline()
        tensor_map["matmul_c_gm"] = matmul_res
        fixpipe_new = False
        tensor_map = set_matmul_fixpipe_scope(res, all_tensor, sch, tensor_map)
    else:
        tensor_map["matmul_c_gm"] = res

    return tensor_map


def set_matmul_fixpipe_scope(res, all_tensor, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all output fixpipe tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: output fixpipe tensor of matmul which after setscope
    :return: dict
    """
    fixpipe_input_tensor = res.op.input_tensors[0]
    while (fixpipe_input_tensor.op.tag != "gemm"):
        if fixpipe_input_tensor.op.tag in ("dequant_vector", "dequant_scale", "requant_vector", "requant_scale"):
            deq_input = fixpipe_input_tensor.op.input_tensors[1]
            deq_ori_shape = list(i.value for i in deq_input.op.attrs["ori_shape"])
            deq_dims = reduce(lambda x, y: x * y, deq_ori_shape[:])
            if deq_dims > 1:
                tensor_map["deq_l1"] = sch.cache_read(deq_input, tbe_platform_info.scope_cbuf, [fixpipe_input_tensor])
                tensor_map["deq_fb"] = sch.cache_read(tensor_map["deq_l1"], "local.FB0", [fixpipe_input_tensor])
        sch[fixpipe_input_tensor].compute_inline()
        fixpipe_input_tensor = fixpipe_input_tensor.op.input_tensors[0]

    return tensor_map


# hannle tiling
def get_fixpipe_flag(tensor_map):
    """
    code the fixpipe
    :param tensor_map: tensor of matmul which after setscope
    :return: int, the flag of fixpipe
    """
    fixpipe_flag = 1
    if tensor_map.get("deq_fb") is not None:
        fixpipe_flag += 1

    return fixpipe_flag


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
    if (al1_size*DATA_SIZE[al1_dtype] + bl1_size*DATA_SIZE[bl1_dtype]) > l1_size_max:
        args_dict = {
            "errCode": "E60114",
            "reason": "tiling size exceed L1 Buffer",
            "value": "tiling size = {}".format(
                al1_size*DATA_SIZE[al1_dtype] + bl1_size*DATA_SIZE[bl1_dtype]
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
        l0_size = reduce(lambda x, y: x * y, tiling[buffer_name + "L0_matrix"][:4]) * DATA_SIZE[data_dtype]
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


def check_tiling_ub(tiling, fuse_num, tensor_map):
    """
    check tiling illgal or not
    :param tiling: the dict of tiling
    :param fuse_num: the numbers of ub parts
    :param tensor_map: the tensor of matmul
    :return: None
    """
    pass


def check_tiling(tiling, fuse_num, tensor_map, check_ub=False):
    """
    check tiling illgal or not
    :param tiling: the dict of tiling
    :param fuse_num: the numbers of ub parts
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
    if check_ub:
        check_tiling_ub(tiling, fuse_num, tensor_map)
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
    m_dim, k_dim = shape_to_list(tensor_map["a_l0a"].shape)[-4:-2]
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
        l0c_tiling_factor[0] //= 2
    # out unit is 16*8 for int32 and fp32
    if tensor_map["c_gm"].dtype ==  "float32" and tensor_map["a_l1"].dtype == "float32":
        l0c_tiling_factor[0] *= 2

    # patrs for GM to AL1, AL1_shape = [(batch), n/16, k/16, 16, 16]
    if tiling["AL1_shape"]:
        al1_parts = [
            int_ceil_div(k_dim, tiling["AL1_shape"][0] // block_reduce),
            int_ceil_div(l0c_parts[1], tiling["AL1_shape"][1])
        ]
    else:
        al1_parts = [1, 1]

    if tiling["BL1_shape"]:
        bl1_parts = [
            int_ceil_div(k_dim, tiling["BL1_shape"][0] // block_reduce),
            int_ceil_div(l0c_parts[0], tiling["BL1_shape"][1])
        ]
    else:
        bl1_parts = [1, 1]

    return l0c_tiling_factor, l0c_ub_parts, al1_parts, bl1_parts


def split_mn_l0c_l1(c_gm, sch, l0c_factor, al1_parts, bl1_parts, is_nd=False):
    """
    get l0c and l1 axis
    :param c_gm: final tensor
    :param sch: schedule
    :param l0c_factor: tilling factor for l0c
    :param al1_parts: tilling parts for al1
    :param bl1_parts: tilling parts for bl1
    :param is_nd: nz or nd format
    :return: axis list after split
    """
    # split c_gm according to factor of loc and out_shape
    if is_nd:
        n_axis = -1
        m_axis = -2
    else:
        n_axis = -4
        m_axis = -3
    l0c_n_outer, l0c_n_inner = sch[c_gm].split(c_gm.op.axis[n_axis], l0c_factor[0])
    l0c_m_outer, l0c_m_inner = sch[c_gm].split(c_gm.op.axis[m_axis], l0c_factor[1])
    if is_nd:
        sch[c_gm].reorder(l0c_n_outer, l0c_m_outer, l0c_m_inner, l0c_n_inner)
    else:
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


def split_ub(c_gm, sch, l1_m_axis, l1_n_axis, ub_split, is_nd=False, handle_ub=False):
    """
    get ub axis
    :param c_gm: final tensor
    :param sch: schedule
    :param l1_m_axis: the m axis of al1
    :param l1_n_axis: the n axis of bl1
    :param ub_split: l0c to ub parts(NZ) or factor(ND)
    :param is_nd: nz or nd format
    :param handle_ub: split ub or not
    :return: axis list after split ub
    """
    if handle_ub:
        if is_nd:
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
        if is_nd:
            c_gm_emit_axis = [l1_m_axis[2], l1_n_axis[2]]
            fixpipe_attach_axis = l1_m_axis[1]
        else:
            c_gm_emit_axis = [l1_n_axis[2], l1_m_axis[2]]
            fixpipe_attach_axis = l1_m_axis[1]
    return c_gm_emit_axis, fixpipe_attach_axis


def split_k(c_l0c, sch, l0c_k_factor, l1a_k_part, l1b_k_part):
    """
    split k dim
    :param c_l0c: the l0c tensor
    :param sch: schedule
    :param l0c_k_factor: the k factor in mmad cal
    :param l1a_k_part: the k parts from L1A to L0c
    :param l1b_k_part: the k parts from L1B to L0c
    :return: None
    """
    k_out, k_inner = sch[c_l0c].op.reduce_axis
    l0c_axis = sch[c_l0c].op.axis
    k_outer_outer, k_outer_inner = sch[c_l0c].split(k_out, l0c_k_factor)
    sch[c_l0c].reorder(k_outer_outer, *l0c_axis, k_outer_inner, k_inner)

    l1_parts_max = max(l1a_k_part, l1b_k_part)
    l1_parts_min = min(l1a_k_part, l1b_k_part)
    k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(k_outer_outer, nparts=l1_parts_max)
    k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(k_outer_outer_outer, nparts=l1_parts_min)
    if l1a_k_part > l1b_k_part:
        return [k_outer_outer_outer_inner, k_outer_outer_outer_outer, k_outer_outer_inner]
    return [k_outer_outer_outer_outer, k_outer_outer_outer_inner, k_outer_outer_inner]


def reorder_l1_mn_axis(sch, tiling, al1_m_parts, bl1_n_parts):
    """
    reorder axis of l1
    :param sch: schedule
    :param tiling: the dict of tiling
    :param al1_parts: tilling parts for al1
    :param bl1_parts: tilling parts for bl1
    :return: None
    """
    reorder_flag = False
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
    if tensor_map.get("deq_l1") is not None:
        deq_l1 = tensor_map["deq_l1"]
        deq_fb = tensor_map["deq_fb"]
        sch[deq_fb].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis) 
        if bl1_parts[1] == 1:
            sch[deq_l1].compute_at(sch[tensor_map["c_gm"]], fully_load_axis)
        else:
            sch[deq_l1].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis)


def attach_of_l1(sch, tensor_map, l1_attch_axis, al1_parts, bl1_parts):
    """
    attach tensor of l1a and l1b
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param l1_attch_axis: l1a_k_axis, l1b_k_axis, l1a_m_axis, l1b_n_axis
    :param al1_parts: tilling parts for al1
    :param bl1_parts: tilling parts for bl1
    :return: None
    """
    a_l1 = tensor_map["a_l1"]
    b_l1 = tensor_map["b_l1"]
    if al1_parts[0] == 1:
        sch[a_l1].compute_at(sch[tensor_map["c_gm"]], l1_attch_axis[2])
    else:
        sch[a_l1].compute_at(sch[tensor_map["c_l0c"]], l1_attch_axis[0])
    if bl1_parts[0] == 1:
        sch[b_l1].compute_at(sch[tensor_map["c_gm"]], l1_attch_axis[3])
    else:
        sch[b_l1].compute_at(sch[tensor_map["c_l0c"]], l1_attch_axis[1])


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
    if tensor_map.get("deq_l1") is not None:
        deq_l1 = tensor_map["deq_l1"]
        deq_fb = tensor_map["deq_fb"]
        sch[deq_l1].double_buffer()
        sch[deq_fb].double_buffer()


def double_buffer_ub(sch, tensor_map):
    """
    double buffer for ub tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :return: None
    """
    pass


# emit func of matmul
def emit_insn_func(sch, tensor_map, tiling, k_axis, c_gm_emit_axis, is_nd=False):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param tiling: the dict of tiling
    :param k_axis: the outer axis of mmad
    :param c_gm_emit_axis: emit axis of c_gm
    :param is_nd:  :param is_nd: nz or nd format
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
        a_l0a_outer, a_l0a_inner = sch[a_l0a].split(a_l0a.op.axis[0], 2)
        sch[a_l0a].emit_insn(a_l0a_inner, "dma_copy")
    elif a_l0a.dtype == "float32" and a_l0a.op.attrs["transpose_a"] == "false":
        sch[a_l0a].split(a_l0a.op.axis[-2], factor=8)
        sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy", {'img2col': 1})
    else:
        sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy")
    b_l0b = tensor_map["b_l0b"]
    if b_l0b.dtype == "int8" and b_l0b.op.attrs["transpose_b"] == "true" :
        b_l0b_outer, b_l0b_inner = sch[b_l0b].split(b_l0b.op.axis[1], 2)
        sch[b_l0b].emit_insn(b_l0b_inner, "dma_copy")
    elif b_l0b.dtype == "float32" and b_l0b.op.attrs["transpose_b"] == "true":
         sch[b_l0b].split(b_l0b.op.axis[-2], factor=8)
         sch[b_l0b].emit_insn(b_l0b.op.axis[0], "dma_copy", {'img2col': 1})
    else:
        sch[b_l0b].emit_insn(b_l0b.op.axis[0], "dma_copy")

    # when output dtype is int8, split c_gm
    c_gm = tensor_map["c_gm"]
    c_l0c = tensor_map["c_l0c"]
    if is_nd:
        sch[c_gm].split(c_gm_emit_axis[1], 16)
        dma_dict = {"layout_transform": "nz2nd"}
        sch[c_gm].emit_insn(c_gm_emit_axis[0], "dma_copy", dma_dict)
    else:
        if c_gm.dtype == "int8":
            n_block_outer, n_block_inner = sch[c_gm].split(c_gm.op.axis[-1], 16)
            n_shape = int_ceil_div(c_l0c.shape[-4].value, 2) * 2
            sch[c_l0c].storage_align(c_l0c.op.axis[-4], n_shape, 0)
        if a_l1.dtype == "float32" and c_gm.dtype == "float32":
            channel_split_out, chanel_split_in = sch[c_gm].split(c_gm_emit_axis[0], factor=2)
            sch[c_gm].emit_insn(chanel_split_in, "dma_copy",  {"layout_transform": "channel_split"})
        else:
            sch[c_gm].emit_insn(c_gm_emit_axis[0], "dma_copy")

    mad_dict = {
        "mad_pattern": tbe_platform.GEMM_MODE,
        "k_outer": k_axis
    }
    sch[c_l0c].emit_insn(c_l0c.op.axis[-4], "mad", mad_dict)
    emit_insn_fp_and_bt(sch, tensor_map)
    emit_insn_ub(sch, tensor_map)


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
    if tensor_map.get("deq_l1") is not None:
        deq_l1 = tensor_map["deq_l1"]
        deq_fb = tensor_map["deq_fb"]
        sch[deq_l1].emit_insn(deq_l1.op.axis[0], "dma_copy")
        sch[deq_fb].emit_insn(deq_fb.op.axis[0], "dma_copy")


def emit_insn_ub(sch, tensor_map):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :return: None
    """
    pass
