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
from tbe.dsl.static_schedule.util import check_support_fixpipe_l0c2ub
from tbe.dsl.static_schedule.util import align as int_ceil_align
from tbe.dsl.static_schedule.util import ceil as int_ceil_div
from tbe.dsl.static_schedule.util import get_value
from tbe.dsl.static_schedule.util import shape_to_list


BATCH_MATMUL_LEN_ND = 3
BATCH_MATMUL_LEN_NZ = 5
MATMUL_LEN_ND = 2
MATMUL_LEN_NZ = 4
MULTI_FACTOR_BY_DTYPE = 2
ND2NZ_SRC_D_LIMIT = 65535
DEFAULT_DATA_SIZE = 2
K_AXIS_ALIGN_FACTOR = 2
# attach flag
KAL1_LARGE = 1
KBL1_LARGE = 2
ATTACH_FULLY_LOAD = 0
ATTACH_EQUAL = 1
ATTACH_LESS = 2
ATTACH_LARGE = 3

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
TRANS_NZ2ND = {"layout_transform": "nz2nd"}
TRANS_SPLIT = {"layout_transform": "channnel_split"}


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
def set_matmul_scope(sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: tensor of matmul which after setscope
    :return: dict
    """
    # set scope for matmul
    sch[tensor_map["a_l0a"]].set_scope(tbe_platform_info.scope_ca)
    sch[tensor_map["b_l0b"]].set_scope(tbe_platform_info.scope_cb)
    sch[tensor_map["c_l0c"]].set_scope(tbe_platform_info.scope_cc)
    if tensor_map.get("input_bias") is not None:
        tensor_map["bias_l1"] = sch.cache_read(tensor_map["input_bias"],
                                               tbe_platform_info.scope_cbuf,
                                               [tensor_map["c_l0c"]])
        tensor_map["bias_bt"] = sch.cache_read(tensor_map["bias_l1"],
                                               "local.BT",
                                               [tensor_map["c_l0c"]])

    al1 = tensor_map["a_l0a"].op.input_tensors[0]
    bl1 = tensor_map["b_l0b"].op.input_tensors[0]
    if not al1.op.input_tensors:
        tensor_map["a_l1"] = sch.cache_read(al1, tbe_platform_info.scope_cbuf, [tensor_map["a_l0a"]])
        tensor_map["a_placehold"] = al1
    elif al1.op.tag == "ND_trans_NZ":
        sch[al1].set_scope(tbe_platform_info.scope_cbuf)
        tensor_map["a_l1"] = al1
        tensor_map["a_placehold"] = al1.op.input_tensors[0]
    elif al1.op.tag == "5HD_trans_FZ":
        # origin data flow is (N,H,W,C)->(N,C1,H,W,C0)->(C1HW,N1,N0,C0)
        # can be simplified as (N,H,W,C) -> (C1HW,N1,N0,C0)
        al1_5hd = al1.op.input_tensors[0]
        sch[al1].set_scope(tbe_platform.scope_cbuf)
        tensor_map["a_l1"] = al1
        tensor_map["a_placehold"] = al1_5hd.op.input_tensors[0]
        sch[al1_5hd].compute_inline()

    if not bl1.op.input_tensors:
        tensor_map["b_l1"] = sch.cache_read(bl1, tbe_platform_info.scope_cbuf, [tensor_map["b_l0b"]])
        tensor_map["b_placehold"] = bl1
    elif bl1.op.tag == "ND_trans_NZ":
        sch[bl1].set_scope(tbe_platform_info.scope_cbuf)
        tensor_map["b_l1"] = bl1
        tensor_map["b_placehold"] = bl1.op.input_tensors[0]

    if tensor_map["fixpipe_matmul"] is not None:
        sch[tensor_map["fixpipe_matmul"]].compute_inline()


def set_out_scope(all_tensor, leaf_tensor, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all output tensor of matmul which before setscope
    :param leaf_tensor: all input tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: output tensor of matmul which after setscope
    :return: dict
    """
    res =  tensor_map["c_gm"]
    if tensor_map.get("multi_output_list") is not None:
        res = tensor_map["multi_output_list"][-1]
    if res.op.tag != "gemm":
        if res.op.tag not in ("fixpipe_reform", "dequant_NZ", "requant_NZ", "NZ_trans_ND"):
            set_matmul_ub_scope(res, all_tensor, leaf_tensor, sch, tensor_map)
        else:
            set_matmul_fixpipe_scope(res, sch, tensor_map)


def _handle_fixpipe_tensor(sch, fixpipe_tensor, tensor_map):
    """
    handle l1 and fb scope in fixpipe tensor
    """
    fixpipe_fb_list = []
    fixpipe_l1_list = []
    for idx, params_mem in enumerate(tensor_map["fixpipe_input_name"]):
        fixpipe_input = tensor_map["fixpipe_input_tensor"][idx]
        fixpipe_scope_name = FIXPIPE_SCOPE_MAP.get(get_value(params_mem))
        if fixpipe_scope_name:
            fixpipe_input_l1 = sch.cache_read(fixpipe_input, tbe_platform_info.scope_cbuf, [fixpipe_tensor])
            fixpipe_fb_list.append(sch.cache_read(fixpipe_input_l1, fixpipe_scope_name, [fixpipe_tensor]))
            fixpipe_l1_list.append(fixpipe_input_l1)
        else:
            # if elewise input is 5HD, trans to Nz on L1, else cache_read directly
            if tensor_map.get("fixpipe_trans_eltwise") is not None:
                fixpipe_input_l1 = tensor_map["fixpipe_trans_eltwise"]
                sch[fixpipe_input_l1].set_scope(tbe_platform_info.scope_cbuf)
            else:
                fixpipe_input_l1 = sch.cache_read(fixpipe_input, tbe_platform_info.scope_cbuf, [fixpipe_tensor])
            tensor_map["fixpipe_l1_eltwise"] = fixpipe_input_l1
    tensor_map["fixpipe_fb"] = fixpipe_fb_list
    tensor_map["fixpipe_l1"] = fixpipe_l1_list


def set_matmul_fixpipe_scope(res, sch, tensor_map):
    """
    set scope for matmul
    :param all_tensor: all  tensor of matmul which before setscope
    :param sch: schedule
    :param tensor_map: output fixpipe tensor of matmul which after setscope
    :return: dict
    """
    fixpipe_input_tensor = res.op.input_tensors[0]
    while fixpipe_input_tensor.op.name != "tensor_c_matrix":
        if fixpipe_input_tensor.op.tag in ("dequant_vector", "requant_vector", "fixpipe"):
            _handle_fixpipe_tensor(sch, fixpipe_input_tensor, tensor_map)
        sch[fixpipe_input_tensor].compute_inline()
        fixpipe_input_tensor = fixpipe_input_tensor.op.input_tensors[0]


def _handle_ub_input_tensor(all_tensor, leaf_tensor, sch, tensor_map):
    """
    handle the input_tensor in ub
    """
    ub_eltwise_input = []
    for tensor_mem_input, next_tensor_list in leaf_tensor.items():
        input_tensor = all_tensor[tensor_mem_input]
        if "broadcast" in input_tensor.op.tag:
            sch[input_tensor].compute_inline()
            continue
        if input_tensor in tensor_map.get("eltwise_input_tensor", []):
            ub_eltwise_input.append(sch.cache_read(input_tensor, tbe_platform_info.scope_ubuf, next_tensor_list))
    tensor_map["ub_eltwise_input"] = ub_eltwise_input


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
    fixpipe_out_tensor = all_tensor.get("tensor_c_gm")
    cache_read_list = []

    # handle the input ub tensor
    _handle_ub_input_tensor(all_tensor, leaf_tensor, sch, tensor_map)

    for tensor_mem in all_tensor.values():
        # the tensor is used to calculation and output in multi outputs scene
        if "elewise" in tensor_mem.op.tag:
            if tensor_mem == res:
                # handle the last eltwise tensor
                ub_write_tensor = sch.cache_write(tensor_mem, tbe_platform_info.scope_ubuf)
            else:
                # the eltwise between fixpipe and last tensor
                sch[tensor_mem].set_scope(tbe_platform_info.scope_ubuf)
                ub_write_tensor = tensor_mem
            if tensor_mem.op.input_tensors[0].op.tag in ("fixpipe_reform", "dequant_NZ",
                                                         "requant_NZ", "NZ_trans_ND", "gemm"):
                cache_read_list.append(ub_write_tensor)
            ub_eltwise.append(ub_write_tensor)

        if "broadcast" in tensor_mem.op.tag:
            sch[tensor_mem].compute_inline()
        if tensor_mem.op.tag in ("fixpipe_reform", "dequant_NZ", "requant_NZ", "NZ_trans_ND"):
            fixpipe_out_tensor = tensor_mem
            set_matmul_fixpipe_scope(tensor_mem, sch, tensor_map)

    # do cache_read to recognize workspace tensor
    if check_support_fixpipe_l0c2ub():
        sch[fixpipe_out_tensor].set_scope(tbe_platform_info.scope_ubuf)
    else:
        tensor_map["spec_mid_list"] = [fixpipe_out_tensor]
        tensor_map["workspace_to_ub"] = sch.cache_read(fixpipe_out_tensor,
                                                       tbe_platform_info.scope_ubuf, cache_read_list)
    tensor_map["fixpipe_out"] = fixpipe_out_tensor
    tensor_map["ub_eltwise"] = ub_eltwise


def _init_tiling_input(tensor_map):
    """
    get the a_shape and b_shape, trans_flag
    """
    a_l0a, b_l0b = tensor_map["a_l0a"], tensor_map["b_l0b"]
    l0a_shape = shape_to_list(a_l0a.shape)
    l0b_shape = shape_to_list(b_l0b.shape)
    trans_a = a_l0a.op.attrs["transpose_a"] == "true"
    trans_b = b_l0b.op.attrs["transpose_b"] == "true"
    if (trans_a == trans_b) and (a_l0a.dtype == "float32" and b_l0b.dtype == "float32"):
        # for some unaligned cases, shape_a=(2,4), shape_b=(4,16) for example, shape_a_l1 will be aligned as
        # (1,1,16,8) while shape_b_l1 is (2,1,16,8), the shapes on L0 are (1,1,16,8) and (2, 1, 16, 8), ka != kb
        l0a_shape[-3] = int_ceil_align(l0a_shape[-3], K_AXIS_ALIGN_FACTOR)
        l0b_shape[-4] = int_ceil_align(l0b_shape[-4], K_AXIS_ALIGN_FACTOR)
    # a_shape dim: batch_a, k1, m1, m0, k0
    a_shape = [1, l0a_shape[-3], l0a_shape[-4], l0a_shape[-2], l0a_shape[-1]]
    a_shape[0] = l0a_shape[0] if len(l0a_shape) == 5 else 1
    # b_shape dim: K1*k0, n1, 1, 1, n0
    b_shape = [l0b_shape[-4] * l0b_shape[-1], l0b_shape[-3], 1, 1, l0b_shape[-2]]
    return [a_shape, b_shape, trans_a, trans_b]


def cal_tiling_info_dict(tensor_map):
    """
    cal the info dict for tiling input
    :param tensor_map: output fixpipe tensor of matmul which after setscope
    :return: dict
    """
    kernel_name = tensor_map.get("c_l0c").op.attrs["kernel_name"]
    a_shape, b_shape, trans_a, trans_b = _init_tiling_input(tensor_map)
    trans_flag = 1
    if trans_a:
        trans_flag += 1
    if trans_b:
        trans_flag += 2

    info_dict = {
        "op_type": "matmul",
        "A_shape": a_shape,
        "B_shape": b_shape,
        "C_shape": None,
        "A_dtype": tensor_map["a_l0a"].dtype,
        "B_dtype": tensor_map["b_l0b"].dtype,
        "C_dtype": tensor_map["c_gm"].dtype,
        "mad_dtype": "int32" if tensor_map["a_l0a"].dtype == "int8" else "float32",
        "padl": 0,
        "padr": 0,
        "padu": 0,
        "padd": 0,
        "strideH": 1,
        "strideW": 1,
        "strideH_expand": get_fixpipe_flag(tensor_map),
        "strideW_expand": 1,
        "dilationH": trans_flag,
        "dilationW": 1,
        "group": 1,
        "bias_flag": tensor_map.get("input_bias") is not None,
        "fused_double_operand_num": get_fused_num(tensor_map),
        "kernel_name": kernel_name.value
    }
    if in_dynamic():
        info_dict_dynamic = {
            "op_tag": "matmul",
            "dynamic_shape_flag": True,
            "trans_a": trans_a,
            "trans_b": trans_b
        }
        info_dict.update(info_dict_dynamic)
    return info_dict


# handle tiling
def process_tiling(tiling_cases, tensor_list):
    """
    set scope for matmul
    :param tiling_cases: the tiling of matmul, which is list
    :param tensor_map: output fixpipe tensor of matmul which after setscope
    :return: dict
    """
    if not tbe_platform_info.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
        return tiling_cases
    for tiling_case in tiling_cases:
        if tensor_list:
            tiling_case["tensor_list"] = tensor_list
        tiling = tiling_case["tiling_strategy"]
        # is binary
        if "attach_at_flag" in tiling.keys():
            continue
        tiling["attach_at_flag"] = dict()
        al1_attach_flag, bl1_attach_flag, abkl1_attach_flag, abl1_reorder_flag =  _process_l1(tiling, tensor_list[0])
        tiling["attach_at_flag"]["al1_attach_flag"] = al1_attach_flag
        tiling["attach_at_flag"]["bl1_attach_flag"] = bl1_attach_flag
        tiling["attach_at_flag"]["abkl1_attach_flag"] = abkl1_attach_flag
        tiling["attach_at_flag"]["abl1_reorder_flag"] = abl1_reorder_flag

    return tiling_cases


def _process_l1_shape(tiling, tensor_map, para_name="AL1_shape"):
    """
    process the l1 shape attach flag
    """
    # the n,m shape of input
    m_dim = shape_to_list(tensor_map["a_l0a"].shape)[-4]
    n_dim = shape_to_list(tensor_map["b_l0b"].shape)[-3]
    k_dim = shape_to_list(tensor_map["a_l0a"].shape)[-3]

    if not tiling[para_name]:
        l1_attach_flag = ATTACH_FULLY_LOAD
        kl1_fully_load = True
        l1_parts = 1
    else:
        kl1_fully_load = (k_dim == tiling[para_name][0] or tiling[para_name][1] > 1)
        if para_name == "AL1_shape":
            l1_parts = int_ceil_div(int_ceil_div(m_dim // tiling["block_dim"][2], tiling["CL0_matrix"][1]),
                                    tiling[para_name][1])
        else:
            l1_parts = int_ceil_div(int_ceil_div(n_dim // tiling["block_dim"][1], tiling["CL0_matrix"][0]),
                                    tiling[para_name][1])
        if kl1_fully_load:
            if l1_parts == 1:
                l1_attach_flag = ATTACH_FULLY_LOAD
            elif tiling[para_name][1] == 1:
                l1_attach_flag = ATTACH_EQUAL
            else:
                l1_attach_flag = ATTACH_LARGE
        else:
            l1_attach_flag = ATTACH_LESS
    return l1_attach_flag, kl1_fully_load, l1_parts


def _process_l1k_shape(tiling, akl1_fully_load, bkl1_fully_load):
    """
    process the l1 shape attach flag
    """
    kbl1_large = (not akl1_fully_load and bkl1_fully_load) or \
                 (not akl1_fully_load and not bkl1_fully_load and tiling["AL1_shape"][0] < tiling["BL1_shape"][0])
    if akl1_fully_load and bkl1_fully_load:
        abkl1_attach_flag =  ATTACH_FULLY_LOAD
    elif kbl1_large:
        abkl1_attach_flag = KBL1_LARGE
    else:
        abkl1_attach_flag = KAL1_LARGE
    return abkl1_attach_flag


def _process_l1(tiling, tensor_map):
    """
    process the l1 shape with tiling
    """
    al1_attach_flag, akl1_fully_load, al1_parts = _process_l1_shape(tiling, tensor_map, "AL1_shape")
    bl1_attach_flag, bkl1_fully_load, bl1_parts = _process_l1_shape(tiling, tensor_map, "BL1_shape")
    abkl1_attach_flag = _process_l1k_shape(tiling, akl1_fully_load, bkl1_fully_load)
    abl1_reorder_flag = _reorder_l1_mn_axis(tiling, al1_parts, bl1_parts)

    return [al1_attach_flag, bl1_attach_flag, abkl1_attach_flag, abl1_reorder_flag]


def _reorder_l1_mn_axis(tiling, al1_m_parts, bl1_n_parts):
    """
    decide the m and n aixs order
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


def get_fixpipe_flag(tensor_map):
    """
    code the fixpipe
    :param tensor_map: tensor of matmul which after setscope
    :return: int, the flag of fixpipe
    """
    fixpipe_flag = 1
    for fixpipe_input in tensor_map.get("fixpipe_input_name", []):
        fixpipe_scope = FIXPIPE_SCOPE_MAP.get(fixpipe_input)
        if fixpipe_scope is not None:
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
    if tensor_map.get("eltwise_tensor"):
        fuse_num += 1
        for ub_eltwise_mem in tensor_map["eltwise_tensor"]:
            ub_data_size = max(ub_data_size, DATA_SIZE.get(ub_eltwise_mem, 1))
    if tensor_map.get("eltwise_input_tensor"):
        fuse_num += 1
        for ub_eltwise_input_mem in tensor_map["eltwise_input_tensor"]:
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
    al1_tensor = tensor_map["a_l0a"].op.input_tensors[0]
    bl1_tensor = tensor_map["b_l0b"].op.input_tensors[0]
    al1_shape = shape_to_list(al1_tensor.shape)
    bl1_shape = shape_to_list(bl1_tensor.shape)
    al1_dtype = al1_tensor.dtype
    bl1_dtype = bl1_tensor.dtype

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
    al1_parts = [None, 1]
    if tiling["attach_at_flag"]["al1_attach_flag"] == ATTACH_LESS:
        al1_parts[0] = tiling["AL1_shape"][0] // block_reduce // tiling["AL0_matrix"][1]
    if tiling["attach_at_flag"]["al1_attach_flag"] != ATTACH_FULLY_LOAD:
        al1_parts[1] = int_ceil_div(l0c_parts[1], tiling["AL1_shape"][1])

    bl1_parts = [None, 1]
    if tiling["attach_at_flag"]["bl1_attach_flag"] == ATTACH_LESS:
        bl1_parts[0] = tiling["BL1_shape"][0] // block_reduce // tiling["AL0_matrix"][1]
    if tiling["attach_at_flag"]["bl1_attach_flag"] != ATTACH_FULLY_LOAD:
        bl1_parts[1] = int_ceil_div(l0c_parts[0], tiling["BL1_shape"][1])

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


def split_k(c_l0c, sch, l0c_k_factor, l1_k_part, tiling):
    """
    split k dim
    :param c_l0c: the l0c tensor
    :param sch: schedule
    :param l0c_k_factor: the k factor in mmad cal
    :param l1a_k_part: the k parts from L1 to L0C
    :param tiling: tiling after process
    :return: [al1_k, bl1_k, l0k]
    """
    l0c_axis = sch[c_l0c].op.axis
    k_outer_outer, k_outer_inner = sch[c_l0c].split(sch[c_l0c].op.reduce_axis[0], l0c_k_factor)
    sch[c_l0c].reorder(k_outer_outer, *l0c_axis, k_outer_inner, sch[c_l0c].op.reduce_axis[1])
    l1a_k_part, l1b_k_part = l1_k_part
    if tiling["attach_at_flag"]["abkl1_attach_flag"] == KBL1_LARGE:
        k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(k_outer_outer, l1a_k_part)
        if l1b_k_part is None:
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(k_outer_outer_outer, nparts=1)
        else:
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                k_outer_outer_outer, l1b_k_part//l1a_k_part)
        return [k_outer_outer_outer_inner, k_outer_outer_outer_outer, k_outer_outer_inner]
    else:
        if tiling["attach_at_flag"]["abkl1_attach_flag"] == ATTACH_FULLY_LOAD:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(k_outer_outer, nparts=1)
            k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(k_outer_outer_outer, nparts=1)
        else:
            k_outer_outer_outer, k_outer_outer_inner = sch[c_l0c].split(k_outer_outer, l1b_k_part)
            if l1a_k_part is None:
                k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(k_outer_outer_outer, nparts=1)
            else:
                k_outer_outer_outer_outer, k_outer_outer_outer_inner = sch[c_l0c].split(
                    k_outer_outer_outer, l1a_k_part//l1b_k_part)
        return [k_outer_outer_outer_outer, k_outer_outer_outer_inner, k_outer_outer_inner]


# compute at of matmul
def attach_of_bias_table(sch, tensor_map, tiling, c_slice_axis, fully_load_axis):
    """
    attach tensor of bias
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param tiling: tilling after_process
    :param c_slice_axis: l0c load axis tensor for tesor
    :param fully_load_axis: fully load axis for tensor
    :return: None
    """
    if tensor_map.get("bias_l1") is not None:
        bias_l1 = tensor_map["bias_l1"]
        bias_bt = tensor_map["bias_bt"]
        sch[bias_bt].compute_at(sch[tensor_map["c_gm"]], c_slice_axis)
        if tiling["attach_at_flag"]["bl1_attach_flag"] == ATTACH_FULLY_LOAD:
            sch[bias_l1].compute_at(sch[tensor_map["c_gm"]], fully_load_axis)
        else:
            sch[bias_l1].compute_at(sch[tensor_map["c_gm"]], c_slice_axis)


def attach_of_fixpipe(sch, tensor_map, tiling, fixpipe_axis, fully_load_axis):
    """
    attach tensor of fixpipe
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param tiling: tilling after process
    :param fixpipe_axis: out load axis tensor for fixpipe tenspr
    :param fully_load_axis: fully load axis for tensor
    :return: None
    """
    for fixpipe_l1_mem in tensor_map.get("fixpipe_l1", []):
        if tiling["attach_at_flag"]["bl1_attach_flag"] == ATTACH_FULLY_LOAD:
            sch[fixpipe_l1_mem].compute_at(sch[tensor_map["c_gm"]], fully_load_axis)
        else:
            sch[fixpipe_l1_mem].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis)

    if tensor_map.get("fixpipe_l1_eltwise") is not None:
        sch[tensor_map["fixpipe_l1_eltwise"]].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis)

    for fixpipe_fb_mem in tensor_map.get("fixpipe_fb", []):
        sch[fixpipe_fb_mem].compute_at(sch[tensor_map["c_gm"]], fixpipe_axis)


def attach_of_ub(sch, tensor_map, ub_axis):
    """
    attach tensor of fixpipe
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param ub_axis: out load axis tensor for ub tenspr
    :return: None
    """
    ub_tensor_list =  tensor_map.get("ub_eltwise", []) + tensor_map.get("multi_output_list", []) + \
                      tensor_map.get("ub_eltwise_input", [])
    if tensor_map.get("fixpipe_out") is not None:
        ub_tensor_list.append(tensor_map["fixpipe_out"])
    if tensor_map.get("workspace_to_ub") is not None:
        ub_tensor_list.append(tensor_map["workspace_to_ub"])
    for ub_tensor_mem in ub_tensor_list:
        sch[ub_tensor_mem].compute_at(sch[tensor_map["c_gm"]], ub_axis)


def attach_of_l1_l0(sch, tensor_map, l1_attch_axis, tiling):
    """
    attach tensor of l1a,l1b, l0a, l0b
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param l1_attch_axis: l1a_k_axis, l1b_k_axis, l0_k_aixs, l1a_m_axis, l1b_n_axis
    :param tiling: the tiling after process
    :return: None
    """
    a_l1, b_l1 = tensor_map.get("a_l1"), tensor_map.get("b_l1")
    a_l0, b_l0, c_l0c = tensor_map.get("a_l0a"), tensor_map.get("b_l0b"), tensor_map.get("c_l0c")
    c_gm = tensor_map.get("c_gm")
    sch[a_l0].compute_at(sch[c_l0c], l1_attch_axis[2])
    sch[b_l0].compute_at(sch[c_l0c], l1_attch_axis[2])
    if tiling["attach_at_flag"]["al1_attach_flag"] != ATTACH_LESS:
        sch[a_l1].compute_at(sch[c_gm], l1_attch_axis[3])
    else:
        sch[a_l1].compute_at(sch[c_l0c], l1_attch_axis[0])
    if tiling["attach_at_flag"]["bl1_attach_flag"] != ATTACH_LESS:
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
        double_buffer_l0c(sch, tensor_map)
    if double_buffer_flag["CUB_pbuffer"] == 2:
        double_buffer_ub(sch, tensor_map)


def double_buffer_l0c(sch, tensor_map):
    """
    double buffer for bias table and fixpipe
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :return: None
    """
    db_l0c_list = tensor_map.get("fixpipe_l1", []) + tensor_map.get("fixpipe_fb", []) + [tensor_map["c_l0c"]]
    if tensor_map.get("input_bias") is not None:
        db_l0c_list += [tensor_map["bias_l1"], tensor_map["bias_bt"]]
    if tensor_map.get("fixpipe_l1_eltwise") is not None:
        db_l0c_list.append(tensor_map["fixpipe_l1_eltwise"])
    for db_l0c_mem in db_l0c_list:
        sch[db_l0c_mem].double_buffer()


def double_buffer_ub(sch, tensor_map):
    """
    double buffer for ub tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :return: None
    """
    ub_tensor_list = tensor_map.get("ub_eltwise", []) + tensor_map.get("ub_eltwise_input", [])
    if tensor_map.get("fixpipe_out") is not None and check_support_fixpipe_l0c2ub():
        ub_tensor_list.append(tensor_map["fixpipe_out"])
    if tensor_map.get("workspace_to_ub") is not None:
        ub_tensor_list.append(tensor_map["workspace_to_ub"])
    for ub_tensor_mem in ub_tensor_list:
        sch[ub_tensor_mem].double_buffer()


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
    emit_insn_l1_and_l0(sch, tensor_map, k_axis)
    emit_c_gm(sch, tensor_map, c_gm_emit_axis)
    emit_insn_fp_and_bt(sch, tensor_map)
    emit_insn_ub(sch, tensor_map)


def emit_fixpipe_from_l0c(sch, fixpipe_tensor, need_align, emit_axises, tensor_map):
    """
    emit insn for fixpipe tensor
    :param sch: schedule
    :param fixpipe_tensor: tensor from l0c
    :param need_align: if target is ub, need_align=True
    :param emit_axises: emit axis
    :param tensor_map: all tensor of matmul
    :return: None
    """
    if len(fixpipe_tensor.shape) in (MATMUL_LEN_ND, BATCH_MATMUL_LEN_ND):
        n_axis_index = 1
        if need_align:
            align_factor = tbe_platform.CUBE_MKN.get(fixpipe_tensor.dtype).get("mac")[1]
            sch[fixpipe_tensor].compute_align(fixpipe_tensor.op.axis[-1], align_factor)
            sch[fixpipe_tensor].storage_align(fixpipe_tensor.op.axis[-2], align_factor, 0)
            n_axis_index = -1
        sch[fixpipe_tensor].split(emit_axises[n_axis_index], 16)
        sch[fixpipe_tensor].emit_insn(emit_axises[0], "fixpipe_op", TRANS_NZ2ND)
    else:
        if fixpipe_tensor.dtype == "int8":
            sch[fixpipe_tensor].split(fixpipe_tensor.op.axis[-1], 16)
        if tensor_map.get("a_l1").dtype == "float32" and fixpipe_tensor.dtype == "float32":
            _, channel_split_in = sch[fixpipe_tensor].split(emit_axises[0], factor=2)
            sch[fixpipe_tensor].emit_insn(channel_split_in, "fixpipe_op", TRANS_SPLIT)
        else:
            sch[fixpipe_tensor].emit_insn(emit_axises[0], "fixpipe_op")


def emit_c_gm(sch, tensor_map, c_gm_emit_axis):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param c_gm_emit_axis: emit axis of c_gm
    :param is_nd: emit nd or nz
    :return: None
    """
    c_gm = tensor_map["c_gm"]
    fixpipe_out = tensor_map.get("fixpipe_out")
    if fixpipe_out is not None:
        emit_fixpipe_from_l0c(sch, fixpipe_out, True, fixpipe_out.op.axis, tensor_map)
        if tensor_map.get("multi_output_list"):
            for tensor in tensor_map["multi_output_list"]:
                emit_str = "fixpie_op" if tensor.op.tag == "gemm" else "dma_copy"
                sch[tensor].emit_insn(tensor.op.axis[0], emit_str)
            sch[c_gm].emit_insn(c_gm_emit_axis[0], "phony_insn")
        else:
            sch[c_gm].emit_insn(c_gm_emit_axis[0], "dma_copy")
    else:
        emit_fixpipe_from_l0c(sch, c_gm, False, c_gm_emit_axis, tensor_map)
    if tensor_map.get("workspace_to_ub") is not None:
        workspace_to_ub = tensor_map.get("workspace_to_ub")
        sch[workspace_to_ub].emit_insn(workspace_to_ub.op.axis[0], "dma_copy")


def _check_nd2nz_tag(tensor_l1):
    """
    nd2nz only support src_d <= 65535
    :param tensor_l1: nd2nz tensor on l1
    :return: bool, True while src_d > 65535 else False
    """
    if tensor_l1.op.tag not in ["5HD_trans_FZ", "ND_trans_NZ"]:
        return False

    src_d = 0
    if tensor_l1.op.tag == "5HD_trans_FZ":
        input_5hd_tensor = tensor_l1.op.input_tensors[0]
        nhwc_shape = shape_to_list(input_5hd_tensor.op.attrs["ori_shape"])
        src_d = reduce(lambda x, y: x * y, nhwc_shape[1:])
    if tensor_l1.op.tag == "ND_trans_NZ":
        nd_shape = shape_to_list(tensor_l1.op.attrs["ori_shape"])
        src_d = nd_shape[-1]
    data_size = DATA_SIZE.get(tensor_l1.dtype, DEFAULT_DATA_SIZE)

    return src_d * data_size <= ND2NZ_SRC_D_LIMIT


def _emit_insn_l1(sch, tensor_map):
    """
    emit insn for tensor on l1
    :param sch: schedule
    :param tensor_map: tensor of matmul
    """
    a_l1 = tensor_map["a_l1"]
    b_l1 = tensor_map["b_l1"]
    dma_dict = {"layout_transform": "nd2nz"}

    if _check_nd2nz_tag(a_l1):
        if a_l1.op.tag == "5HD_trans_FZ":
            tensor_5hd = a_l1.op.input_tensors[0]
            _, _, h_in, w_in, _ = shape_to_list(tensor_5hd.shape)
            #c1hw should be split as c1 and hw when trans nhwc to fractal_z
            chw_out, _ = sch[a_l1].split(a_l1.op.axis[0], h_in * w_in)
            sch[a_l1].emit_insn(chw_out, "dma_copy", dma_dict)
        else:
            sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy", dma_dict)
    else:
        sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy")

    if _check_nd2nz_tag(b_l1):
        sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy", dma_dict)
    else:
        sch[b_l1].emit_insn(b_l1.op.axis[0], "dma_copy")


def emit_insn_l1_and_l0(sch, tensor_map, k_axis):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
    :param k_axis: the outer axis of mmad
    :return: None
    """
    _emit_insn_l1(sch, tensor_map)

    a_l0a = tensor_map["a_l0a"]
    if a_l0a.dtype == "int8" and a_l0a.op.attrs["transpose_a"] == "true":
        _, a_l0a_inner = sch[a_l0a].split(a_l0a.op.axis[-4], 2) # split m1 axis
        sch[a_l0a].emit_insn(a_l0a_inner, "dma_copy")
    elif a_l0a.dtype == "float32" and a_l0a.op.attrs["transpose_a"] == "true":
        sch[a_l0a].split(a_l0a.op.axis[-2], factor=8)
        sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy", {'img2col': 1})
    else:
        sch[a_l0a].emit_insn(a_l0a.op.axis[0], "dma_copy")
    b_l0b = tensor_map["b_l0b"]
    if b_l0b.dtype == "int8" and b_l0b.op.attrs["transpose_b"] == "false" :
        _, b_l0b_inner = sch[b_l0b].split(b_l0b.op.axis[-3], 2) # split n1 axis
        sch[b_l0b].emit_insn(b_l0b_inner, "dma_copy")
    elif b_l0b.dtype == "float32" and b_l0b.op.attrs["transpose_b"] == "false":
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
    for fixpipe_fb_mem in tensor_map.get("fixpipe_fb", []):
        sch[fixpipe_fb_mem].emit_insn(fixpipe_fb_mem.op.axis[0], "dma_copy")


def emit_insn_ub(sch, tensor_map):
    """
    emit insn for all tensor
    :param sch: schedule
    :param tensor_map: tensor of matmul
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


def do_buffer_align(sch, tensor_map):
    """
    do buffer align,
    m1 and n1 should be aligned to even number to do load2d_transpose when dtype is int8
    :param sch: schedule
    :param tensor_map: tensor of matmul
    """
    a_l0a, b_l0b, c_l0c = tensor_map.get("a_l0a"), tensor_map.get("b_l0b"), tensor_map.get("c_l0c")
    trans_a = a_l0a.op.attrs["transpose_a"] == "true"
    trans_b = b_l0b.op.attrs["transpose_b"] == "true"
    m_align = (1, 2) if trans_a and a_l0a.dtype == "int8" else (1, 1)
    n_align = (1, 2) if not trans_b and b_l0b.dtype == "int8" else (1, 1)
    batch_length = len(c_l0c.shape) - MATMUL_LEN_NZ
    sch[c_l0c].buffer_align(
        *([(1, 1)] * batch_length),
        n_align,
        m_align,
        (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]),
        (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]),
        (1, 1),
        (1, tbe_platform.CUBE_MKN[a_l0a.dtype]["mac"][1])
    )
    fixpipe_l1_eltwise = tensor_map.get("fixpipe_l1_eltwise")
    if fixpipe_l1_eltwise is not None:
        sch[fixpipe_l1_eltwise].buffer_align(
            *([(1, 1)] * batch_length),
            (1, 1),
            (1, 1),
            (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][0]),
            (1, tbe_platform.CUBE_MKN[c_l0c.dtype]["mac"][2]),
        )