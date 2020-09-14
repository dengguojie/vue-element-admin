"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Get the tiling
"""
import copy
import numpy as np
import tvm
from te.domain.tiling.tiling_helper import TILING_INSTANCE
from te.domain.tiling.op_param_encode.params_encoder import ParamsEncoder
from te.domain.tiling.auto_tiling_log import AUTOTILINGLOG

# define the max value
MAX_UINT32 = 4294967295
MAX_UINT16 = 65535
MAX_UINT8 = 255
MAX_UINT4 = 15
MAX_BOOL = 1

# define the tiling_type
AUTO_TILING_TYPE = 0
CCE_TILING_TYPE = 1
REPOSITORY_TILING_TYPE = 2
PRIORITY_TILING_TYPE = 3
CUSTOM_TILING_TYPE = 4
COST_TILING_TYPE = 5
TUNING_TILING_TYPE = 6


# define the support tiling type
SUPPORT_TILING_TYPE = {
    "auto_tiling": AUTO_TILING_TYPE,
    "cce_tiling": CCE_TILING_TYPE,
    "repository_tiling": REPOSITORY_TILING_TYPE,
    "custom_tiling": CUSTOM_TILING_TYPE,
    "priority_tiling": PRIORITY_TILING_TYPE,
    "tuning_tiling": TUNING_TILING_TYPE,
    "cost_model_tiling": COST_TILING_TYPE
}


def op_shape_copy(info_dict, op_type):
    """Copy a/b shape according to op type

    Parameters
    ----------
    info_dict: the params of op
    op_type

    Returns
    -------
    a_shape, b_shape
    """
    if op_type == "conv2d_backprop_input" or op_type == "depthwise_conv2d_forward":
        a_shape = copy.deepcopy(info_dict.get("A_shape"))
        b_shape = copy.deepcopy(info_dict.get("B_shape"))
        c_shape = copy.deepcopy(info_dict.get("C_shape"))
    else:
        a_shape = copy.deepcopy(info_dict.get("a_shape"))
        b_shape = copy.deepcopy(info_dict.get("b_shape"))
        c_shape = copy.deepcopy(info_dict.get("c_shape"))

    return a_shape, b_shape, c_shape


def get_tiling(info_dict):
    """Get the tiling from module

    Parameters
    ----------
    info_dict: dict
        the params of operator

    Returns
    -------
    tiling : dict
        The result.
    """
    # check the params
    check_params(info_dict)
    op_type = copy.deepcopy(info_dict.get("op_type"))
    a_shape, b_shape, c_shape = op_shape_copy(info_dict, op_type)
    tiling_type = info_dict.get("tiling_type")

    if tiling_type not in [None, "cost_model_tiling"]:
        raise ValueError("the tiling_type is error, \
                only support %s, but tiling_stype is %s" % \
                (str(["None", "cost_model_tiling"]), str(tiling_type)))

    # encode the params of operator
    editor = ParamsEncoder(op_type)
    input_str = editor.encode_array(info_dict)
    # Read the config of tiling type, get the type of tiling
    if tiling_type == "cost_model_tiling":
        mode_old = TILING_INSTANCE.get_tiling_type()
        TILING_INSTANCE.set_tiling_type("cost_model_tiling")
    mode = TILING_INSTANCE.get_tiling_type()
    # default mode is auto_tiling
    if mode is None:
        mode = "auto_tiling"

    # check mode
    if mode not in SUPPORT_TILING_TYPE.keys():
        raise ValueError("the tiling_type is error, \
                only support %s, but mode is %s" % \
                (str(SUPPORT_TILING_TYPE.keys()), str(mode)))

    tiling_type_num = SUPPORT_TILING_TYPE.get(mode)
    if tiling_type_num != TUNING_TILING_TYPE:
        if (int(a_shape[2]) == -1 and int(a_shape[3]) == -1) or \
        int(a_shape[0]) == -1 or \
        tiling_type == "cost_model_tiling":
            tiling_result = tvm.get_global_func("_get_tiling_dynamic")
            ret = tiling_result(input_str, tiling_type_num)
            res = list(ret.asnumpy())
            tiling = decode_dynamic(res, a_shape, b_shape, c_shape)
            if tiling_type == "cost_model_tiling":
                TILING_INSTANCE.set_tiling_type(mode_old)
                mode = TILING_INSTANCE.get_tiling_type()
        else:
            tiling_result = tvm.get_global_func("_get_tiling")
            ret = tiling_result(input_str, tiling_type_num)
            tiling = editor.decode(ret)
        TILING_INSTANCE.set_params(info_dict)
    else:
        tiling = TILING_INSTANCE.get_tiling(info_dict)
    AUTOTILINGLOG.debug("[auto_tiling]info_dict:{}".format(info_dict))
    AUTOTILINGLOG.debug("[auto_tiling]tiling:{}, tiling_type:{}"\
        .format(tiling, mode))
    return tiling

def check_params(info_dict):
    """check the tiling from module

    Parameters
    ----------
    info_dict: dict
        the params of operator

    Returns
    -------

    """
    # check the info_dict
    if not isinstance(info_dict, dict):
        raise TypeError("info_dict should be dict, but the input is %s" % \
            type(info_dict))

    # check the op_type info
    if "op_type" not in info_dict.keys():
        raise KeyError("the keyword 'op_type' is missing in input params")


def decode_dynamic(tiling_encode, a_shape, b_shape, c_shape):
    """decode the information of tiling from the list of uint32 digit

    Parameters
    ----------
    tiling_encode: list of encoded tiling
        encoded tiling, includes 11 uint32 digits
    a_shape:the information of fmap shape
    b_shape: the information of weight shape
    c_shape:the information of output shape

    Returns
    -------
    tiling : list of tiling
        The decoded tiling
    """
    # according to the define of TBETilingResult structure,
    # the one member of tiling_encode occupies 32 bits
    # and includes multiple members of TBETilingResult structure
    # tiling_encode[0] includes 32-bit digits, AL1_shape_0 occupies 32-bit

    count = tiling_encode[0]
    tiling_result = []
    tiling_value = {}
    tiling_len= 29

    for i in range(count):
        shape_ha = int(tiling_encode[1 + i*tiling_len])
        shape_wa = int(tiling_encode[2 + i*tiling_len])
        shape_batchA = int(tiling_encode[3 + i*tiling_len])
        shape_hc = int(tiling_encode[4 + i*tiling_len])
        shape_wc = int(tiling_encode[5 + i*tiling_len])
        al1_shape_0 = (tiling_encode[6 + i*tiling_len] & MAX_UINT32)
        # tiling_encode[1] includes 32-bit digits, BL1_shape_0 occupies 32-bit
        bl1_shape_0 = (tiling_encode[7 + i*tiling_len] & MAX_UINT32)
        # tiling_encode[2] includes 32-bit digits,
        # AL1_shape_1 occupies low 16-bit , AL1_shape_2 occupies high 16-bit
        al1_shape_1 = (tiling_encode[8 + i*tiling_len] & MAX_UINT16)
        al1_shape_2 = ((tiling_encode[8 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[3] includes AL1_shape_3 and BL1_shape_1,
        # AL1_shape_3 occupies low 16-bit, BL1_shape_1 occupies high 16-bit
        al1_shape_3 = (tiling_encode[9 + i*tiling_len] & MAX_UINT16)
        bl1_shape_1 = ((tiling_encode[9 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[4] includes BL1_shape_2 and BL1_shape_3,
        # BL1_shape_2 occupies low 16-bit, BL1_shape_3 occupies high 16-bit
        bl1_shape_2 = (tiling_encode[10 + i*tiling_len] & MAX_UINT16)
        bl1_shape_3 = ((tiling_encode[10 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[5] includes AL0_matrix_0 and AL0_matrix_1,
        # AL0_matrix_0 occupies low 16-bit, AL0_matrix_1 occupies high 16-bit
        al0_matrix_0 = (tiling_encode[11 + i*tiling_len] & MAX_UINT16)
        al0_matrix_1 = ((tiling_encode[11 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[6] includes AL0_matrix_2, AL0_matrix_3 and AL0_matrix_4,
        # AL0_matrix_2 occupies low 8-bit, AL0_matrix_3 occupies middle 8-bit,
        # AL0_matrix_4 occupies high 16-bit
        al0_matrix_2 = (tiling_encode[12 + i*tiling_len] & MAX_UINT8)
        al0_matrix_3 = ((tiling_encode[12 + i*tiling_len] >> 8) & MAX_UINT8)
        al0_matrix_4 = ((tiling_encode[12 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[7] includes AL0_matrix_5 and BL0_matrix_0,
        # AL0_matrix_5 occupies low 16-bit, BL0_matrix_0 occupies high 16-bit
        al0_matrix_5 = (tiling_encode[13 + i*tiling_len] & MAX_UINT16)
        bl0_matrix_0 = ((tiling_encode[13 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[8] includes BL0_matrix_1, BL0_matrix_2 and BL0_matrix_3,
        # BL0_matrix_1 occupies low 16-bit,
        # BL0_matrix_2 occupies middle 8-bit,
        # BL0_matrix_3 occupies high 8-bit
        bl0_matrix_1 = (tiling_encode[14 + i*tiling_len] & MAX_UINT16)
        bl0_matrix_2 = ((tiling_encode[14 + i*tiling_len] >> 16) & MAX_UINT8)
        bl0_matrix_3 = ((tiling_encode[14 + i*tiling_len] >> 24) & MAX_UINT8)
        # tiling_encode[9] includes BL0_matrix_4 and BL0_matrix_5,
        # BL0_matrix_4 occupies low 16-bit, BL0_matrix_5 occupies high 16-bit
        bl0_matrix_4 = (tiling_encode[15 + i*tiling_len] & MAX_UINT16)
        bl0_matrix_5 = ((tiling_encode[15 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[10] includes CL0_matrix_0 and CL0_matrix_1,
        # CL0_matrix_0 occupies low 16-bit, CL0_matrix_1 occupies high 16-bit
        cl0_matrix_0 = (tiling_encode[16 + i*tiling_len] & MAX_UINT16)
        cl0_matrix_1 = ((tiling_encode[16 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[11] includes CL0_matrix_2, CL0_matrix_3
        # and CL0_matrix_4,
        # CL0_matrix_2 occupies low 8-bit, # CL0_matrix_3 occupies middle 8-bit,
        # CL0_matrix_4 occupies high 16-bit
        cl0_matrix_2 = (tiling_encode[17 + i*tiling_len] & MAX_UINT8)
        cl0_matrix_3 = ((tiling_encode[17 + i*tiling_len] >> 8) & MAX_UINT8)
        cl0_matrix_4 = ((tiling_encode[17 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[12] includes CL0_matrix_5 and CUB_matrix_0,
        # CL0_matrix_5 occupies low 16-bit, CUB_matrix_0 occupies high 16-bit
        cl0_matrix_5 = (tiling_encode[18 + i*tiling_len] & MAX_UINT16)
        cub_matrix_0 = ((tiling_encode[18 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[13] includes CUB_matrix_1, CUB_matrix_2
        # and CUB_matrix_3,
        # CUB_matrix_1 occupies low 16-bit,
        # CUB_matrix_2 occupies middle 8-bit, CUB_matrix_3 occupies high 8-bit
        cub_matrix_1 = (tiling_encode[19 + i*tiling_len] & MAX_UINT16)
        cub_matrix_2 = ((tiling_encode[19 + i*tiling_len] >> 16) & MAX_UINT8)
        cub_matrix_3 = ((tiling_encode[19 + i*tiling_len] >> 24) & MAX_UINT8)
        # tiling_encode[14] includes CUB_matrix_4 and CUB_matrix_5,
        # CUB_matrix_4 occupies low 16-bit, CUB_matrix_5 occupies high 16-bit
        cub_matrix_4 = (tiling_encode[20 + i*tiling_len] & MAX_UINT16)
        cub_matrix_5 = ((tiling_encode[20 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[15] includes AUB_shape_0, AUB_shape_0 occupies 32-bit
        aub_shape_0 = (tiling_encode[21 + i*tiling_len] & MAX_UINT32)
        # tiling_encode[16] includes BUB_shape_0, BUB_shape_0 occupies 32-bit
        bub_shape_0 = (tiling_encode[22 + i*tiling_len] & MAX_UINT32)
        # tiling_encode[17] includes AUB_shape_1 and AUB_shape_2,
        # AUB_shape_1 occupies low 16-bit, AUB_shape_2 occupies high 16-bit
        aub_shape_1 = (tiling_encode[23 + i*tiling_len] & MAX_UINT16)
        aub_shape_2 = ((tiling_encode[23 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[18] includes AUB_shape_3 and BUB_shape_1,
        # AUB_shape_3 occupies low 16-bit, BUB_shape_1 occupies high 16-bit
        aub_shape_3 = (tiling_encode[24 + i*tiling_len] & MAX_UINT16)
        bub_shape_1 = ((tiling_encode[24 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[19] includes BUB_shape_2 and BUB_shape_3,
        # BUB_shape_2 occupies low 16-bit, BUB_shape_3 occupies high 16-bit
        bub_shape_2 = (tiling_encode[25 + i*tiling_len] & MAX_UINT16)
        bub_shape_3 = ((tiling_encode[25 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[20] includes batch_dim and n_dim,
        # batch_dim occupies low 16-bit, n_dim occupies high 16-bit
        batch_dim = (tiling_encode[26 + i*tiling_len] & MAX_UINT16)
        n_dim = ((tiling_encode[26 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[21] includes m_dim and group_dim,
        # m_dim occupies low 16-bit, group_dim occupies high 16-bit
        m_dim = (tiling_encode[27 + i*tiling_len] & MAX_UINT16)
        group_dim = ((tiling_encode[27 + i*tiling_len] >> 16) & MAX_UINT16)
        # tiling_encode[22] includes AUB_pbuffer, BUB_pbuffer,
        # AL1_pbuffer, BL1_pbuffer, AL0_pbuffer, BL0_pbuffer,
        # CL0_pbuffer and CUB_pbuffer,
        # AUB_pbuffer occupies low 16-bit, BUB_pbuffer occupies middle 4-bit,
        # AL1_pbuffer occupies next 4-bit, BL1_pbuffer occupies next 4-bit,
        # AL0_pbuffer: 4 bits, BL0_pbuffer: 4 bits,
        # CL0_pbuffer: 4 bits, CUB_pbuffer: 4 bits
        aub_pbuffer = (tiling_encode[28 + i*tiling_len] & MAX_UINT4)
        bub_pbuffer = ((tiling_encode[28 + i*tiling_len] >> 4) & MAX_UINT4)
        al1_pbuffer = ((tiling_encode[28 + i*tiling_len] >> 8) & MAX_UINT4)
        bl1_pbuffer = ((tiling_encode[28 + i*tiling_len] >> 12) & MAX_UINT4)
        al0_pbuffer = ((tiling_encode[28 + i*tiling_len] >> 16) & MAX_UINT4)
        bl0_pbuffer = ((tiling_encode[28 + i*tiling_len] >> 20) & MAX_UINT4)
        cl0_pbuffer = ((tiling_encode[28 + i*tiling_len] >> 24) & MAX_UINT4)
        cub_pbuffer = ((tiling_encode[28 + i*tiling_len] >> 28) & MAX_UINT4)
        # tiling_encode[23] includes UBG_pbuffer, n_bef_batch_flag,
        # n_bef_group_flag, batch_bef_group_flag,
        # A_overhead_opt_flag and B_overhead_opt_flag,
        # UBG_pbuffer occupies low 4-bit, n_bef_batch_flag occupies next 1-bit,
        # n_bef_group_flag: 1 bits, batch_bef_group_flag: 1 bits,
        # A_overhead_opt_flag: 1 bits, B_overhead_opt_flag occupies 1 bit,
        ubg_pbuffer = (tiling_encode[29 + i*tiling_len] & MAX_UINT4)
        n_bef_batch_flag = ((tiling_encode[29 + i*tiling_len] >> 4) & MAX_BOOL)
        n_bef_group_flag = ((tiling_encode[29 + i*tiling_len] >> 5) & MAX_BOOL)
        batch_bef_group_flag = ((tiling_encode[29 + i*tiling_len] >> 6) & MAX_BOOL)
        a_overhead_opt_flag = ((tiling_encode[29 + i*tiling_len] >> 7) & MAX_BOOL)
        b_overhead_opt_flag = ((tiling_encode[29 + i*tiling_len] >> 8) & MAX_BOOL)

        # BUB_shape_2 support special value None
        if bub_shape_2 == 0:
            bub_shape_2 = None

        # BL1_shape_2 support special value None
        if bl1_shape_2 == 0:
            bl1_shape_2 = None

        # BL0_matrix_4 support special value None
        if bl0_matrix_4 == 0:
            bl0_matrix_4 = None

        # default set  channel_wise_flag

        aub_channel_wise_flag = None
        bub_channel_wise_flag = None
        cub_channel_wise_flag = True

        # Fill the dictionary of Tiling
        a_shape[0] = shape_batchA
        a_shape[2] = shape_ha
        a_shape[3] = shape_wa
        c_shape[2] = shape_hc
        c_shape[3] = shape_wc
        tiling = {"AUB_shape": [aub_shape_0, aub_shape_1, aub_shape_2, \
                                aub_shape_3], \
                "BUB_shape": [bub_shape_0, bub_shape_1, bub_shape_2, \
                                bub_shape_3], \
                "AL1_shape": [al1_shape_0, al1_shape_1, al1_shape_2, \
                                al1_shape_3], \
                "BL1_shape": [bl1_shape_0, bl1_shape_1, bl1_shape_2, \
                                bl1_shape_3], \
                "AL0_matrix": [al0_matrix_0, al0_matrix_1, al0_matrix_2, \
                                al0_matrix_3, al0_matrix_4, al0_matrix_5], \
                "BL0_matrix": [bl0_matrix_0, bl0_matrix_1, bl0_matrix_2, \
                                bl0_matrix_3, bl0_matrix_4, bl0_matrix_5], \
                "CL0_matrix": [cl0_matrix_0, cl0_matrix_1, cl0_matrix_2, \
                                cl0_matrix_3, cl0_matrix_4, cl0_matrix_5], \
                "CUB_matrix": [cub_matrix_0, cub_matrix_1, cub_matrix_2, \
                                cub_matrix_3, cub_matrix_4, cub_matrix_5], \
                "block_dim": [batch_dim, n_dim, m_dim, group_dim], \
                "n_bef_batch_flag": n_bef_batch_flag, \
                "n_bef_group_flag": n_bef_group_flag, \
                "batch_bef_group_flag": batch_bef_group_flag, \
                "A_overhead_opt_flag": a_overhead_opt_flag, \
                "B_overhead_opt_flag": b_overhead_opt_flag, \
                "AUB_channel_wise_flag": aub_channel_wise_flag, \
                "BUB_channel_wise_flag": bub_channel_wise_flag, \
                "CUB_channel_wise_flag": cub_channel_wise_flag, \
                "manual_pingpong_buffer": {"AUB_pbuffer": aub_pbuffer, \
                                            "BUB_pbuffer": bub_pbuffer, \
                                            "AL1_pbuffer": al1_pbuffer, \
                                            "BL1_pbuffer": bl1_pbuffer, \
                                            "AL0_pbuffer": al0_pbuffer, \
                                            "BL0_pbuffer": bl0_pbuffer, \
                                            "CL0_pbuffer": cl0_pbuffer, \
                                            "CUB_pbuffer": cub_pbuffer, \
                                            "UBG_pbuffer": ubg_pbuffer}}

        # AUB_shape support special value None
        if aub_shape_0 == 0:
            aub_shape_0 = None
            tiling["AUB_shape"] = aub_shape_0

        # BUB_shape support special value None
        if bub_shape_0 == 0:
            bub_shape_0 = None
            tiling["BUB_shape"] = bub_shape_0

        # AL1_shape support special value [] and None
        if al1_shape_0 == MAX_UINT32:
            al1_shape_0 = []
            tiling["AL1_shape"] = al1_shape_0
        elif al1_shape_0 == 0:
            al1_shape_0 = None
            tiling["AL1_shape"] = al1_shape_0

        # BL1_shape support special value [] and None
        if bl1_shape_0 == 0:
            bl1_shape_0 = None
            tiling["BL1_shape"] = bl1_shape_0
        elif bl1_shape_0 == MAX_UINT32:
            bl1_shape_0 = []
            tiling["BL1_shape"] = bl1_shape_0

        # BL0_matrix support special value []
        if bl0_matrix_0 == MAX_UINT16:
            tiling['BL0_matrix'] = []

        tiling_value["A_shape"] = a_shape
        tiling_value["B_shape"] = b_shape
        tiling_value["C_shape"] = c_shape
        tiling_value["tiling"] = tiling
        tiling_result.append(copy.deepcopy(tiling_value))
    return tiling_result
