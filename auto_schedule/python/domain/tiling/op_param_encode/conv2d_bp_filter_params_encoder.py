#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

TBE operator param encoder
"""

import json
import math
from copy import deepcopy
from te.domain.tiling.op_param_encode.operator_params_encoder \
    import BaseClassParamsEncoder

# define const value
CONST_VALUE0 = 0
CONST_VALUE1 = 1

# define length of shape
SHAPE_LENGTH2 = 2
SHAPE_LENGTH3 = 3
SHAPE_LENGTH4 = 4
SHAPE_LENGTH5 = 5
SHAPE_LENGTH6 = 6

OP_TYPE = "conv2d_backprop_filter"
MAX_UINT32 = 4294967295
MAX_UINT16 = 65535


def decode_for_dynamic(tiling, input_args):
    #dynamic shape tiling
    tiling_result = []
    for tiling_res in tiling:
        # default set  channel_wise_flag
        tiling_res["tiling"]["AUB_channel_wise_flag"] = None
        tiling_res["tiling"]["BUB_channel_wise_flag"] = None
        tiling_res["tiling"]["CUB_channel_wise_flag"] = False
        if tiling_res["tiling"]["AUB_shape"][0] == 0:
            tiling_res["tiling"]["AUB_shape"] = None

        # BUB_shape support special value None
        if tiling_res["tiling"]["BUB_shape"][0] == 0:
            tiling_res["tiling"]["BUB_shape"] = None

        # AL1_shape support special value [] and None
        if tiling_res["tiling"]["AL1_shape"][0] == MAX_UINT32:
            tiling_res["tiling"]["AL1_shape"] = []
        elif tiling_res["tiling"]["AL1_shape"][0] == 0:
            tiling_res["tiling"]["AL1_shape"] = None

        # BL1_shape support special value [] and None
        if tiling_res["tiling"]["BL1_shape"][0] == 0:
            tiling_res["tiling"]["BL1_shape"] = None
        elif tiling_res["tiling"]["BL1_shape"][0] == MAX_UINT32:
            tiling_res["tiling"]["BL1_shape"] = []

        # BL0_matrix support special value []
        if tiling_res["tiling"]["BL0_matrix"][0] == MAX_UINT16:
            tiling_res["tiling"]['BL0_matrix'] = []
        a_shape = deepcopy(input_args["A_shape"])
        b_shape = deepcopy(input_args["B_shape"])
        c_shape = deepcopy(input_args["C_shape"])
        pad = [deepcopy(input_args["padl"]),
            deepcopy(input_args["padr"]),
            deepcopy(input_args["padu"]),
            deepcopy(input_args["padd"])]
        if input_args.get("tiling_type") is None:
            a_shape[0] = tiling_res["A_shape"][0]
            a_shape[2] = tiling_res["A_shape"][1]
            a_shape[3] = tiling_res["A_shape"][2]
            b_shape[0] = tiling_res["B_shape"][0]
            b_shape[2] = tiling_res["B_shape"][1]
            b_shape[3] = tiling_res["B_shape"][2]
            c_shape[2] = tiling_res["C_shape"][0]
            c_shape[3] = tiling_res["C_shape"][1]
            pad = tiling_res["pad"]
        tiling_res["A_shape"] = deepcopy(a_shape)
        tiling_res["B_shape"] = deepcopy(b_shape)
        tiling_res["C_shape"] = deepcopy(c_shape)
        tiling_res["pad"] = deepcopy(pad)
        tiling_result.append(tiling_res)
    return tiling_result


class Conv2dBpFilterParamsEncoder(BaseClassParamsEncoder):
    """
    Child class for conv2d backprop input Params Encoder
    """
    def __init__(self):
        super(Conv2dBpFilterParamsEncoder, self).__init__()
        self.input_args = {}

    def encode_array(self, input_args):
        """
        encode the input params to NDArray

        Parameters
        ----------
        input_args: input params

        Returns
        ----------
        NDArray: tvm.nd.array
        """
        params_in = deepcopy(input_args)
        self.input_args = params_in
        # check params
        self.check_info_dict(params_in)
        # preprocess params
        self.preprocess_info_dict(params_in)

        return self.encode(params_in)

    def decode(self, tiling_encode):
        """
        encode the input params to tvm.nd.array
        Parameters
        ----------
        input_args: the input params
        Returns
        -------
        tvm.nd.array: the NDArray
        """
        if not self.input_args["dynamic_shape_flag"] and not tiling_encode:
            raise TypeError("only support legal tiling, " \
                "but the return value of tiling is [%s]." % tiling_encode)
        elif self.input_args["dynamic_shape_flag"] and not tiling_encode:
            return []
        tiling = json.loads(tiling_encode)
        if isinstance(tiling, dict):
            #fixed shape tiling
            tiling["AUB_channel_wise_flag"] = None
            tiling["BUB_channel_wise_flag"] = None
            tiling["CUB_channel_wise_flag"] = False
            return tiling
        else:
            tiling_result = decode_for_dynamic(tiling, self.input_args)
            return tiling_result

    def check_info_dict(self, params_in):
        """
        check the type, length and support-range of input params

        Parameters
        ----------
        params_in: input params

        Returns
        """
        # check the type of param
        self.check_param_type(params_in, [dict])
        self.check_param_type(params_in.get('A_shape'), [list])
        self.check_param_type(params_in.get('B_shape'), [list])
        # check the type and value and C_shape
        if isinstance(params_in.get('C_shape'), list):
            if not (len(params_in.get('C_shape')) in \
                [SHAPE_LENGTH4, SHAPE_LENGTH5, SHAPE_LENGTH6]):
                raise ValueError("the length of param is error, \
                    only support 4 or 5")
        elif params_in.get('C_shape') is not None:
            raise TypeError("the type of param is error, \
                    only support list tuple or None, but the type of param \
                    is %s" % type(params_in.get('C_shape')))
        # check the type of param
        self.check_param_type(params_in.get('A_dtype'), [str])
        self.check_param_type(params_in.get('B_dtype'), [str])
        self.check_param_type(params_in.get('C_dtype'), [str])
        self.check_param_type(params_in.get('mad_dtype'), [str])
        self.check_param_type(params_in.get('padl'), [int])
        self.check_param_type(params_in.get('padr'), [int])
        self.check_param_type(params_in.get('padu'), [int])
        self.check_param_type(params_in.get('padd'), [int])
        self.check_param_type(params_in.get('strideH'), [int])
        self.check_param_type(params_in.get('strideW'), [int])
        self.check_param_type(params_in.get('strideH_expand'), [int])
        self.check_param_type(params_in.get('strideW_expand'), [int])
        self.check_param_type(params_in.get('dilationH'), [int])
        self.check_param_type(params_in.get('dilationW'), [int])
        self.check_param_type(params_in.get('group'), [int])
        self.check_param_type(params_in.get("fused_double_operand_num"),
                              [int, float])
        self.check_param_type(params_in.get('bias_flag'), [bool, int])
        self.check_param_type(params_in.get('op_type'), [str])

        # check the length of param
        self.check_param_length(params_in.get('A_shape'),
                        [SHAPE_LENGTH5, SHAPE_LENGTH6])
        self.check_param_length(params_in.get('B_shape'),
                        [SHAPE_LENGTH5, SHAPE_LENGTH6])
        self.check_support_range(params_in.get('A_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('B_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('C_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('mad_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('op_type'), self.op_type_dict)

    def preprocess_info_dict(self, params_in):
        """
        encode input params and set default value of input params

        Parameters
        ----------
        params_in: input params

        Returns
        """
        # set the defalut value of params
        params_in["op_type"] = self.op_type_dict.get( \
                               params_in.get("op_type", OP_TYPE))
        params_in["A_shape"] = params_in.get("A_shape")
        params_in["B_shape"] = params_in.get("B_shape")
        c_shape = params_in.get("C_shape")
        params_in["C_shape"] = ([0, 0, 0, 0] \
                                if c_shape is None else c_shape)

        params_in["A_dtype"] = self.dtype_dict.get( \
                               params_in.get("A_dtype", "float16"))
        params_in["B_dtype"] = self.dtype_dict.get( \
                               params_in.get("B_dtype", "float16"))
        params_in["C_dtype"] = self.dtype_dict.get( \
                               params_in.get("C_dtype", "float16"))
        params_in["mad_dtype"] = self.dtype_dict.get( \
                                 params_in.get("mad_dtype", "float16"))

        params_in["padl"] = params_in.get("padl", CONST_VALUE0)
        params_in["padr"] = params_in.get("padr", CONST_VALUE0)
        params_in["padu"] = params_in.get("padu", CONST_VALUE0)
        params_in["padd"] = params_in.get("padd", CONST_VALUE0)
        params_in["padf"] = params_in.get("padf", CONST_VALUE0)
        params_in["padb"] = params_in.get("padb", CONST_VALUE0)
        params_in["strideH"] = params_in.get("strideH", CONST_VALUE1)
        params_in["strideW"] = params_in.get("strideW", CONST_VALUE1)
        params_in["strideD"] = params_in.get("strideD", CONST_VALUE0)
        params_in["strideH_expand"] = params_in.get("strideH_expand", \
                                                        CONST_VALUE1)
        params_in["strideW_expand"] = params_in.get("strideW_expand", \
                                                        CONST_VALUE1)
        params_in["dilationH"] = params_in.get("dilationH", CONST_VALUE1)
        params_in["dilationW"] = params_in.get("dilationW", CONST_VALUE1)
        params_in["group"] = params_in.get("group", CONST_VALUE1)
        params_in["bias_flag"] = params_in.get("bias_flag", 0)

        # process fixed-point number (%2.f)
        fused_double_operand_num = params_in.get("fused_double_operand_num")
        params_in["fused_double_operand_num"] = math.ceil(100 * \
                                                fused_double_operand_num)
        params_in["kernel_name"] = params_in.get( \
                                   "kernel_name", OP_TYPE + "_kernel")
        #Determine whether it is dynamic shape or fixed shape
        params_in["dynamic_shape_flag"] = params_in.get( \
                                          "dynamic_shape_flag", False)