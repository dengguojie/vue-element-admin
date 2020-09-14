#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

TBE operator param encoder
"""
from te.domain.tiling.op_param_encode.conv2d_params_encoder \
    import Conv2dParamsEncoder
from te.domain.tiling.op_param_encode.conv2d_bp_input_params_encoder \
    import Conv2dBpInputParamsEncoder
from te.domain.tiling.op_param_encode.depthwise_conv2d_params_encoder \
    import DepthwiseConv2dParamsEncoder

# define the support encoder of operator
SUPPORT_ENCODER_MAP = {
    "conv2d": Conv2dParamsEncoder,
    "conv2d_backprop_input": Conv2dBpInputParamsEncoder,
    "depthwise_conv2d_forward": DepthwiseConv2dParamsEncoder
}
MAX_UINT32 = 4294967295
MAX_UINT16 = 65535

class ParamsEncoder():
    """
    factory class for conv2d Params Encoder
    """

    def __init__(self, op_type):
        """
        init the specific object
        """
        self.encoder = mapping_op_type(op_type)()

    def encode_array(self, info_dict):
        """
        encode the info_dict

        Parameters
        ----------
        info_dict: the input params

        Returns
        -------
        tvm.nd.array: the NDArray
        """
        self.encoder.check_info_dict(info_dict)
        return self.encoder.encode_array(info_dict)

    def decode(self, tiling_encode):
        """
        encode the info_dict

        Parameters
        ----------
        info_dict: the input params

        Returns
        -------
        tvm.nd.array: the NDArray
        """
        if tiling_encode:
            tiling = self.encoder.decode(tiling_encode)
            # AUB_shape support special value None
            if tiling["AUB_shape"][0] == 0:
                tiling["AUB_shape"] = None

            # BUB_shape support special value None
            if tiling["BUB_shape"][0] == 0:
                tiling["BUB_shape"] = None

            # AL1_shape support special value [] and None
            if tiling["AL1_shape"][0] == MAX_UINT32:
                tiling["AL1_shape"] = []
            elif tiling["AL1_shape"][0] == 0:
                tiling["AL1_shape"] = None

            # BL1_shape support special value [] and None
            if tiling["BL1_shape"][0] == 0:
                tiling["BL1_shape"] = None
            elif tiling["BL1_shape"][0] == MAX_UINT32:
                tiling["BL1_shape"] = []

            # BL0_matrix support special value []
            if tiling["BL0_matrix"][0] == MAX_UINT16:
                tiling['BL0_matrix'] = []
            return tiling
        else:
            raise TypeError("only support effective tilting, " \
                "but the return value of tiling is [%s]." % tiling_encode)


def mapping_op_type(op_type):
    """
    map the op_type to object of specific class

    Parameters
    ----------
    op_type: the input type of operator

    Returns
    -------
    class_name: the specific class
    """
    if op_type in SUPPORT_ENCODER_MAP.keys():
        return SUPPORT_ENCODER_MAP[op_type]
    else:
        raise TypeError("only support the operator: %s, \
            but the input is %s" % (SUPPORT_ENCODER_MAP.keys(), op_type))
