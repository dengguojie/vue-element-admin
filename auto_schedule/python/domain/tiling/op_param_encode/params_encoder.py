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
TBE operator param encoder
"""
from te.domain.tiling.op_param_encode.conv2d_params_encoder import Conv2dParamsEncoder
from te.domain.tiling.op_param_encode.conv2d_bp_input_params_encoder import Conv2dBpInputParamsEncoder
from te.domain.tiling.op_param_encode.conv2d_bp_filter_params_encoder import Conv2dBpFilterParamsEncoder
from te.domain.tiling.op_param_encode.depthwise_conv2d_params_encoder import DepthwiseConv2dParamsEncoder

# define the support encoder of operator
SUPPORT_ENCODER_MAP = {
    "conv2d": Conv2dParamsEncoder,
    "conv2d_backprop_input": Conv2dBpInputParamsEncoder,
    "conv2d_backprop_filter": Conv2dBpFilterParamsEncoder,
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
        tiling = self.encoder.decode(tiling_encode)
        # if the mode is dynamic shape tiling is a list
        if isinstance(tiling, list):
            # dynamic shape tiling
            return tiling

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
        raise TypeError("only support the operator: %s, but the input is %s" % (SUPPORT_ENCODER_MAP.keys(), op_type))
