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

import math
import json
import numpy as np
from copy import deepcopy
from te import tvm
from te.platform import CUBE_MKN
from te.domain.tiling.op_param_encode.operator_params_encoder \
    import BaseClassParamsEncoder


C04 = 4
# define the type of memory
DDR_MEMORY = 0
L1_MEMORY = 1
L2_MEMORY = 2
EMPTY_MEMORY = 3

# define the type of L1 fusion
DEFAULT_VALUE = -1
L1_DEPTH_FUSION = 0
L1_BREADTH_FUSION = 1
L1_NO_FUSION = 2
L2_FUSION = 3
L2_NO_FUSION = 2

# define the const value
CONST_VALUE0 = 0
CONST_VALUE1 = 1

# length of shape
SHAPE_LENGTH2 = 2
SHAPE_LENGTH3 = 3
SHAPE_LENGTH4 = 4
SHAPE_LENGTH5 = 5


class DepthwiseConv2dParamsEncoder(BaseClassParamsEncoder):
    """
    Child class for depthwise conv2d Params Encoder
    """

    def __init__(self):
        '''
        init the super class
        '''
        super(DepthwiseConv2dParamsEncoder, self).__init__()

    def encode_array(self, input_args):
        """
        encode the input params to tvm.nd.array

        Parameters
        ----------
        input_args: the input params

        Returns
        -------
        tvm.nd.array: the NDArray
        """
        params_in = deepcopy(input_args)
        # first: check the params from the interface
        self.check_info_dict(params_in)
        # second: preprocess the params from the interface
        self.preprocess_info_dict(params_in)

        # third: encode the params to tvm.nd.array
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
        if tiling_encode:
            tiling = json.loads(tiling_encode)
            # default set  channel_wise_flag
            tiling["AUB_channel_wise_flag"] = None
            tiling["BUB_channel_wise_flag"] = None
            tiling["CUB_channel_wise_flag"] = \
                False \
                if tiling["CUB_channel_wise_flag"] == 0 \
                else True
            return tiling
        else:
            raise TypeError("only support legal tiling, "
                    "but the return value of tiling is [%s]." % tiling_encode)

    def check_info_dict(self, params_in):
        """
        check the input params

        Parameters
        ----------
        params_in: the input params

        Returns
        -------
        """
        # preprocess the param
        params_in['C_shape'] = None
        params_in['strideH_expand'] = 1
        params_in['strideW_expand'] = 1
        params_in['fm_l1_valid_size'] = params_in.get('fm_l1_valid_size', \
            DEFAULT_VALUE)

        # check the type of param
        self.check_param_type(params_in, [dict])
        self.check_param_type(params_in.get('A_shape'), [list])
        self.check_param_type(params_in.get('B_shape'), [list])
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
        self.check_param_type(params_in.get('dilationH'), [int])
        self.check_param_type(params_in.get('dilationW'), [int])
        self.check_param_type(params_in.get(
            'fused_double_operand_num'), [int])
        self.check_param_type(params_in.get('group', CONST_VALUE1), [int])
        self.check_param_type(params_in.get('bias_flag', False), [bool, int])
        self.check_param_type(params_in.get('fm_l1_valid_size'), [int])
        self.check_param_type(params_in.get(
            'op_type', 'depthwise_conv2d_forward'), [str])

        self.check_param_type(params_in.get(
            'in_fm_memory_type'), [list])
        self.check_param_type(params_in.get(
            'out_fm_memory_type'), [list])
        self.check_param_type(params_in.get('l1_fusion_type'), [int])
        self.check_param_type(params_in.get(
            'fusion_type', CONST_VALUE0), [int])
        self.check_param_type(
            params_in.get('kernel_name', "depthwise_conv2d_kernel"), [str])

        # check the length of param
        self.check_param_length(params_in.get('A_shape'), [SHAPE_LENGTH5])
        self.check_param_length(params_in.get('B_shape'), [SHAPE_LENGTH5])

        # check the support range of param
        self.check_support_range(params_in.get('A_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('B_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('C_dtype'), self.dtype_dict)
        self.check_support_range(params_in.get('mad_dtype'),
                                 self.dtype_dict)
        self.check_support_range(params_in.get(
            'op_type', 'depthwise_conv2d_forward'), self.op_type_dict)
        # the channel align to unit of cube
        _ca0 = params_in["B_shape"][4]

        # check the fm_l1_valid_size whether is legal
        if params_in.get('fm_l1_valid_size') != DEFAULT_VALUE:
            fm_shape_size = self.input_data_byte_width[ \
                params_in.get('A_dtype')]
            for index, elt in enumerate(params_in.get('A_shape')):
                fm_shape_size *= elt
            input_data_level = params_in.get('fm_l1_valid_size') / \
                fm_shape_size
            if input_data_level not in self.input_data_level:
                raise ValueError("the fm_l1_valid_size must be \
                    1/2, 3/4, 1 times fm_shape_size, \
                    but fm_l1_valid_size is %s, fm_shape_size is %s" % \
                    (params_in.get('fm_l1_valid_size'), fm_shape_size))
            params_in['fm_l1_valid_size_level'] = input_data_level
        else:
            # if no fm_l1_valid_size param, set the default value 0
            params_in['fm_l1_valid_size_level'] = CONST_VALUE0

    def preprocess_info_dict(self, params_in):
        """encode the information of shape to the list of uint32 digit

        Parameters
        ----------
        params_in: dict of params
            include all information of shape

        Returns
        -------
        """
        def encode_memory_type(type_flag, memory_type_list):
            """
            encode the input params to encode_value

            Parameters
            ----------
            memory_type_list: the input memory list

            Returns
            -------
            encode_value: encode value
            """
            # get the length of memory_type_list
            value_length = len(memory_type_list)
            # define the encode table
            encode_table = {EMPTY_MEMORY: 0,
                            L1_MEMORY: 1,
                            L2_MEMORY: 2,
                            DDR_MEMORY: 3}
            # encode the encode_value
            encode_value = 0
            encode_index = 0
            while value_length:
                # if the type_flag is output, using ternary code
                if type_flag == "out":
                    encode_value += \
                        encode_table[memory_type_list[encode_index]] * \
                        (4**encode_index)
                # if if the type_flag is input, using binary code
                elif type_flag == "in":
                    encode_value += memory_type_list[encode_index] * \
                        (3**encode_index)
                else:
                    raise ValueError("input type_list not support")
                value_length -= 1
                encode_index += 1

            return encode_value

        # get the missing information on interface
        l1_fusion_type = params_in.get('l1_fusion_type', DEFAULT_VALUE)
        # source buffer of input and destination buffer of output
        in_fm_memory_type = params_in.get('in_fm_memory_type',
                                          [DDR_MEMORY])
        out_fm_memory_type = params_in.get('out_fm_memory_type',
                                           [DDR_MEMORY])
        # process the fusion type
        # if fuison type is depth fusion, then source and destination is DDR,
        # the l1_fusion_type is L1_no_fusion
        # 0 represent L1_depth_fusion; 1 represent L1_breadth_fusion,
        # 2 represent L1_no_fusion; 3 represent L2_fusion
        if l1_fusion_type == DEFAULT_VALUE:
            l1_fusion_type = L1_NO_FUSION
        # 2 represent L2_no_fusion; 3 represent L2_fusion
        if (L2_MEMORY in in_fm_memory_type) or \
                (L2_MEMORY in out_fm_memory_type):
            l2_fusion_type = L2_FUSION
        else:
            l2_fusion_type = L2_NO_FUSION

        # encode the memory type
        in_fm_memory_type_encode = encode_memory_type("in",
                                                      in_fm_memory_type)
        out_fm_memory_type_encode = encode_memory_type("out",
                                                       out_fm_memory_type)
        # set the default value of these params
        op_type = params_in.get('op_type', 'depthwise_conv2d_forward')
        fusion_type = params_in.get('fusion_type', CONST_VALUE0)
        kernel_name = params_in.get('kernel_name', "depthwise_conv2d_kernel")
        A_dtype = params_in.get('A_dtype', 'float16')
        B_dtype = params_in.get('B_dtype', 'float16')
        C_dtype = params_in.get('C_dtype', 'float16')
        mad_dtype = params_in.get('mad_dtype', 'float16')
        bias_flag = params_in.get('bias_flag', False)
        bias_flag = (1 if bias_flag else 0)

        # the channel align to unit of cube
        _ca0 = params_in["B_shape"][4]
        if _ca0 != C04:
            config = CUBE_MKN[params_in["B_dtype"]]
            _ca0 = config['mac'][1]

        A_shape = params_in.get('A_shape')
        A_shape[1] = (A_shape[1]*A_shape[4] + _ca0 - 1) // _ca0
        A_shape[4] = _ca0
        B_shape = params_in.get('B_shape')
        B_shape[1] = (B_shape[1]*B_shape[4] + _ca0 - 1) // _ca0
        B_shape[4] = _ca0
        C_shape = params_in.get('C_shape')
        C_shape = ([0, 0, 0, 0, 0] if C_shape is None else C_shape)

        # processing fixed-point number
        fused_double_operand_num = params_in.get("fused_double_operand_num")
        params_in["fused_double_operand_num"] = math.ceil(100 *
                                                    fused_double_operand_num)

        # transform the value of -1 to 0 for fm_l1_valid_size
        if params_in.get('fm_l1_valid_size') == DEFAULT_VALUE:
            params_in['fm_l1_valid_size'] = CONST_VALUE0
        # endocde the value of fm_l1_valid_size_level
        if params_in.get('fm_l1_valid_size_level') != CONST_VALUE0:
            raw_fm_l1_valid_size_level = \
                params_in.get('fm_l1_valid_size_level')
            params_in['fm_l1_valid_size_level'] = \
                self.input_data_level[raw_fm_l1_valid_size_level]

        # processed params
        params_in['A_shape'] = A_shape
        params_in['B_shape'] = B_shape
        params_in['C_shape'] = C_shape
        params_in['A_dtype'] = self.dtype_dict.get(A_dtype)
        params_in['B_dtype'] = self.dtype_dict.get(B_dtype)
        params_in['C_dtype'] = self.dtype_dict.get(C_dtype)
        params_in['mad_dtype'] = self.dtype_dict.get(mad_dtype)
        # l1_fusion_type have three states:  0 represent L1_depth_fusion;
        # 1 represent L1_breadth_fusion, 2 represent L1_no_fusion
        params_in['l1_fusion_type'] = l1_fusion_type
        # 2 represent L2_no_fusion; 3 represent L2_fusion
        params_in['l2_fusion_type'] = l2_fusion_type
        # source buffer of input and destination buffer of output
        params_in['in_fm_memory_type'] = in_fm_memory_type_encode
        params_in['out_fm_memory_type'] = out_fm_memory_type_encode
        # set the default value of these params
        params_in["padl"] = params_in.get("padl", CONST_VALUE0)
        params_in["padr"] = params_in.get("padr", CONST_VALUE0)
        params_in["padu"] = params_in.get("padu", CONST_VALUE0)
        params_in["padd"] = params_in.get("padd", CONST_VALUE0)
        params_in["strideH"] = params_in.get("strideH", CONST_VALUE1)
        params_in["strideW"] = params_in.get("strideW", CONST_VALUE1)
        params_in["strideH_expand"] = params_in.get("strideH_expand",
                                                    CONST_VALUE1)
        params_in["strideW_expand"] = params_in.get("strideW_expand",
                                                    CONST_VALUE1)
        params_in["dilationH"] = params_in.get("dilationH", CONST_VALUE1)
        params_in["dilationW"] = params_in.get("dilationW", CONST_VALUE1)
        params_in['group'] = params_in.get('group', CONST_VALUE1)
        params_in['bias_flag'] = bias_flag
        params_in['op_type'] = self.op_type_dict.get(op_type)
        params_in['fusion_type'] = fusion_type
        params_in['kernel_name'] = kernel_name
        # Determine whether it is dynamic shape or fixed shape
        params_in["dynamic_shape_flag"] = params_in.get(
            "dynamic_shape_flag", False)
