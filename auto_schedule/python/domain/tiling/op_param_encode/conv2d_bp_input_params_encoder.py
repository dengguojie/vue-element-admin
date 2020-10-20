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
import json
import math
from copy import deepcopy
from te.domain.tiling.op_param_encode.operator_params_encoder \
    import BaseClassParamsEncoder


# define memory type
DDR_MEMORY = 0
L1_MEMORY = 1
L2_MEMORY = 2
EMPTY_MEMORY = 3

# define L1 fusion type
DEFAULT_VALUE = -1
L1_DEPTH_FUSION = 0
L1_BREATH_FUSION = 1
L1_NO_FUSION = 2

# define const value
CONST_VALUE0 = 0
CONST_VALUE1 = 1

# define length of shape
SHAPE_LENGHT2 = 2
SHAPE_LENGHT3 = 3
SHAPE_LENGHT4 = 4
SHAPE_LENGHT5 = 5

OP_TYPE = "conv2d_backprop_input"
MAX_UINT32 = 4294967295
MAX_UINT16 = 65535


def decode_for_dynamic(tiling, input_args):
    """
    encode the input params to NDArray

    Parameters
    ----------
    input_args: input params

    Returns
    ----------
    NDArray: tvm.nd.array
    """

    # dynamic shape tiling
    tiling_result = []
    for tiling_res in tiling:
        # default set  channel_wise_flag
        tiling_res["tiling"]["AUB_channel_wise_flag"] = None
        tiling_res["tiling"]["BUB_channel_wise_flag"] = None
        tiling_res["tiling"]["CUB_channel_wise_flag"] = \
            False \
            if tiling_res["tiling"]["CUB_channel_wise_flag"] == 0 \
            else True
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


class Conv2dBpInputParamsEncoder(BaseClassParamsEncoder):
    """
    Child class for conv2d backprop input Params Encoder
    """

    def __init__(self):
        super(Conv2dBpInputParamsEncoder, self).__init__()
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
            raise TypeError("only support legal tiling, "
                            "but the return value of tiling is [%s]." % tiling_encode)
        elif self.input_args["dynamic_shape_flag"] and not tiling_encode:
            return []
        tiling = json.loads(tiling_encode)
        if isinstance(tiling, dict):
            # fixed shape tiling
            tiling["AUB_channel_wise_flag"] = None
            tiling["BUB_channel_wise_flag"] = None
            tiling["CUB_channel_wise_flag"] = \
                False if tiling["CUB_channel_wise_flag"] == 0 else True
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

        # check param types
        self.check_param_type(params_in, [dict])
        self.check_param_type(params_in.get("op_type", OP_TYPE), [str])
        self.check_param_type(params_in.get("A_shape"), [list])
        self.check_param_type(params_in.get("B_shape"), [list])
        c_shape = params_in.get("C_shape")
        if c_shape is not None:
            self.check_param_type(c_shape, [list])

        self.check_param_type(params_in.get("A_dtype"), [str])
        self.check_param_type(params_in.get("B_dtype"), [str])
        self.check_param_type(params_in.get("C_dtype"), [str])
        self.check_param_type(params_in.get("mad_dtype"), [str])

        self.check_param_type(params_in.get("padl"), [int])
        self.check_param_type(params_in.get("padr"), [int])
        self.check_param_type(params_in.get("padu"), [int])
        self.check_param_type(params_in.get("padd"), [int])
        self.check_param_type(params_in.get("strideH"), [int])
        self.check_param_type(params_in.get("strideW"), [int])
        self.check_param_type(params_in.get("strideH_expand"), [int])
        self.check_param_type(params_in.get("strideW_expand"), [int])
        self.check_param_type(params_in.get("dilationH"), [int])
        self.check_param_type(params_in.get("dilationW"), [int])
        self.check_param_type(params_in.get("group", CONST_VALUE1), [int])
        self.check_param_type(params_in.get("bias_flag", False), [bool, int])

        self.check_param_type(params_in.get("fused_double_operand_num"),
                              [int, float])

        self.check_param_type(params_in.get("in_fm_memory_type"), [list])
        self.check_param_type(params_in.get("out_fm_memory_type"), [list])
        self.check_param_type(params_in.get("l1_fusion_type"), [int])
        self.check_param_type(params_in.get(
            "fusion_type", CONST_VALUE0), [int])
        self.check_param_type(params_in.get(
            "kernel_name", OP_TYPE + "_kernel"), [str])

        # check length of params
        self.check_param_length(params_in.get("A_shape"), [SHAPE_LENGHT5])
        self.check_param_length(params_in.get("B_shape"), [SHAPE_LENGHT5])
        if c_shape is not None:
            self.check_param_length(c_shape, [SHAPE_LENGHT4, SHAPE_LENGHT5])

        # check the support range of params
        self.check_support_range(params_in.get(
            "op_type", OP_TYPE), self.op_type_dict)
        self.check_support_range(params_in.get("A_dtype"), self.dtype_dict)
        self.check_support_range(params_in.get("B_dtype"), self.dtype_dict)
        self.check_support_range(params_in.get("C_dtype"), self.dtype_dict)
        self.check_support_range(params_in.get("mad_dtype"), self.dtype_dict)

    def preprocess_info_dict(self, params_in):
        """
        encode input params and set default value of input params

        Parameters
        ----------
        params_in: input params

        Returns
        """
        def encode_memory_type(type_flag, memory_type_list):
            """
            encode input params to decimal value
            e.g. [a, b], a and b belong to [0, 1, 2] respectively
                 there will be 9 statuses (ternary to decimal)

            Parameters
            ----------
            type_flag: now support "in" or "out"
            memory_type_list: input memory list

            Returns
            -------
            encode_value: encoded value
            """
            # define encode rule
            # L2_MEMORY is not available for now
            encode_table = {EMPTY_MEMORY: 0,
                            L1_MEMORY: 1,
                            L2_MEMORY: 2,
                            DDR_MEMORY: 3}
            k = len(encode_table)
            # encode
            encode_value = 0
            for m_idx, m_type in enumerate(memory_type_list[::-1]):
                if type_flag == "in":  # no EMPTY_MEMORY
                    encode_value += encode_table[m_type] * ((k-1)**m_idx)
                elif type_flag == "out":
                    encode_value += encode_table[m_type] * (k**m_idx)
                else:
                    raise ValueError("input type_list is not support")

            return encode_value

        # get the memory type of op
        in_fm_memory_type = params_in.get("in_fm_memory_type", [DDR_MEMORY])
        out_fm_memory_type = params_in.get("out_fm_memory_type", [DDR_MEMORY])
        l1_fusion_type = params_in.get("l1_fusion_type", DEFAULT_VALUE)

        if l1_fusion_type == DEFAULT_VALUE:
            l1_fusion_type = L1_NO_FUSION

        # encode memory type
        in_fm_memory_type_encode = encode_memory_type("in", in_fm_memory_type)
        out_fm_memory_type_encode = encode_memory_type(
            "out", out_fm_memory_type)

        # set the defalut value of params
        params_in["op_type"] = self.op_type_dict.get(
            params_in.get("op_type", OP_TYPE))
        params_in["A_shape"] = params_in.get("A_shape")
        params_in["B_shape"] = params_in.get("B_shape")
        c_shape = params_in.get("C_shape")
        params_in["C_shape"] = ([0, 0, 0, 0]
                                if c_shape is None else c_shape)

        params_in["A_dtype"] = self.dtype_dict.get(
            params_in.get("A_dtype", "float16"))
        params_in["B_dtype"] = self.dtype_dict.get(
            params_in.get("B_dtype", "float16"))
        params_in["C_dtype"] = self.dtype_dict.get(
            params_in.get("C_dtype", "float16"))
        params_in["mad_dtype"] = self.dtype_dict.get(
            params_in.get("mad_dtype", "float16"))

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
        params_in["group"] = params_in.get("group", CONST_VALUE1)
        bias_flag = params_in.get("bias_flag", False)
        params_in["bias_flag"] = (1 if bias_flag else 0)

        # process fixed-point number (%2.f)
        fused_double_operand_num = params_in.get("fused_double_operand_num")
        params_in["fused_double_operand_num"] = math.ceil(100 *
                                                          fused_double_operand_num)

        params_in["in_fm_memory_type"] = in_fm_memory_type_encode
        params_in["out_fm_memory_type"] = out_fm_memory_type_encode
        params_in["l1_fusion_type"] = l1_fusion_type
        params_in["fusion_type"] = params_in.get("fusion_type", CONST_VALUE0)
        params_in["kernel_name"] = params_in.get(
            "kernel_name", OP_TYPE + "_kernel")
        # Determine whether it is dynamic shape or fixed shape
        params_in["dynamic_shape_flag"] = params_in.get(
            "dynamic_shape_flag", False)
