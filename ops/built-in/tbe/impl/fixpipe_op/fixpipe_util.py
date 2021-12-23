# Copyright 2021 Huawei Technologies Co., Ltd
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
fixpipe common function
"""
from typing import List
import functools
import tbe
from tbe.tvm.tensor import Tensor
from tbe.common.utils import shape_to_list
from te.platform.cce_params import scope_fb0
from te.platform.cce_params import scope_fb1
from te.platform.cce_params import scope_fb2
from te.platform.cce_params import scope_fb3
from te.platform.cce_params import scope_cbuf

DTYPE_TRANS_MAP = {
    "int4": "S4",
    "int8": "B8",
    "float16": "F16",
    "float32": "F32",
    "int32": "S32",
    "bfloat16": "BF16"
}

ANTI_QUANT_MAP = {
    "int4": "S4",
    "int8": "S8"
}

QUANT_SCALE_0_STR = "quant_scale_0"
QUANT_SCALE_1_STR = "quant_scale_1"
RELU_WEIGHT_0_STR = "relu_weight_0"
RELU_WEIGHT_1_STR = "relu_weight_1"
ELTWISE_SRC_STR = "eltwise_src"

FIXPIPE_OP_TAG = "fixpipe"
FIXPIPE_REFORM_TAG = "fixpipe_reform"

PASS_PRE_CONVERT_MODE = ["F322F32", "S322S32"]

PRE_CONVERT_MODE = ["F322F16", "F322B8", "F322S4", "F322BF16", "S322F16", "S322B8", "S322S4",
                    "VF322F16", "VF322B8", "VF322S4", "VF322BF16", "VS322F16", "VS322B8", "VS322S4"]

POST_QUANT_MODE = ["F162S4", "F162B8", "VF162S4", "VF162B8"]

FIXPIPE_VECTOR_TENSOR_LIST = [QUANT_SCALE_0_STR, QUANT_SCALE_1_STR, RELU_WEIGHT_0_STR,
                              RELU_WEIGHT_1_STR, ELTWISE_SRC_STR]

NC1HWC0_C1_IDX = 1
NC1HWC0_C0_IDX = 4

DTYPE_FLOAT32 = "float32"
DTYPE_FLOAT16 = "float16"
DTYPE_INT32 = "int32"

VECTOR_RELU_MODE = "VECTOR_RELU"
SCALAR_RELU_MODE = "SCALAR_RELU"
NORMAL_RELU_MODE = "NORMAL_RELU"

PRE_ACT_UNIT_STR = "pre_act"
POST_ACT_UNIT_STR = "post_act"

FIXPIPE_SCOPE_MAP = {
    QUANT_SCALE_0_STR: scope_fb0,
    QUANT_SCALE_1_STR: scope_fb3,
    RELU_WEIGHT_0_STR: scope_fb1,
    RELU_WEIGHT_1_STR: scope_fb2,
    ELTWISE_SRC_STR: scope_cbuf
}


def get_op_type(x: Tensor):
    if x.op.tag == "gemm":
        return "matmul"
    if len(x.op.input_tensors) == 1 and \
            x.op.input_tensors[0].name in ["mad1", "res_conv2d"]:
        return "conv2d"
    return "None"


def get_op_info_from_attrs(key: str, tensor: Tensor):
    if key not in tensor.op.attrs:
        raise RuntimeError("key [{}] not in attrs".format(key))
    return tensor.op.attrs[key]


def calc_shape_total_dim(shape: List):
    if len(shape) == 0:
        raise RuntimeError("shape cannot be []")
    dim = functools.reduce(lambda x, y: x * y, shape[:])
    return dim


def is_scaler_input(input: (Tensor, dict, None)) -> bool:
    if input is None:
        return False

    if isinstance(input, Tensor):
        input_shape = shape_to_list(get_op_info_from_attrs("ori_shape", input))
    else:
        input_shape = input.get("ori_shape")

    # scalar: ori_shape:(), check shape:[1]
    if len(input_shape) == 0:
        if isinstance(input, Tensor):
            input_shape = shape_to_list(input.shape)
        else:
            input_shape = input.get("shape")

        if len(input_shape) != 1 or input_shape[0] != 1:
            raise RuntimeError("shape should be 1 when ori_shape is empty")

    dim = calc_shape_total_dim(input_shape)
    if dim == 1:
        return True

    return False


def is_vector_input(input:(Tensor, dict, None)) -> bool:
    if input is None:
        return False

    if isinstance(input, Tensor):
        input_shape = shape_to_list(get_op_info_from_attrs("ori_shape", input))
    else:
        input_shape = input.get("ori_shape")

    if len(input_shape) == 0:
        return False

    dim = calc_shape_total_dim(input_shape)
    if dim > 1:
        return True

    return False


def get_input_scalar_value(input: (Tensor, dict, None)):
    if input is None:
        return None

    if isinstance(input, Tensor):
        const_value = get_op_info_from_attrs("const_value", input)
    else:
        const_value = input.get("const_value")

    if len(const_value) == 0:
        raise RuntimeError("scalar's const_value is empty")

    return const_value[0]


def check_fixpipe_support():
    """
    fixpipe support check
    """
    is_support_fixpipe = tbe.common.platform.platform_info.intrinsic_check_support(
        "Intrinsic_fix_pipe_unit_list")
    if not is_support_fixpipe:
        raise RuntimeError("fixpipe is not supported for current soc")
