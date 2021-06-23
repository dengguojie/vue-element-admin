# Copyright 2019 Huawei Technologies Co., Ltd
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
fused_mul_add
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


SHAPE_SIZE_LIMIT = 2 ** 30  # shape limit
SIZE_SIXTEEN = 16
BATCH_MATMUL_LENGTH = 5
BATCH_LENGTH_1 = 1
BATCH_LENGTH_2 = 2
BATCH_LENGTH_3 = 3


def _shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


def _check_format(format_input0, format_input1, format_input2):
    """
    check the format_list
    """
    list_format = [format_input0, format_input1, format_input2]

    nd_format = {"ND", "NHWC", "NCHW", "HWCN"}
    standard_format = []

    for item in list_format:
        if item in nd_format:
            standard_format.append("ND")
        else:
            standard_format.append(item)

    list_pattern = [
        ["FRACTAL_NZ", "ND", "ND"],
        ["ND", "FRACTAL_NZ", "ND"],
        ["ND", "ND", "FRACTAL_NZ"],
        ["FRACTAL_NZ", "ND", "FRACTAL_NZ"]
    ]
    if standard_format in list_pattern:
        format_pattern = list_pattern.index(standard_format) + 1
    else:
        format_pattern = 0

    return format_pattern


# pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("FusedMulAdd", op_mode="dynamic", support_fusion=True)
def fused_mul_add_compute(data_input0, data_input1, data_input2,
                          output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fuesd_mul_add"

    Returns
    -------
    output tensor
    """
    # mul
    data_input0, data_input1 = _shape_broadcast(data_input0, data_input1)
    mul_result = tbe.vmul(data_input0, data_input1)

    # add
    mul_result, data_input2 = _shape_broadcast(mul_result, data_input2)
    res = tbe.vadd(mul_result, data_input2)

    return res


@register_operator("FusedMulAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fused_mul_add(input0, input1, input2,
                  output, kernel_name="fused_mul_add"):
    """
    function: fused for mul+add

    Parameters
    ----------
    input0: dict
         the dict of input of mul, support float16,float32,int32
    input1: dict
         the dict of input of mul, support float16,float32,int32
    input2: dict
         the dict of input of add, support float16,float32,int32
    output: dict
         the dict of output of add, support float16,float32,int32
    kernel_name: str
        cce kernel name, default value is fused_mul_add

    Returns
    -------
    None
    """
    # check dtype
    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()
    check_dtype_list = ["float32", "float16", "int32"]
    para_check.check_dtype(dtype_input0, check_dtype_list, param_name="input0")
    para_check.check_dtype(dtype_input1, check_dtype_list, param_name="input1")
    para_check.check_dtype(dtype_input2, check_dtype_list, param_name="input2")

    # check format
    format_input0 = input0.get("format").upper()
    format_input1 = input1.get("format").upper()
    format_input2 = input2.get("format").upper()
    _check_format(format_input0, format_input1, format_input2)

    # classify
    ins = classify([input0, input1, input2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input0, _input1, _input2) in ins:
        with tbe.compute():
            shape_input0, shape_input1, shape_input2 = shape_util.variable_shape([_input0, _input1, _input2])

            data_input0 = tvm.placeholder(shape_input0, name="data_input0", dtype=dtype_input0)
            data_input1 = tvm.placeholder(shape_input1, name="data_input1", dtype=dtype_input1)
            data_input2 = tvm.placeholder(shape_input2, name="data_input2", dtype=dtype_input2)

            res = fused_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name)

            tensor_list = [data_input0, data_input1, data_input2, res]
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
