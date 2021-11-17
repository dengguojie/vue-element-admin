# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic mul
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_common


# 'pylint: disable=unused-argument,too-many-locals,redefined-argument-from-local
@register_operator_compute("Mul", op_mode="dynamic", support_fusion=True)
def mul_compute(input1, input2, output, kernel_name="mul"):
    """
    calculating data's mul, c = a * b

    Parameters
    ----------
    input1: TVM tensor
        the placeholder of first input data
    input2: TVM tensor
        the placeholder of second input data
    output: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is mul

    Returns
    -------
    res : output of the data's mul
    """
    x0_shape = shape_util.shape_to_list(input1.shape)
    x1_shape = shape_util.shape_to_list(input2.shape)
    x0_shape, x1_shape, y_shape = shape_util.broadcast_shapes(x0_shape, x1_shape,
                                                              param_name_input1="input1",
                                                              param_name_input2="input2")
    input1_dtype = input1.dtype.lower()
    if input1_dtype in ("uint8", "int8"):
        input1 = tbe.cast_to(input1, "float32")
        input2 = tbe.cast_to(input2, "float32")

    input1 = tbe.broadcast(input1, y_shape)
    input2 = tbe.broadcast(input2, y_shape)
    res = tbe.vmul(input1, input2)

    if input1_dtype in ("uint8", "int8"):
        res = util_common.uint8_int8_overflow_proc(res, input1_dtype)

    return res


@register_operator("Mul")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def mul(input1, input2, output, kernel_name="mul"):
    """
    algorithm: mul
    calculating data's mul, c = a * b

    Parameters
    ----------
    input1 : dict
        include ori_shape, shape, ori_format, format, dtype and range
        dtype only support float16, float32, int32, uint8, int8
    input2 : dict
        include ori_shape, shape, ori_format, format, dtype and range
        dtype only support float16, float32, int32, uint8, int8
    output: dict
        include ori_shape, shape, ori_format, format, dtype and range
        shape must be broadcast shape of input
    kernel_name : str
        cce kernel name, default value is mul

    Returns
    -------
    None
    """

    # check dtype
    dtype_x1 = input1.get("dtype").lower()
    dtype_x2 = input2.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "int8")
    para_check.check_dtype(dtype_x1, check_list, param_name="input1")
    para_check.check_dtype(dtype_x2, check_list, param_name="input2")
    para_check.check_elewise_shape_range([input1, input1], support_broadcast=True)
    if dtype_x1 != dtype_x2:
        error_manager_vector.raise_err_inputs_dtype_not_equal("mul", "input1", "input2",
                                                              str(dtype_x1), str(dtype_x2))

    ins = classify([input1, input2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input1, input2) in ins:
        with tbe.compute():
            # shape
            shape_x1, shape_x2 = shape_util.variable_shape([input1, input2])
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=dtype_x1, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=dtype_x2, name="data_x2")
            res = mul_compute(data_x1, data_x2, output, kernel_name)

            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
