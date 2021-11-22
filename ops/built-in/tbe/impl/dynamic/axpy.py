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
dynamic axpy
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.common_util import get_attr


# 'pylint: disable=invalid-name,too-many-locals, unused-argument
@register_operator_compute("Axpy", op_mode="dynamic", support_fusion=True)
def axpy_compute(x1, x2, y, alpha, kernel_name="axpy"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of input_x
    x2 : TVM tensor
        the placeholder of x2
    y : dict
        dict of y, include keys(shape and dtype)
    alpha : float
        scalar of mul-factor
    kernel_name : str
        kernel name, default value is "axpy"

    Returns
    -------
    output tensor
    """
    # broadcast
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    dtype = x1.dtype.lower()

    alpha_dtype_in_ir = "float"

    # neg_1_axis_flag
    neg_1_axis_flag = 0
    if shape_x != shape_y:
        # if shape not equal, then apply broadcast.
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                  shape_y,
                                                                  param_name_input1='x1',
                                                                  param_name_input2='x2')

        for i in range(len(shape_x) - 1):
            if shape_x[i] != shape_y[i]:
                neg_1_axis_flag = 1
                break
        x1 = tbe.broadcast(x1, shape_max)
        x2 = tbe.broadcast(x2, shape_max)

    # start the main logic
    if dtype in ("float16", "float32"):
        # fp16 or fp32
        alpha_var = get_attr(alpha, "alpha", dtype, alpha_dtype_in_ir)

        if neg_1_axis_flag:
            res_muls = tbe.vmuls(x2, alpha_var)
            res = tbe.vadd(x1, res_muls)
        else:
            res = tbe.vaxpy(x2, x1, alpha_var)
    else:
        # int32
        if alpha != 1:
            # add+muls use fp32
            to_type = "float32"
            alpha_var = get_attr(alpha, "alpha", to_type, alpha_dtype_in_ir)

            input_x_cast = tbe.cast_to(x1, to_type)
            input_y_cast = tbe.cast_to(x2, to_type)

            if neg_1_axis_flag:
                res_muls = tbe.vmuls(input_y_cast, alpha_var)
                res_tmp = tbe.vadd(input_x_cast, res_muls)
            else:
                res_tmp = tbe.vaxpy(input_y_cast, input_x_cast, alpha_var)

            res = tbe.cast_to(res_tmp, dtype)

        else:
            # if alpha == 1
            res = tbe.vadd(x2, x1)

    return res


@register_operator("Axpy")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def axpy(x1, x2, y, alpha, kernel_name="axpy"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input_x
    x2 : dict
        shape and dtype of input_y
    y : dict
        shape and dtype of output, should be same shape and type as input
    alpha : float
        scalar apply to input_y:input_y*alpha
    kernel_name : str
        kernel name, default value is "axpy"

    Returns
    -------
    None
    """
    # check dtype
    dtype_list = ("float16", "float32", "int32")
    dtype_x1 = x1.get("dtype").lower()
    para_check.check_dtype(dtype_x1, dtype_list)
    dtype_x2 = x2.get("dtype").lower()
    para_check.check_dtype(dtype_x2, dtype_list)

    # produce shapes
    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x1, _input_x2) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = \
                shape_util.variable_shape([_input_x1, _input_x2])

            data_input_x1 = tvm.placeholder(shape_x1, name="data_input_x1", dtype=dtype_x1)
            data_input_x2 = tvm.placeholder(shape_x2, name="data_input_x2", dtype=dtype_x2)
            res = axpy_compute(data_input_x1, data_input_x2, y, alpha, kernel_name)
            tensors.append((data_input_x1, data_input_x2, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
