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
relu6_grad
"""
import te.platform as tbe_platform
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import broadcast_shapes
from te.utils.error_manager import error_manager_vector
from te.utils import shape_util
from topi import generic


# pylint: disable=too-many-arguments,unused-argument
# pylint: disable=ttoo-many-locals,redefined-argument-from-local
def relu6_grad_compute(input_grad, input_x, output_y, kernel_name="relu6_grad"):
    """
    Parameters
    ----------
    input_grad : TVM tensor
        the placeholder of input_grad
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6_grad"

    Returns
    ------
    compute result of relu6grad
    """
    # input_x<=6 and input_x>=0
    # min(input,6)
    min_positive_6 = tbe.vmins(input_x, 6)
    # max(input,0)
    max_zero_min_6 = tbe.vmaxs(min_positive_6, 0)

    # (X-6), X*(X-6)
    x_sub_6 = tbe.vadds(max_zero_min_6, -6)
    x_mul_x_6 = tbe.vmul(max_zero_min_6, x_sub_6)

    input_dtype = input_x.dtype
    if input_dtype == "float16":
        # algrithm : Y = X*(X-6)*1024/(X*(X-6)*1024+ESP_MIN)
        # for float16, add a small number which value is 1.18e-7, so that the divisor is not equal to 0, and for
        # accuracy, multiply by a number which value is 1024.
        x_mul_x_6_big = tbe.vmuls(x_mul_x_6, 1024)
        y_add_espmin = tbe.vadds(x_mul_x_6_big, 1.18e-7)
        y_y_esp_min = tbe.vdiv(x_mul_x_6_big, y_add_espmin)
    if input_dtype == "float32":
        # algrithm : Y = X*(X-6)/(X*(X-6)+ESP_MIN)
        # for float32, add a small number which value is 1.18e-38, so that the divisor is not equal to 0.
        y_add_espmin = tbe.vadds(x_mul_x_6, 1.18e-38)
        y_y_esp_min = tbe.vdiv(x_mul_x_6, y_add_espmin)

    x0_shape = shape_util.shape_to_list(y_y_esp_min.shape)
    x1_shape = shape_util.shape_to_list(input_grad.shape)
    _, _, y_shape = broadcast_shapes(x0_shape, x1_shape,
                                     param_name_input1="y_y_esp_min",
                                     param_name_input2="input_grad")

    input1 = tbe.broadcast(y_y_esp_min, y_shape)
    input2 = tbe.broadcast(input_grad, y_shape)

    final_res = tbe.vmul(input1, input2)

    return final_res


# pylint: disable=too-many-locals
@tbe_base.register_operator("Relu6Grad")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def relu6_grad(input_grad, input_x, output_y, kernel_name="relu6_grad"):
    """
    Parameters
    ----------
    input_grad : dict
        shape and dtype of input_grad
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6_grad"

    Returns
    ------
    None
    """
    # check input shape
    g_dtype = input_grad.get("dtype").lower()
    x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(g_dtype, check_list, param_name="input_g")
    check_dtype(x_dtype, check_list, param_name="input_x")
    if x_dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, "relu6_grad",
                                                                 "float16 while input dtype is float32", x_dtype)

    if g_dtype != x_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "input_grad", "input_x",
                                                              g_dtype, x_dtype)

    ins = classify([input_grad, input_x], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_grad, input_x) in ins:
        with tbe_base.compute():
            g_shape, x_shape = variable_shape([input_grad, input_x], support_broadcast=True)
            g_shape, x_shape = refine_shapes_for_broadcast(g_shape, x_shape)
            tensor_g = tvm.placeholder(g_shape, g_dtype, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            res = relu6_grad_compute(tensor_g, tensor_x, output_y, kernel_name)
            tensors.append((tensor_g, tensor_x, res))
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
