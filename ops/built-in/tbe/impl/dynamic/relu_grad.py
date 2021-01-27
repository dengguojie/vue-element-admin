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
relu_grad
"""
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.utils.op_utils import check_op_params
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_elewise_shape_range
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.error_manager import error_manager_vector
from te.utils import shape_util
from impl.util.platform_adapter import register_operator


# pylint: disable=too-many-locals,invalid-name
def calculate_one_or_zero(input_tensor, dtype):
    """
    if input_tensor>0, then output is 1, or input_tensor <=0, then output is 0

    Parameters
    ----------
    input_tensor: TVM tensor
        input_tensor tensor
    shape: list or tuple
        the shape of input_tensor
    dtype: tr
        he dtype of input_tensor

    returns
    ----------
    result: TVM tensor
        a tensor all value is 1 or 0
    """
    # define help constant. use help_min*help_rec_one*help_rec_sec to get the
    # result 1
    if dtype == "float32":
        help_min = tvm.const(2 ** (-126), "float32")
        help_rec_one = tvm.const(2 ** 38, "float32")
        help_rec_sec = tvm.const(2 ** 44, "float32")
    elif dtype == "float16":
        help_min = tvm.const(2 ** (-24), "float16")
        help_rec_one = tvm.const(2 ** 12, "float16")
        help_rec_sec = help_rec_one
    elif dtype == "int32":
        help_min = tvm.const(1, "int32")
        help_rec_one = help_min
        help_rec_sec = help_min

    # broadcast constant to tensor to do vmul
    shape = input_tensor.shape
    help_tensor = tbe.broadcast(help_min, shape, dtype)
    help_zero_tensor = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    help_rec_one_tensor = tbe.broadcast(help_rec_one, shape, dtype)
    help_rec_sec_tensor = tbe.broadcast(help_rec_sec, shape, dtype)

    # process to get tmp_min_y in (input_tensor, help_tensor)
    tmp_min_y = tbe.vmin(input_tensor, help_tensor)
    # process to get tmp_max_y in (help_zero_tensor, help_tensor)
    tmp_max_y = tbe.vmax(tmp_min_y, help_zero_tensor)
    result_tmp = tbe.vmul(tmp_max_y, help_rec_one_tensor)
    if dtype == "float32":
        result_tmp = tbe.vmul(result_tmp, help_rec_sec_tensor)
    result = tbe.vmul(result_tmp, help_rec_sec_tensor)

    return result


# pylint: disable=locally-disabled,unused-argument
def relu_grad_compute(input_gradients, input_features, output_backprops,
                      kernel_name="relu_grad"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).

    Parameters
    ----------
    input_gradients: TVM tensor
        input tensor of grad
    input_features: TVM tensor
        input tensor of relu output
    output_backprops: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad"

    Returns
    -------
    res: TVM tensor
        the result of relu_grad_compute
    """
    dtype = input_gradients.dtype
    trans_type = dtype

    # need cast int8 or uint8 to float16
    if dtype in ("int8", "uint8"):
        input_gradients = tbe.cast_to(input_gradients, "float16")
        input_features = tbe.cast_to(input_features, "float16")
        trans_type = "float16"

    x0_shape = shape_util.shape_to_list(input_gradients.shape)
    x1_shape = shape_util.shape_to_list(input_features.shape)
    _, _, y_shape = shape_util.broadcast_shapes(x0_shape, x1_shape,
                                                param_name_input1="input_gradients",
                                                param_name_input2="input_features")
    input_gradients = tbe.broadcast(input_gradients, y_shape)
    input_features = tbe.broadcast(input_features, y_shape)

    derivative_relu = calculate_one_or_zero(input_features, trans_type)

    result = tbe.vmul(input_gradients, derivative_relu)

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8"):
        result = tbe.cast_to(result, dtype, f1628IntegerFlag=True)

    return result


@register_operator("ReluGrad")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def relu_grad(input_gradients, input_features, output_backprops, kernel_name="relu_grad"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).
    support dtype:float16,float32,int32,int8,uint8

    Parameters
    ----------
    input_gradients: dict
        the backpropagated gradients to the corresponding relu operation
    input_features: dict
        the features passed as output of relu operation
    output_backprops: dict
        the output of relu back propagation
    kernel_name: str
        cce kernel name, default value is "relu_grad"

    Returns
    -------
    None
    """
    g_dtype = input_gradients.get("dtype").lower()
    x_dtype = input_features.get("dtype").lower()
    check_list =("float16", "float32", "int32", "int8", "uint8")
    check_dtype(g_dtype, check_list, param_name="input_gradients")
    check_dtype(x_dtype, check_list, param_name="input_features")
    check_elewise_shape_range([input_gradients, input_features], support_broadcast=True)
    if g_dtype != x_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "input_gradients", "input_features",
                                                              g_dtype, x_dtype)
    ins = classify([input_gradients, input_features], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (g, x) in ins:
        with tbe_base.compute():
            g_shape, x_shape = shape_util.variable_shape([g, x], support_broadcast=True)
            tensor_g = tvm.placeholder(g_shape, g_dtype, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            res = relu_grad_compute(tensor_g, tensor_x, output_backprops,
                                    kernel_name)
            tensors.append((tensor_g, tensor_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
