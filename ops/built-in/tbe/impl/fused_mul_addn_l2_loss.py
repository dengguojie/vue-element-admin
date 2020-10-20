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
fused_mul_addn_l2_loss
"""
import te.platform as tbe_platform
from te import tvm
from te.lang import cce as tbe
from te.utils import para_check


@tbe_platform.fusion_manager.fusion_manager.register("fused_mul_addn_l2loss")
def fused_mul_addn_l2loss_compute(weight, const_input, weight_grad):
    """
    calculating data

    Parameters
    ----------
    weight : TVM tensor
        the placeholder of input_x
    const_input : TVM tensor
        the placeholder of input_x
    weight_grad : TVM tensor
        the placeholder of input_y
    kernel_name : str
        kernel name, default value is "fused_mul_addn_l2loss"

    Returns
    -------
    output tensor
    """

    # cal vmul and addn
    const_input = tbe.broadcast(const_input, weight.shape)
    data_mul = tbe.vmul(weight, const_input)
    data_addn = tbe.vadd(data_mul, weight_grad)

    axis = [i for i in range(len(weight.shape))]
    # cal l2 loss
    coeff_sqrt = tvm.const(1.0 / (2**(0.5)), dtype=weight.dtype)
    l2_loss_vmuls = tbe.vmuls(weight, coeff_sqrt)
    l2_loss_sqr = tbe.vmul(l2_loss_vmuls, l2_loss_vmuls)
    l2_loss = tbe.sum(l2_loss_sqr, axis)

    return data_addn, l2_loss


# pylint: disable=too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def fused_mul_addn_l2loss(input_x, input_y, input_z, output_x, output_y, kernel_name="fused_mul_addn_l2loss"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_y : dict
        shape and dtype of input_y
    input_z : dict
        shape and dtype of input_z
    output_x : dict
        shape and dtype of first output, which should have shape (1,) and dtype
        as input
    output_y : dict
        shape and dtype of second output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "fused_mul_addn_l2loss"

    Returns
    -------
    None
    """

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()

    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype").lower()

    shape_z = [1 for _ in range(len(shape_x))]
    dtype_z = input_z.get("dtype").lower()

    check_list = ("float16", "float32")
    # check input x attr
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    # check input y attr
    para_check.check_shape(shape_y, param_name="input_y")
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")
    # check input z attr
    para_check.check_shape(shape_z, param_name="input_z")
    para_check.check_dtype(dtype_z, check_list, param_name="input_z")

    if dtype_x != dtype_y or dtype_x != dtype_z or dtype_y != dtype_z:
        raise RuntimeError(" Three input dtype must keep the same")

    if dtype_x == "float32":
        if not tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
            raise RuntimeError("Input dtype only support float16 while input dtype is float32")

        if not tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
            raise RuntimeError("Input dtype only support float16 while input dtype is float32")

        if not tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
            raise RuntimeError("Input dtype only support float16 while input dtype is float32")

        if not tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
            raise RuntimeError("Input dtype only support float16 while input dtype is float32")

    weight = tvm.placeholder(shape_x, name="weight", dtype=dtype_x)
    weight_grad = tvm.placeholder(shape_y, name="weight_grad", dtype=dtype_y)
    const_input = tvm.placeholder(shape_z, name="const_input", dtype=dtype_z)

    res1, res2 = fused_mul_addn_l2loss_compute(weight, const_input, weight_grad)
    res_list = [res1, res2]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res_list)
    config = {"name": kernel_name, "tensor_list": [weight, weight_grad, const_input] + res_list}

    tbe.cce_build_code(sch, config)
