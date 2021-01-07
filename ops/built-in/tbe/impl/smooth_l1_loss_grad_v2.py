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
smooth_l1_loss_grad_v2
"""
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
import te.platform as tbe_platform


# pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-arguments
@tbe_platform.fusion_manager.fusion_manager.register("smooth_l1_loss_grad_v2")
def smooth_l1_loss_grad_v2_compute(input_predict,
                                   input_label,
                                   input_dout,
                                   sigma,
                                   reduction):
    """
    calculating data

    Parameters
    ----------
    input_predict : TVM tensor
       the placeholder of input_predict
    input_label : TVM tensor
       the placeholder of input_label
    input_dout : TVM tensor
        the placeholder of input_dout
    sigma : float
        default value is 1.0
    reduction: str
       type of result, default value is "mean"
    kernel_name : str
       kernel name, default value is "smooth_l1_loss_grad_v2"

    Returns
    -------
    output tensor
    """

    ori_dtype = input_predict.dtype
    all_shape = input_predict.shape
    all_dtype = "float32"

    if ori_dtype == "float16":
        input_predict = tbe.cast_to(input_predict, all_dtype)
        input_label = tbe.cast_to(input_label, all_dtype)
        input_dout = tbe.cast_to(input_dout, all_dtype)

    # calculate input_predict-input_label
    x = tbe.vsub(input_predict, input_label)

    # calculate |input_predict-input_label|
    x_abs = tbe.vabs(x)

    # create sigma_tensor and negative_sigma_tensor
    sigma_const = tvm.const(sigma, dtype=all_dtype)
    negative_sigma_const = tvm.const(-sigma, dtype=all_dtype)
    sigma_tensor = tbe.broadcast(sigma_const, all_shape)
    negative_sigma_tensor = tbe.broadcast(negative_sigma_const, all_shape)

    # calculate smooth
    temp = tbe.vdiv(x, sigma_tensor)
    smooth1 = tbe.vcmpsel(x, negative_sigma_tensor, 'le', -1.0, 0.0)
    smooth2 = tbe.vcmpsel(x, sigma_tensor, 'ge', 1.0, 0.0)
    smooth3_temp = tbe.vcmpsel(x_abs, sigma, 'lt', 1.0, 0.0)
    smooth3 = tbe.vmul(temp, smooth3_temp)
    smooth1_2 = tbe.vadd(smooth1, smooth2)
    smooth = tbe.vadd(smooth1_2, smooth3)

    # calculate the res value and return
    res = tbe.vmul(smooth, input_dout)

    # choose dtype
    if ori_dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def smooth_l1_loss_grad_v2(predict, label, dout, gradient, sigma=1.0, reduction='mean',
                           kernel_name="smooth_l1_loss_grad_v2"):
    """
    smooth_l1_loss_grad_v2
    """
    # check input: predict label dout
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    para_check.check_dtype(dtype_predict, check_list, param_name="predict")

    shape_label = label.get("shape")
    dtype_label = label.get("dtype").lower()
    para_check.check_dtype(dtype_label, check_list, param_name="label")

    shape_dout = dout.get("shape")
    dtype_dout = dout.get("dtype").lower()
    para_check.check_dtype(dtype_dout, check_list, param_name="dout")

    para_check.check_shape(shape_predict, param_name="predict")
    para_check.check_shape(shape_label, param_name="label")
    para_check.check_shape(shape_dout, param_name="dout")

    # check reduction
    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()

    para_check.check_dtype(reduction_type, check_list_reduction, param_name="reduction")

    input_predict = tvm.placeholder(
        shape_predict, name="predict", dtype=dtype_predict)
    input_label = tvm.placeholder(
        shape_label, name="label", dtype=dtype_label)
    input_dout = tvm.placeholder(
        shape_dout, name="dout", dtype=dtype_dout)

    res = smooth_l1_loss_grad_v2_compute(input_predict, input_label, input_dout, sigma, reduction_type)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [input_predict, input_label, input_dout, res]
    }

    tbe.cce_build_code(sch, config)
