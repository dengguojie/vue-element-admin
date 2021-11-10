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
smooth_l1_loss_grad_v2
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context


# pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-arguments
@register_operator_compute("SmoothL1LossGradV2", op_mode="dynamic", support_fusion=False)
def smooth_l1_loss_grad_v2_compute(input_predict,
                                   input_label,
                                   input_dout,
                                   gradient,
                                   sigma,
                                   reduction,
                                   kernel_name):
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
    gradient : dict
        shape and dtype of output, should be same shape and type as predict
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
    product = tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32")
    if product:
        all_dtype = "float32"
    else:
        all_dtype = "float16"

    if ori_dtype != all_dtype:
        input_predict = tbe.cast_to(input_predict, all_dtype)
        input_label = tbe.cast_to(input_label, all_dtype)
        input_dout = tbe.cast_to(input_dout, all_dtype)

    # broadcast inputs
    predict_shape = shape_util.shape_to_list(input_predict.shape)
    label_shape = shape_util.shape_to_list(input_label.shape)
    dout_shape = shape_util.shape_to_list(input_dout.shape)

    predict_shape, label_shape, dout_shape, max_shape = shape_util.unify_broadcast_shapes(
        [predict_shape, label_shape, dout_shape])

    input_predict = tbe.broadcast(input_predict, max_shape)
    input_label = tbe.broadcast(input_label, max_shape)
    input_dout = tbe.broadcast(input_dout, max_shape)

    # calculate input_predict-input_label
    x = tbe.vsub(input_predict, input_label)

    # calculate |input_predict-input_label|
    x_abs = tbe.vabs(x)

    # create sigma_tensor and negative_sigma_tensor
    sigma_const = tvm.const(sigma, dtype=all_dtype)
    negative_sigma_const = tvm.const(-sigma, dtype=all_dtype)
    sigma_tensor = tbe.broadcast(sigma_const, max_shape)
    negative_sigma_tensor = tbe.broadcast(negative_sigma_const, max_shape)

    # calculate smooth
    temp = tbe.vdiv(x, sigma_tensor)
    smooth1 = tbe.vcmpsel(x, negative_sigma_tensor, 'le', -1.0, 0.0)
    smooth2 = tbe.vcmpsel(x, sigma_tensor, 'ge', 1.0, 0.0)
    smooth3_temp = tbe.vcmpsel(x_abs, sigma, 'lt', 1.0, 0.0)
    smooth3 = tbe.vmul(temp, smooth3_temp)
    smooth1_2 = tbe.vadd(smooth1, smooth2)
    smooth = tbe.vadd(smooth1_2, smooth3)

    # if choose "mean", res should divide over n
    res = tbe.vmul(smooth, input_dout)
    if reduction == "mean":
        reduce_elts = 1.0
        for i in predict_shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=all_dtype)
        else:
            cof = tbe.var("cof", dtype=all_dtype)
            if all_dtype == "float16":
                tbe.var("cof_empty", dtype=all_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", all_dtype)
        res = tbe.vmuls(res, cof)

    # calculate finish and return
    if ori_dtype != res.dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


# pylint: disable=too-many-arguments,too-many-locals
@register_operator("SmoothL1LossGradV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def smooth_l1_loss_grad_v2(predict,
                           label,
                           dout,
                           gradient,
                           sigma=1.0,
                           reduction='mean',
                           kernel_name="smooth_l1_loss_grad_v2"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of input
    dout : dict
        shape and dtype of input
    gradient : dict
        shape and dtype of output, should be same shape and type as predict
    sigma : float
        sigma
    reduction : str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad_v2"

    Returns
    -------
    None
    """
    # check input: predict label dout
    check_list = ("float16", "float32")

    dtype_predict = predict.get("dtype").lower()
    para_check.check_dtype(dtype_predict, check_list, param_name="predict")

    dtype_label = label.get("dtype").lower()
    para_check.check_dtype(dtype_label, check_list, param_name="label")

    dtype_dout = dout.get("dtype").lower()
    para_check.check_dtype(dtype_dout, check_list, param_name="dout")

    # check reduction
    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()
    para_check.check_dtype(reduction_type, check_list_reduction, param_name="reduction")

    # do compute
    ins = classify([predict, label, dout], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_predict, _label, _dout) in ins:
        with tbe.compute():
            shape_predict, shape_label, shape_dout = shape_util.variable_shape([_predict, _label, _dout])
            tensor_predict = tvm.placeholder(shape_predict, name="tensor_predict", dtype=dtype_predict)
            tensor_label = tvm.placeholder(shape_label, name="tensor_label", dtype=dtype_label)
            tensor_dout = tvm.placeholder(shape_dout, name="tensor_dout", dtype=dtype_dout)

            res = smooth_l1_loss_grad_v2_compute(tensor_predict,
                                                 tensor_label,
                                                 tensor_dout,
                                                 gradient,
                                                 sigma,
                                                 reduction_type,
                                                 kernel_name)
            tensors.append([tensor_predict, tensor_label, tensor_dout, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
