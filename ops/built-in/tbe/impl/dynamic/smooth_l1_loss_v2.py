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
smooth_l1_loss_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator


# pylint: disable=redefined-builtin,too-many-locals,too-many-arguments
@register_operator_compute("smooth_l1_loss_v2", op_mode="dynamic", support_fusion=True)
def smooth_l1_loss_v2_compute(input_predict,
                              input_label,
                              input_axis,
                              sigma,
                              reduction, 
                              kernel_name="smooth_l1_loss"):
    """
    calculating data

    Parameters
    ----------
    input_predict : TVM tensor
       the placeholder of input_predict
    input_label : TVM tensor
       the placeholder of input_label
    input_axis :  axis from classify for reduce
    sigma :  const param 
    reduction: str
       type of result, default value is "mean"
    kernel_name : str
       kernel name, default value is "smooth_l1_loss_v2"

    Returns
    -------
    output tensor
    """
    ori_dtype = input_predict.dtype
    product = tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32")
    shape = shape_util.shape_to_list(input_predict.shape)
    if product:
        input_dtype = "float32"
    else:
        input_dtype = "float16"

    tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", input_dtype)

    if ori_dtype != input_dtype:
        input_predict = tbe.cast_to(input_predict, input_dtype)
        input_label = tbe.cast_to(input_label, input_dtype)

    # xi - yi
    input_sub_res = tbe.vsub(input_predict, input_label)
    half_scalar = tvm.const(0.5, dtype=input_dtype)
    input_sub_res_half = tbe.vmuls(input_sub_res, half_scalar)
    input_sub_square_half = tbe.vmul(input_sub_res_half, input_sub_res)
    sigma_rec_scalar = tvm.const(1.0 / sigma, dtype=input_dtype)
    # (xi - yi) ** 2  * 0.5 / sigma
    method_one_res = tbe.vmuls(input_sub_square_half, sigma_rec_scalar)

    predict_label_sub_abs = tbe.vabs(input_sub_res)
    half_sigma_scalar = tvm.const(-0.5 * sigma, dtype=input_dtype)
    # |xi - yi| - 0.5 * sigma
    method_two_res = tbe.vadds(predict_label_sub_abs, half_sigma_scalar)
    sigma_scalar = tvm.const(sigma, dtype=input_dtype)
    # |xi - yi| < sigma ? (xi - yi) ** 2  * 0.5 / sigma  :  |xi - yi| - 0.5 * sigma
    res = tbe.vcmpsel(predict_label_sub_abs, sigma_scalar, 'lt', method_one_res, method_two_res)

    if reduction == "sum":
        res = tbe.reduce_sum(res, axis=input_axis["value"], keepdims=False)
    elif reduction == "mean":
        reduce_elts = 1.0
        for i in shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=input_dtype)
        else:
            cof = tbe.var("cof", dtype=input_dtype)
            if input_dtype == "float16":
                tbe.var("cof_empty", dtype=input_dtype)
        res = tbe.reduce_sum(res, axis=input_axis["value"], keepdims=False)
        res = tbe.vmuls(res, cof)

    if ori_dtype != input_dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


@register_operator("SmoothL1LossV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def smooth_l1_loss_v2(predict,
                      label,
                      loss,
                      sigma=1.0,
                      reduction="mean",
                      kernel_name="smooth_l1_loss_v2"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of input
    loss : dict
        shape and dtype of output,
        should be same shape and type as input
    sigma: float
        sigma, default value is 1
    reduction: str
        type of result, default value is "mean"
    kernel_name : str
        kernel name, default value is "smooth_l1_lossV2"

    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    para_check.check_dtype(dtype_predict, check_list, param_name="predict")

    shape_label = label.get("shape")
    dtype_label = label.get("dtype").lower()
    para_check.check_dtype(dtype_label, check_list, param_name="label")

    shape_loss = label.get("shape")
    dtype_loss = loss.get("dtype").lower()
    para_check.check_dtype(dtype_loss, check_list, param_name="loss")

    para_check.check_shape(shape_predict, param_name="predict")
    para_check.check_shape(shape_label, param_name="label")
    para_check.check_shape(shape_loss, param_name="loss")

    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()

    para_check.check_dtype(reduction_type, check_list_reduction, param_name="reduction")
    
    schedules, tensors = [], []
    axes = list(range(len(shape_predict)))
    predict["rel_pos_to_reduce"] = "before"
    label["rel_pos_to_reduce"] = "before"
    input_axis = {"shape":[len(axes), ], "value":axes, "rel_pos_to_reduce":"axis"}
    tbe_context.get_context().add_compile_info("reduction", reduction_type)
    if reduction_type != "none":
        ins = classify([predict, label, input_axis], OpPatternMode.REDUCE, {"keepdims":False})
        for (_predict, _label, _axis) in ins:
            with tbe.compute():
                predict_shape, label_shape, _ = shape_util.variable_shape([_predict, _label, _axis], op_mode="reduce")
                input_predict = tvm.placeholder(
                    predict_shape, name="predict", dtype=dtype_predict)
                input_label = tvm.placeholder(
                    label_shape, name="label", dtype=dtype_label)

                res = smooth_l1_loss_v2_compute(input_predict, input_label, _axis, sigma,
                                                reduction_type)
                tensors.append([input_predict, input_label, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = classify([predict, label], OpPatternMode.ELEWISE)
    
        for (_predict, _label) in ins:
            with tbe.compute():
                predict_shape, label_shape = shape_util.variable_shape([_predict, _label])
                input_predict = tvm.placeholder(
                    predict_shape, name="predict", dtype=dtype_predict)
                input_label = tvm.placeholder(
                    label_shape, name="label", dtype=dtype_label)

                res = smooth_l1_loss_v2_compute(input_predict, input_label, None, sigma,
                                                reduction_type)
                tensors.append([input_predict, input_label, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_1bit": False
    }

    tbe.build(schedules, config)
