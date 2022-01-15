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
bn_training_reduce_grad
"""
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform


def _check_format_nd(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW", "NDC1HWC0"):
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad",
                                                      "The data format only supports NC1HWC0,NDC1HWC0,and NCHW.")
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad",
                                                          "The origin format only supports NCHW when format is NCHW")


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
# 'pylint: disable=unused-argument,invalid-name
def bn_training_reduce_grad_compute(grads,
                                    x,
                                    diff_scale,
                                    diff_offset,
                                    scale,
                                    batch_mean,
                                    batch_variance,
                                    y,
                                    epsilon,
                                    kernel_name="bn_training_reduce_grad"):
    """
    Compute for batch_norm_train_reduce_grad
    y:(grads*scale*np.power((batch_variance + epsilon), (-0.5)))+
      np.sum(grads*scale*(-0.5)*x_norm*np.power((batch_variance+epsilon),(-1))))
      *(2/m)+np.sum(grads*scale*(-1)*
      np.power((batch_variance+epsilon),(-0.5)))*(1/m)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads.
        Must be one of the following type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x.
        Must be one of the following type: `float32`, 'float16.
    diff_scale: TVM tensor 5D
        the placeholder of diff_scale.
        Must be one of the following type: `float32`.
    diff_offset: TVM tensor 5D
         the placeholder of diff_offset.
         Must be one of the following types: `float32`.
    scale: TVM tensor 5D
        the placeholder of scale.
        Must be one of the following types: `float32`.
    batch_mean: dict 5D
        the placeholder of batch_mean.
        Must be one of the following types: `float32`.
    batch_variance: dict 5D
        the placeholder of batch_variance.
        Must be one of the following types: `float32`.
    y: dict
        dict of y, include keys(shape and dtype).
    epsilon: float
        A small float number added to the variance of x.

    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad"

    Returns
    -------
    res: TVM tensor
    """
    shape_grads = shape_util.shape_to_list(grads.shape)

    is_cast = False

    if not tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        error_reson = "Platform does not support float32 vdiv."
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
        return []

    if grads.dtype == "float16":
        is_cast = True
        grads = tbe.cast_to(grads, "float32")

    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")
        is_cast = True

    tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", "float32")

    num_rec = tbe.var("num_rec", dtype="float32")
    neg_num_rec = tbe.var("neg_num_rec", dtype="float32")

    data_sqrt = tbe.vsqrt(tbe.vadds(batch_variance, epsilon))
    scale_inv = tbe.vmuls(diff_scale, num_rec)
    scale_inv_reverse = tbe.vmuls(diff_scale, neg_num_rec)
    offset_inv_reverse = tbe.vmuls(diff_offset, neg_num_rec)

    multiplier = tbe.vdiv(scale_inv_reverse, data_sqrt)
    addend_div = tbe.vdiv(batch_mean, data_sqrt)
    addend_mul = tbe.vmul(addend_div, scale_inv)
    addend = tbe.vadd(addend_mul, offset_inv_reverse)

    multiplier_broadcast = tbe.broadcast(multiplier, shape_grads)
    addend_broadcast = tbe.broadcast(addend, shape_grads)

    coef_mul = tbe.vmul(multiplier_broadcast, x)
    coef_add = tbe.vadd(grads, coef_mul)
    coef = tbe.vadd(coef_add, addend_broadcast)

    mul_scale = tbe.vdiv(scale, data_sqrt)
    mul_scale_broadcast = tbe.broadcast(mul_scale, shape_grads)

    res = tbe.vmul(coef, mul_scale_broadcast)

    if is_cast:
        res = tbe.cast_to(res, "float16")
    return res


def _check_shape(shape_grads, shape_diff_scale, data_format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_grads: list or tuple
        input grads's data shape
    shape_diff_scale: list or tuple
        input diff_scale's data shape
    Returns
    -------
    None
    """
    para_check.check_shape(shape_grads, param_name="grads")
    para_check.check_shape(shape_diff_scale, param_name="diff_scale")
    dim_c0 = 0
    if data_format == "NDC1HWC0":
        dim_c0 = shape_grads[5]
        n_shape = shape_diff_scale[0] * shape_diff_scale[1]
        if n_shape != 1 or shape_diff_scale[3] != 1 or shape_diff_scale[4] != 1:
            error_reson = "Dimensions except Dimension C must be one for shape_diff_scale"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
        if shape_diff_scale[5] != dim_c0:
            error_reson = "Dimension C must be equal"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
    else:
        dim_c0 = shape_grads[4]
        if shape_diff_scale[0] != 1 or shape_diff_scale[2] != 1 or shape_diff_scale[3] != 1:
            error_reson = "Dimensions except Dimension C must be one for shape_diff_scale"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
        if shape_diff_scale[4] != dim_c0:
            error_reson = "Dimension C must be equal"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)

    if len(shape_grads) not in (5, 6) or len(shape_diff_scale) not in (5, 6):
        error_reson = "This operator can only support 5D,6D, but some input's shape length is not 5 or 6"
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
    if dim_c0 != 16:
        error_reson = "shape_grads last dim must be 16"
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)


# 'pylint: disable=too-many-statements
@register_operator("BNTrainingReduceGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def bn_training_reduce_grad(grads,
                            x,
                            diff_scale,
                            diff_offset,
                            scale,
                            batch_mean,
                            batch_variance,
                            y,
                            epsilon=0.0001,
                            kernel_name="bn_training_reduce_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_reduce_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
        source data type, support "float32", "float16".
    x: dict
        dict of s, A 5D Tensor for input x.
        source data type, support "float32", "float16".
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for input diff_scale.
        The output of bn_training_update_grad.
        source data type, support "float32".
    diff_offset: dict
        dict of diff_offset, A 5HD Tensor for input diff_offset.
        The output of bn_training_update_grad.
        source data type, support "float32".
    scale: dict
        dict of scale, A 5HD Tensor for input scale.
        source data type, support "float32".
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
        source data type, support "float32".
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
        source data type, support "float32".
    y: dict
        dict of output, A `Tensor`. Has the same type as `grads`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_x = x.get("shape")
    shape_diff_scale = diff_scale.get("shape")
    shape_util.compare_tensor_dict_key(grads, x, "shape")

    dtype_grads = grads.get("dtype")
    dtype_x = x.get("dtype")
    dtype_diff_scale = diff_scale.get("dtype")
    dtype_diff_offset = diff_offset.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_batch_mean = batch_mean.get("dtype")
    dtype_batch_variance = batch_variance.get("dtype")

    input_grads_dtype = dtype_grads.lower()
    x_dtype = dtype_x.lower()
    diff_scale_dtype = dtype_diff_scale.lower()
    diff_offset_dtype = dtype_diff_offset.lower()
    scale_dtype = dtype_scale.lower()
    batch_mean_dtype = dtype_batch_mean.lower()
    batch_variance_dtype = dtype_batch_variance.lower()

    para_check.check_dtype(input_grads_dtype, ("float32", "float16"), param_name="grads")
    para_check.check_dtype(x_dtype, ("float32", "float16"), param_name="x")
    para_check.check_dtype(diff_scale_dtype, ("float32",), param_name="diff_scale")
    para_check.check_dtype(diff_offset_dtype, ("float32",), param_name="diff_offset")
    para_check.check_dtype(scale_dtype, ("float32",), param_name="scale")
    para_check.check_dtype(batch_mean_dtype, ("float32",), param_name="batch_mean")
    para_check.check_dtype(batch_variance_dtype, ("float32",), param_name="batch_variance")

    shape_util.compare_tensor_dict_key(diff_scale, diff_offset, "shape")
    shape_util.compare_tensor_dict_key(diff_scale, scale, "shape")
    shape_util.compare_tensor_dict_key(diff_scale, batch_mean, "shape")
    shape_util.compare_tensor_dict_key(diff_scale, batch_variance, "shape")
    shape_util.compare_tensor_dict_key(grads, x, "shape")

    data_format = grads.get("format").upper()
    ori_format = grads.get("ori_format").upper()
    _check_format_nd(data_format, ori_format)

    if data_format in ("NC1HWC0", "NDC1HWC0"):
        _check_shape(shape_grads, shape_diff_scale, data_format)
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]

    ins = classify([grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance],
                   OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedules, tensors = [], []

    for (_grads, _x, _diff_scale, _diff_offset, _scale, _batch_mean, _batch_variance) in ins:
        with tbe.compute():
            _shape_grads, _, _, _, _shape_scale, _, _ = \
                shape_util.variable_shape([_grads, _x, _diff_scale, _diff_offset, _scale, _batch_mean, _batch_variance])

            grads_input = tvm.placeholder(_shape_grads, name="grads_input", dtype=input_grads_dtype)
            x_input = tvm.placeholder(_shape_grads, name="x_input", dtype=x_dtype)
            diff_scale_input = tvm.placeholder(_shape_scale, name="diff_scale_input", dtype=diff_scale_dtype)
            diff_offset_input = tvm.placeholder(_shape_scale, name="diff_offset_input", dtype=diff_offset_dtype)
            scale_input = tvm.placeholder(_shape_scale, name="scale_input", dtype=scale_dtype)
            batch_mean_input = tvm.placeholder(_shape_scale, name="batch_mean_input", dtype=batch_mean_dtype)
            batch_variance_input = tvm.placeholder(_shape_scale,
                                                   name="batch_variance_input",
                                                   dtype=batch_variance_dtype)

            res = bn_training_reduce_grad_compute(grads_input,
                                                  x_input,
                                                  diff_scale_input,
                                                  diff_offset_input,
                                                  scale_input,
                                                  batch_mean_input,
                                                  batch_variance_input,
                                                  y,
                                                  epsilon,
                                                  kernel_name=kernel_name)

            tensor_list = [
                grads_input, x_input, diff_scale_input, diff_offset_input, scale_input, batch_mean_input,
                batch_variance_input, res
            ]
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
