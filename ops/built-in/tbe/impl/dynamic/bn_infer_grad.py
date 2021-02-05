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
bn_infer_grad
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=locally-disabled,too-many-arguments
@register_operator_compute("BNInferGrad", op_mode="dynamic", support_fusion=False)
def bn_infer_grad_compute(grads, scale, batch_variance, x_backprop,
                          epsilon, kernel_name="bn_infer_grad"):
    """
    Compute for bn_infer_grad_compute
    x_norm:(x-input_reserve_space_1)*
            np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_scale:np.sum(y*(x-input_reserve_space_1)*
                         np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_offset: np.sum(y)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads. Must be one of the following
        type: `float16`, `float32`.
    scale: TVM tensor 5D
        the placeholder of x. Must be one of the following
        type: `float32`.
    batch_variance: TVM tensor 5D
        the placeholder of batch_variance. Must be one of the following
        type: `float32`.
    x_backprop: dict
        dict of x_norm, A 5D Tensor for output x_norm.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_infer_grad"

    Returns
    -------
    res: x_backprop
   """
    shape_x = shape_util.shape_to_list(grads.shape)

    is_cast = False
    if grads.dtype == "float16" and \
           tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        is_cast = True
        grads = tbe.cast_to(grads, "float32")

    data_adds = tbe.vadds(batch_variance, epsilon)
    data_rsqrt = tbe.vsqrt(data_adds)
    shape_var = shape_util.shape_to_list(batch_variance.shape)
    data_cast = tbe.broadcast(tvm.const(1, "float32"), shape_var)
    data_rsqrts = tbe.vdiv(data_cast, data_rsqrt)

    scale_mul = tbe.vmul(scale, data_rsqrts)
    scale_mul_broadcast = tbe.broadcast(scale_mul, shape_x)
    res = tbe.vmul(scale_mul_broadcast, grads)
    if is_cast:
        res = tbe.cast_to(res, "float16")

    return res


def _check_shape(shape_grads, shape_batch_variance):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_grads: list or tuple
        input grads's data shape
    shape_batch_variance: list or tuple
        input batch_variance's data shape
    Returns
    -------
    None
    """
    para_check.check_shape(shape_grads, param_name="grads")

    para_check.check_shape(shape_batch_variance, param_name="batch_variance")

    dim_c1 = shape_grads[1]
    dim_c0 = shape_grads[4]

    if len(shape_grads) != 5:
        error_detail = "This operator can only support 5D"
        error_manager_vector.raise_err_input_shape_invalid("bn_infer_grad", "grads", error_detail)
    if dim_c0 != 16:
        error_detail = "shape_grads last dim must be 16"
        error_manager_vector.raise_err_input_shape_invalid("bn_infer_grad", "grads", error_detail)

    if len(shape_batch_variance) != 5:
        error_detail = "This operator can only support 5D"
        error_manager_vector.raise_err_input_shape_invalid("bn_infer_grad", "batch_variance", error_detail)

    if shape_batch_variance[0] != 1 or shape_batch_variance[2] != 1 \
       or shape_batch_variance[3] != 1:
        error_detail = "Dimensions except Dimension C must be one for shape_batch_mean"
        error_manager_vector.raise_err_input_shape_invalid("bn_infer_grad", "batch_variance", error_detail)

    if shape_batch_variance[1] != dim_c1 or shape_batch_variance[4] != dim_c0:
        batch_variance_rule = "Dimension C of grads and batch_variance must be equal"
        error_manager_vector.raise_err_check_params_rules("bn_infer_grad", batch_variance_rule, "batch_variance",
                                                          shape_batch_variance[1] * shape_batch_variance[4])


@register_operator("BNInferGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_infer_grad(grads, scale, batch_variance,
                  x_backprop, epsilon=0.0001,
                  kernel_name="bn_infer_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_infer_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    scale: dict
        dict of scale, A 5D Tensor for input scale.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
    x_backprop: dict
        dict of x_backprop, A 5D Tensor for output x_backprop.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_infer_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_scale = scale.get("shape")
    shape_batch_variance = batch_variance.get("shape")

    input_grads_dtype = grads.get("dtype").lower()
    input_scale_dtype = scale.get("dtype").lower()
    batch_variance_dtype = batch_variance.get("dtype").lower()

    para_check.check_dtype(input_grads_dtype, ("float32", "float16"), param_name="grads")
    para_check.check_dtype(input_scale_dtype, ("float32",), param_name="scale")
    para_check.check_dtype(batch_variance_dtype, ("float32",), param_name="batch_variance")

    ins = classify([grads, scale, batch_variance], Mode.ELEWISE)
    schedules, tensors = [], []
    for (_grads, _scale, _batch_variance) in ins:
        with tbe_base.compute():
            shape_grads, shape_scale, shape_batch_variance = shape_util.variable_shape([_grads, 
                                                                                        _scale,
                                                                                        _batch_variance])
            grads_input = tvm.placeholder(shape_grads, name="grads_input",
                                          dtype=input_grads_dtype)
            scale_input = tvm.placeholder(shape_scale, name="x_input",
                                          dtype=input_scale_dtype)
            batch_variance_input = tvm.placeholder(shape_batch_variance,
                                                   name="batch_variance_input",
                                                   dtype=batch_variance_dtype)

            res = bn_infer_grad_compute(grads_input, scale_input,
                                        batch_variance_input,
                                        x_backprop, epsilon,
                                        kernel_name=kernel_name)
            tensors.append([grads_input, scale_input, batch_variance_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
