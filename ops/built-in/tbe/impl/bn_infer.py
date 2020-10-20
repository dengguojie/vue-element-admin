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
bn_infer
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
tbe_platform.cce_policy.disableL2()


def _check_shape(shape_x, shape_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape

    Returns
    -------
    None
    """
    para_check.check_shape(shape_x, param_name="x")

    para_check.check_shape(shape_scale, param_name="scale")

    if len(shape_x) != 5 or len(shape_scale) != 5:
        raise RuntimeError(
            "The data format is 5HD, "
            "but some input's shape length is not 5")

    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal")


def _check_dtype(dtype_x, dtype_scale, dtype_offset,
                 dtype_mean, dtype_variance):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type
    dtype_mean: str
        mean's data type
    dtype_variance: str
        variance's data type

    Returns
    -------
    None
    """
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")
    para_check.check_dtype(dtype_scale.lower(), ("float32", "float16"), param_name="scale")
    para_check.check_dtype(dtype_offset.lower(), ("float32", "float16"), param_name="offset")
    para_check.check_dtype(dtype_mean.lower(), ("float32", "float16"), param_name="mean")
    para_check.check_dtype(dtype_variance.lower(), ("float32", "float16"), param_name="variance")


# pylint: disable=locally-disabled,invalid-name,too-many-arguments
# pylint: disable=locally-disabled,too-many-locals,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("bn_infer")
def bn_infer_compute(x, scale, offset, mean, variance,
                     y, epsilon, kernel_name="bn_inf"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: TVM tensor
        contains x data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    res: TVM tensor
        the result of bn_training_update_v2 compute for inference
    """
    shape_x = shape_util.shape_to_list(x.shape)

    # compute the oefficient of y
    multiplier_add = tbe.vadds(variance, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier = tbe.broadcast(multiplier_div, shape_x)

    addend_mul = tbe.vmul(multiplier_div, mean)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
        is_cast = True
        x = tbe.cast_to(x, "float32")

    res = tbe.vadd(tbe.vmul(multiplier, x), addend)
    if is_cast:
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_infer(x, scale, offset, mean, variance, y,
             epsilon, kernel_name="bn_infer"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    scale: dict
        dict of scale, A 5HD Tensor for scale.
    offset: dict
        dict of offset, A 5HD Tensor for offset.
    mean: dict
        dict of mean, A `Tensor`.
        dict of scale, A 5HD Tensor for mean.
    variance: dict
        dict of batch_variance, A `Tensor`.
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_infer"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_scale = scale.get("shape")

    dtype_x = x.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")

    data_format = x.get("format").upper()

    if data_format not in ("NC1HWC0",):
        format_rule = "Format only support 5HD"
        error_manager_vector.raise_err_check_params_rules("bn_infer", format_rule, "x", data_format)

    _check_shape(shape_x, shape_scale)
    shape_util.compare_tensor_dict_key(scale, offset, "shape")
    shape_util.compare_tensor_dict_key(scale, mean, "shape")
    shape_util.compare_tensor_dict_key(scale, variance, "shape")

    _check_dtype(dtype_x, dtype_scale, dtype_offset, dtype_mean, dtype_variance)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    scale_input = tvm.placeholder(shape_scale, name="scale_input", dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_scale, name="offset_input", dtype=dtype_offset.lower())
    mean_input = tvm.placeholder(shape_scale, name="mean_input", dtype=dtype_mean.lower())
    variance_input = tvm.placeholder(shape_scale, name="variance_input", dtype=dtype_variance.lower())

    res = bn_infer_compute(x_input, scale_input, offset_input, mean_input, variance_input,
                           y, epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = [x_input, scale_input, offset_input, mean_input, variance_input, res]

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
