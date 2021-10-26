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
in_training_update_v2
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-arguments
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-statements,too-many-locals
def op_select_format(x,
                     sum,
                     square_sum,
                     gamma,
                     beta,
                     mean,
                     variance,
                     y,
                     batch_mean,
                     batch_variance,
                     momentum,
                     epsilon,
                     kernel_name="in_training_update_v2"):
    """
    select format dynamically
    """
    input_format = "NC1HWC0, NC1HWC0"
    ori_format = x.get("ori_format")
    if ori_format in ("NDHWC", "NCDHW"):
        input_format = "NDC1HWC0, NDC1HWC0"

    input0 = gen_param(classify="input0",
                       name="x",
                       datatype="float16,float",
                       format=input_format)
    input1 = gen_param(classify="input1",
                       name="sum",
                       datatype="float,float",
                       format=input_format)
    input2 = gen_param(classify="input2",
                       name="square_sum",
                       datatype="float,float",
                       format=input_format)
    input3 = gen_param(classify="input3",
                       name="gamma",
                       datatype="float,float",
                       format=input_format)
    input4 = gen_param(classify="input4",
                       name="beta",
                       datatype="float,float",
                       format=input_format)
    input5 = gen_param(classify="input5",
                       name="mean",
                       datatype="float,float",
                       format=input_format)
    input6 = gen_param(classify="input6",
                       name="variance",
                       datatype="float,float",
                       format=input_format)
    output0 = gen_param(classify="output0",
                        name="y",
                        datatype="float16,float",
                        format=input_format)
    output1 = gen_param(classify="output1",
                        name="batch_mean",
                        datatype="float,float",
                        format=input_format)
    output2 = gen_param(classify="output2",
                        name="batch_variance",
                        datatype="float,float",
                        format=input_format)

    param_list = [input0, input1, input2, input3, input4, input5, input6, output0, output1, output2]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def in_training_update_compute(x,
                               sum,
                               square_sum,
                               gamma,
                               beta,
                               mean,
                               variance,
                               y,
                               mean_out,
                               variance_out,
                               momentum,
                               epsilon,
                               format_x,
                               kernel_name="in_training_update_v2"):
    """
    DSL description of the instancenorm operator's mathematical calculation process

    x: dict
        the placeholder of input x
    sum: dict
        the placeholder of input sum
    square_sum: dict
        the placeholder of input square_sum
    gamma: dict
        the placeholder of input gamma
    beta: dict
        the placeholder of input beta
    mean: dict
        the placeholder of input mean
    variance: dict
        the placeholder of input variance
    y: dict
        shape and dtype of output y
    batch_mean: dict
        shape and dtype of output batch_mean
    batch_variance: dict
        shape and dtype of output batch_variance
    momentum: float
        A ratio to calculate the update mean or variance
    epsilon: float
        A small float number added to the variance of x
    kernel_name: str
        cce kernel name, default value is "in_training_update_v2"

    Returns
    -------
    res_list: list
        [result, result_mean, result_variance]
    """
    shape_x = shape_util.shape_to_list(x.shape)
    dtype_x = x.dtype.lower()

    # compute the instance normalization of x
    if dtype_x == "float16":
        x = tbe.cast_to(x, "float32")

    if format_x in ("NDC1HWC0",):  # only support NDC1HWC0 and NC1HWC0
        num = shape_x[1] * shape_x[3]
    else:
        num = shape_x[2] * shape_x[3]

    num_rec = 1.0 / num
    # compute the saved mean of x
    compute_mean = tbe.vmuls(sum, num_rec)
    mean_boardcast = tbe.broadcast(compute_mean, shape_x)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(compute_mean, compute_mean)
    compute_var = tbe.vsub(variance_div, variance_square)

    x_mean = tbe.vsub(x, mean_boardcast)
    multiplier_add = tbe.vadds(compute_var, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    sqrt_boardcast = tbe.broadcast(multiplier_sqrt, shape_x)
    mean_wo_scale = tbe.vdiv(x_mean, sqrt_boardcast)
    result = mean_wo_scale
    if gamma is not None and beta is not None:
        gamma = tbe.broadcast(gamma, shape_x)
        beta = tbe.broadcast(beta, shape_x)
        gamma_scale = tbe.vmul(result, gamma)
        result = tbe.vadd(gamma_scale, beta)

    if dtype_x == "float16":
        result = tbe.cast_to(result, "float16")

    if num == 1:
        batch_var_scalar = 0.0
    else:
        batch_var_scalar = float(num) / (num - 1)

    result_mean = compute_mean
    result_variance = tbe.vmuls(compute_var, batch_var_scalar)

    # if input mean and var, use input values and momentum to update
    if mean is not None and variance is not None:
        factor_reverse = 1.0 - momentum
        mean_mul = tbe.vmuls(compute_mean, momentum)
        mean_mul_rev = tbe.vmuls(mean, factor_reverse)
        result_mean = tbe.vadd(mean_mul, mean_mul_rev)

        var_mul = tbe.vmuls(result_variance, momentum)
        var_mul_rev = tbe.vmuls(variance, factor_reverse)
        result_variance = tbe.vadd(var_mul, var_mul_rev)

    return [result, result_mean, result_variance]


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def in_training_update_v2(x,
                          sum,
                          square_sum,
                          gamma,
                          beta,
                          mean,
                          variance,
                          y,
                          batch_mean,
                          batch_variance,
                          momentum=0.1,
                          epsilon=0.00001,
                          kernel_name="in_training_update_v2"):
    """
    instancenorm operator interface implementation

    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16, float32
    sum: dict
        shape and dtype of input sum, only support float32
    square_sum: dict
        shape and dtype of input square_sum, only support float32
    gamma: dict
        shape and dtype of input gamma, only support float32
    beta: dict
        shape and dtype of input beta, only support float32
    mean: dict
        shape and dtype of input mean, only support float32
    variance: dict
        shape and dtype of input variance, only support float32
    y: dict
        shape and dtype of output y, only support float16, float32
    batch_mean: dict
        shape and dtype of output batch_mean, only support float32
    batch_variance: dict
        shape and dtype of output batch_variance, only support float32
    momentum: float
        A ratio to calculate the update mean or variance
    epsilon: float
        A small float number added to the variance of x
    kernel_name: str
        cce kernel name, default value is "in_training_update_v2"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    format_x = x.get("format")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    para_check.check_shape(shape_sum, param_name="sum")
    para_check.check_shape(shape_square_sum, param_name="square_sum")

    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    para_check.check_dtype(dtype_sum.lower(), ("float32",), param_name="sum")
    para_check.check_dtype(dtype_square_sum.lower(), ("float32",), param_name="square_sum")

    if format_x in ("NDC1HWC0",):
        shape_x = [shape_x[0], shape_x[1], shape_x[2], shape_x[3] * shape_x[4], shape_x[5]]
        shape_sum = [shape_sum[0], shape_sum[1], shape_sum[2], shape_sum[3] * shape_sum[4], shape_sum[5]]
        shape_square_sum = [
            shape_square_sum[0], shape_square_sum[1], shape_square_sum[2], shape_square_sum[3] * shape_square_sum[4],
            shape_square_sum[5]
        ]

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input", dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum, name="square_sum_input", dtype=dtype_square_sum.lower())
    gamma_input, beta_input, mean_input, var_input = None, None, None, None

    scale = False
    if gamma is not None and beta is not None:
        scale = True
        shape_gamma = gamma.get("shape")
        shape_beta = beta.get("shape")
        para_check.check_shape(shape_gamma, param_name="gamma")
        para_check.check_shape(shape_beta, param_name="beta")
        dtype_gamma = gamma.get("dtype")
        dtype_beta = beta.get("dtype")
        para_check.check_dtype(dtype_gamma.lower(), ("float32",), param_name="gamma")
        para_check.check_dtype(dtype_beta.lower(), ("float32",), param_name="beta")

        if format_x in ("NDC1HWC0",):
            shape_gamma = [
                shape_gamma[0], shape_gamma[1], shape_gamma[2], shape_gamma[3] * shape_gamma[4], shape_gamma[5]
            ]
            shape_beta = [shape_beta[0], shape_beta[1], shape_beta[2], shape_beta[3] * shape_beta[4], shape_beta[5]]

        gamma_input = tvm.placeholder(shape_gamma, name="gamma_input", dtype=dtype_gamma.lower())
        beta_input = tvm.placeholder(shape_beta, name="beta_input", dtype=dtype_beta.lower())

    use_mean = False
    if mean is not None and variance is not None:
        use_mean = True
        shape_mean = mean.get("shape")
        shape_var = variance.get("shape")
        para_check.check_shape(shape_mean, param_name="mean")
        para_check.check_shape(shape_var, param_name="variance")
        dtype_mean = mean.get("dtype")
        dtype_var = variance.get("dtype")
        para_check.check_dtype(dtype_mean.lower(), ("float32",), param_name="mean")
        para_check.check_dtype(dtype_var.lower(), ("float32",), param_name="variance")

        if format_x in ("NDC1HWC0",):
            shape_mean = [shape_mean[0], shape_mean[1], shape_mean[2], shape_mean[3] * shape_mean[4], shape_mean[5]]
            shape_var = [shape_var[0], shape_var[1], shape_var[2], shape_var[3] * shape_var[4], shape_var[5]]

        mean_input = tvm.placeholder(shape_mean, name="mean_input", dtype=dtype_mean.lower())
        var_input = tvm.placeholder(shape_var, name="variance_input", dtype=dtype_var.lower())

    res = in_training_update_compute(x_input,
                                     sum_input,
                                     square_sum_input,
                                     gamma_input,
                                     beta_input,
                                     mean_input,
                                     var_input,
                                     y,
                                     batch_mean,
                                     batch_variance,
                                     momentum,
                                     epsilon,
                                     format_x,
                                     kernel_name=kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    if use_mean:
        if scale:
            tensor_list = [x_input, sum_input, square_sum_input, gamma_input, beta_input, mean_input, var_input
                          ] + list(res)
        else:
            tensor_list = [x_input, sum_input, square_sum_input, mean_input, var_input] + list(res)
    else:
        if scale:
            tensor_list = [x_input, sum_input, square_sum_input, gamma_input, beta_input] + list(res)
        else:
            tensor_list = [x_input, sum_input, square_sum_input] + list(res)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.build(sch, config)
