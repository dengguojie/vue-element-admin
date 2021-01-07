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
softplus_v2
"""
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
import te.platform as tbe_platform

# log(2), the recalculated part
NEG_LN_2 = - 0.69314718055994530941723212145818


# pylint: disable=too-many-locals,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("softplus_v2")
def softplus_v2_compute(input_features, beta, threshold, kernel_name="softplus_v2"):
    """
    calculating data

    Parameters
    ----------
    input_features : TVM tensor
       the placeholder of input_features
    beta : float
       scale factor, default value is 1.0
    threshold: float
       boundary value, default value is 20.0
    kernel_name : str
       kernel name, default value is "softplus_v2_grad"

    Returns
    -------
    output tensor
    """
    input_dtype = input_features.dtype
    input_shape = input_features.shape

    if input_dtype != "float32":
        input_features = tbe.cast_to(input_features, "float32")

    beta_const_tensor = tbe.broadcast(tvm.const(beta, dtype="float32"), input_shape)
    one_const_tensor = tbe.broadcast(tvm.const(1.0, dtype="float32"), input_shape)
    # calculate log(1+exp(beta*x))/beta
    beta_mul_x = tbe.vmul(input_features, beta_const_tensor)
    positive_part = tbe.vmaxs(beta_mul_x, tvm.const(0, dtype="float32"))
    negative_part = tbe.vmins(beta_mul_x, tvm.const(0, dtype="float32"))
    # calculate negative part softplus
    neg_to_pos = tbe.vmuls(negative_part, tvm.const(-1, dtype="float32"))
    exp_neg = tbe.vexp(neg_to_pos)
    exp_add_one = tbe.vadd(exp_neg, one_const_tensor)
    log_neg = tbe.vlog(exp_add_one)
    res_neg = tbe.vadd(log_neg, negative_part)
    # calculate positive part softplus
    exp_pos = tbe.vexp(positive_part)
    add_one = tbe.vadd(exp_pos, one_const_tensor)
    res_pos = tbe.vlog(add_one)

    log_res = tbe.vadd(res_neg, res_pos)
    res_tmp = tbe.vadds(log_res, NEG_LN_2)
    method_one_res = tbe.vdiv(res_tmp, beta_const_tensor)

    # combine two results
    left_cmp = tbe.vmul(input_features, beta_const_tensor)
    is_method_two = tbe.vcmpsel(left_cmp, tvm.const(threshold, dtype="float32"), 'gt', 1.0, 0.0)
    is_method_one = tbe.vsub(one_const_tensor, is_method_two)

    method_one_get_res = tbe.vmul(method_one_res, is_method_one)
    method_two_get_res = tbe.vmul(input_features, is_method_two)

    res_tmp = tbe.vadd(method_one_get_res, method_two_get_res)
    if input_dtype == "float16":
        res = tbe.cast_to(res_tmp, "float16")
    else:
        res = res_tmp

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def softplus_v2(x, y, beta=1.0, threshold=20.0, kernel_name="softplus_v2"):
    """
    Computes softplus operation with attribute beta and threshold.
    The output: log(1+exp(beta*x))/beta if x/beta <= threshold else x.

    Parameters
    ----------
    x: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32".
    y: dict
        data of output.
    beta: float16/float32, option, default:1.0
    threshold: float16/float32, option, default:20.0

    kernel_name: str
        kernel name, default value is "softplus_v2".
    Returns
    -------
    None
    """
    shape_feature = x.get("shape")
    dtype_feature = x.get("dtype")
    # check dtype and shape
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_feature, check_list, param_name="x")
    para_check.check_shape(shape_feature, param_name="x")

    if beta == 0.0:
        raise ZeroDivisionError("The value of beta must be non-zero.")

    data_features = tvm.placeholder(shape_feature, dtype=dtype_feature, name="data_features")

    res = softplus_v2_compute(data_features, beta, threshold, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_features, res]}
    tbe.cce_build_code(schedule, config)
