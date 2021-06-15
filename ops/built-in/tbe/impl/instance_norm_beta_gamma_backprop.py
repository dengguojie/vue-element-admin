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
instance_norm_grad
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import para_check


# pylint: disable=too-many-locals,too-many-arguments,unused-argument,invalid-name
def instance_norm_beta_gamma_backprop_compute(data_dy,
                                              data_res_gamma,
                                              pd_gamma,
                                              pd_beta,
                                              format_dy,
                                              kernel_name="instance_norm_beta_gamma_backprop"):
    """
    DSL description of the layernorm_grad operator's mathematical

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    res_for_gamma: dict
        shape and dtype of input res_for_gamma, only support float32
    pd_gamma: dict
        shape and dtype of output pd_gamma, only support float16, float32
    pd_beta: dict
        shape and dtype of output pd_beta, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "instance_norm_beta_gamma_backprop"

    Returns
    -------
    res_tuple: tuple
        (pd_gamma, pd_beta)
    """
    dtype_dy = data_dy.dtype.lower()
    param_axis = []
    if format_dy == "NDC1HWC0":  # only support NDC1HWC0 and NC1HWC0
        param_axis = [0, 1, 3, 4]
    else:
        param_axis = [0, 2, 3]

    if dtype_dy == "float16":
        data_dy = tbe.cast_to(data_dy, "float32")
        data_res_gamma = tbe.cast_to(data_res_gamma, "float32")

    data_mul = tbe.vmul(data_res_gamma, data_dy)
    pd_gamma, pd_beta = tuple_sum([data_mul, data_dy], param_axis, keepdims=True)

    if dtype_dy == "float16":
        pd_gamma = tbe.cast_to(pd_gamma, dtype_dy)
        pd_beta = tbe.cast_to(pd_beta, dtype_dy)

    res_list = [pd_gamma, pd_beta]

    return res_list


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def instance_norm_beta_gamma_backprop(dy,
                                      res_for_gamma,
                                      pd_gamma,
                                      pd_beta,
                                      kernel_name="instance_norm_beta_gamma_backprop"):
    """
    instancenormbetagammabackprop operator interface implementation

    Parameters
    ----------
    dy: dict
        shape and dtype of input dy, only support float16, float32
    res_for_gamma: dict
        shape and dtype of input res_for_gamma, only support float32
    pd_gamma: dict
        shape and dtype of output pd_gamma, only support float16, float32
    pd_beta: dict
        shape and dtype of output pd_beta, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "instance_norm_beta_gamma_backprop"

    Returns
    -------
    None
    """
    shape_dy = dy.get("shape")
    shape_res_gamma = res_for_gamma.get("shape")
    dtype_dy = dy.get("dtype").lower()

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    para_check.check_shape(shape_dy, param_name="dy")
    para_check.check_shape(shape_res_gamma, param_name="res_for_gamma")

    data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype_dy)
    data_res_gamma = tvm.placeholder(shape_res_gamma, name="res_for_gamma", dtype=dtype_dy)

    format_dy = dy.get("format")
    res = instance_norm_beta_gamma_backprop_compute(data_dy, data_res_gamma, pd_gamma, pd_beta, format_dy, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_dy, data_res_gamma] + list(res)}

    tbe.build(sch, config)
