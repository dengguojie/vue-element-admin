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
expm1
"""
import functools
from functools import reduce as reduceIns

import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
import te.lang.base as tbe_base
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import shape_util
from te.utils import para_check


# define a scalar , value = -1
SCALAR_NEGATIVE_ONE = -1.0
# define taylor negative threshold , value = -0.7
TAYLOR_NEGATIVE_THRESHOLD = -0.7
# define taylor positive threshold , value = 1.7
TAYLOR_POSITIVE_THRESHOLD = 1.7
# define second order parameter , value = 1 / 2.0
TAYLOR_SECOND_ORDER_PARAM = 1 / 2.0
# define third order parameter , value = 1 / 6.0
TAYLOR_THIRD_ORDER_PARAM = 1 / 6.0
# define fourth order parameter , value = 1 / 24.0
TAYLOR_FOURTH_ORDER_PARAM = 1 / 24.0
# define fifth order parameter , value = 1 / 120.0
TAYLOR_FIFTH_ORDER_PARAM = 1 / 120.0
# define sixth order parameter , value = 1 / 720.0
TAYLOR_SIXTH_ORDER_PARAM = 1 / 720.0
# define seventh order parameter , value = 1 / 5040.0
TAYLOR_SEVENTH_ORDER_PARAM = 1 / 5040.0


# pylint: disable=locally-disabled,too-many-locals
def _expm1_taylor_compute(input_x):
    """
    Calculate e^x - 1, Use seventh order taylor expansion
    e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!) + \
          (x^6 / 6!) + (x^7 / 7!)
    e^x - 1 = x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!) + \
            (x^6 / 6!) + (x^7 / 7!)

    Parameters:
    ----------
    input_x : TVM tensor, the placeholder of input_x

    Returns : A Tensor. Has the same type as input_x.
    -------
    """
    # calculate second order tayloy section : x^2 / 2!
    taylor_second_order_param = tvm.const(TAYLOR_SECOND_ORDER_PARAM, "float32")
    data_power_2 = tbe.vmul(input_x, input_x)
    data_power_2_div_2 = tbe.vmuls(data_power_2,
                                           taylor_second_order_param)

    # calculate third order tayloy section : x^3 / 3!
    taylor_third_order_param = tvm.const(TAYLOR_THIRD_ORDER_PARAM, "float32")
    data_power_3 = tbe.vmul(data_power_2, input_x)
    data_power_3_div_6 = tbe.vmuls(data_power_3,
                                           taylor_third_order_param)

    # calculate fourth order tayloy section : x^4 / 4!
    taylor_fourth_order_param = tvm.const(TAYLOR_FOURTH_ORDER_PARAM, "float32")
    data_power_4 = tbe.vmul(data_power_3, input_x)
    data_power_4_div_24 = tbe.vmuls(data_power_4,
                                            taylor_fourth_order_param)

    # calculate fifth order tayloy section : x^5 / 5!
    taylor_fifth_order_param = tvm.const(TAYLOR_FIFTH_ORDER_PARAM, "float32")
    data_power_5 = tbe.vmul(data_power_4, input_x)
    data_power_5_div_120 = tbe.vmuls(data_power_5,
                                             taylor_fifth_order_param)

    # xcalculate sixth order tayloy section : ^6 / 6!
    taylor_sixth_order_param = tvm.const(TAYLOR_SIXTH_ORDER_PARAM, "float32")
    data_power_6 = tbe.vmul(data_power_5, input_x)
    data_power_6_div_720 = tbe.vmuls(data_power_6,
                                             taylor_sixth_order_param)

    # calculate seventh order tayloy section : x^7 / 7!
    taylor_seventh_order_param = tvm.const(TAYLOR_SEVENTH_ORDER_PARAM,
                                           "float32")
    data_power_7 = tbe.vmul(data_power_6, input_x)
    data_power_7_div_5040 = tbe.vmuls(data_power_7,
                                              taylor_seventh_order_param)

    res_second_taylor = tbe.vadd(input_x, data_power_2_div_2)
    res_third_taylor = tbe.vadd(res_second_taylor, data_power_3_div_6)
    res_fourth_taylor = tbe.vadd(res_third_taylor, data_power_4_div_24)
    res_fifth_taylor = tbe.vadd(res_fourth_taylor, data_power_5_div_120)
    res_sixth_taylor = tbe.vadd(res_fifth_taylor, data_power_6_div_720)
    res = tbe.vadd(res_sixth_taylor, data_power_7_div_5040)

    return res


def _expm1_mini_compute(mini_res, input_x, shape):
    """
    do element-wise e^x - 1 compute in mini scene
    f(x) = e^x - 1,
    x <= TAYLOR_NEGATIVE_THRESHOLD or x >= TAYLOR_POSITIVE_THRESHOLD
    f(x) = seventh taylor computer,
    TAYLOR_NEGATIVE_THRESHOLD < x < TAYLOR_POSITIVE_THRESHOLD

    Parameters:
    ----------
    mini_res: TVM tensor, the tensor of e^x - 1

    input_x : TVM tensor, the placeholder of input data

    shape : tuple, the shape of input data

    Returns : A Tensor. Has the same type as mini_res.
    -------
    """
    taylor_res = _expm1_taylor_compute(input_x)

    input_right_border = tvm.const(TAYLOR_POSITIVE_THRESHOLD, "float32")
    tensor_input_right_border = tbe.broadcast(input_right_border, shape)

    input_left_border = tvm.const(TAYLOR_NEGATIVE_THRESHOLD, "float32")
    tensor_input_left_border = tbe.broadcast(input_left_border, shape)

    b_gt_left_border = tbe.vcmp(input_x, tensor_input_left_border, 'gt')
    exp_taylor_neg = tbe.vsel(b_gt_left_border, taylor_res, mini_res)

    b_lt_right_border = tbe.vcmp(input_x, tensor_input_right_border, 'lt')
    mini_res = tbe.vsel(b_lt_right_border, exp_taylor_neg, mini_res)

    return mini_res

# pylint: disable=locally-disabled,too-many-locals,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("expm1")
def expm1_compute(input_x, output_y, kernel_name="expm1"):
    """
    algorithm: expm1
    calculating data's expm1, y = (e^x) - 1
    in cloud scene, for all inputs :
    f(x) = e^x - 1,
    in mini scene :
    f(x) = e^x - 1,
    x <= TAYLOR_NEGATIVE_THRESHOLD or x >= TAYLOR_POSITIVE_THRESHOLD
    f(x) = seventh taylor computer,
    TAYLOR_NEGATIVE_THRESHOLD < x < TAYLOR_POSITIVE_THRESHOLD

    Parameters
    ----------
    input_x : TVM tensor, the placeholders of input data
    output_y : dict, shape and dtype of output, should be same shape
               and type as input
    kernel_name : str, kernel name, default value is "expm1"

    Returns
    -------
    res : the result of compute
    """
    dtype = input_x.dtype
    shape = input_x.shape
    flag_cloud = tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32")
    flag_mini = tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32")
    if dtype.lower() == "float16" and flag_cloud:
        input_x = tbe.cast_to(input_x, "float32")

    scalar_negative_one = tvm.const(SCALAR_NEGATIVE_ONE, "float32")
    exp_res = tbe.vexp(input_x)
    res = tbe.vadds(exp_res, scalar_negative_one)

    if (not flag_cloud) and flag_mini:
        input_x = tbe.cast_to(input_x, "float32")
        res = _expm1_mini_compute(res, input_x, shape)

    if dtype.lower() == "float16" and (flag_cloud or flag_mini):
        res = tbe.cast_to(res, dtype)

    return res


@tbe_base.register_operator("Expm1")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def expm1(input_x, output_y, kernel_name="expm1"):
    """
    algorithm: expm1
    calculating data's expm1, y = (e^x) - 1

    Parameters
    ----------
    input_x : dict,shape and dtype of input, only support float16, float32
    output_y: dict,shape and dtype of output, should be same shape
              and type as input
    kernel_name : str, kernel name, default value is "expm1"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)

    for (input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_input")
            res = expm1_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name":kernel_name,
        "tensor_list":tensors,
        "bool_storage_as_list":False
    }

    tbe.build(schedules, config)