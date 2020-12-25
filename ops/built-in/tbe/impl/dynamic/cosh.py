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
dynamic cosh
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
import te.lang.base as tbe_base
from te.utils import shape_util

# define a scaler , value = -1
SCALER_NEGATIVE_ONE = -1
# define a scaler , value = 0.5
SCALER_ZERO_POINT_FIVE = 0.5
# define a scaler , value = 2
SCALAR_TWO = 2


# pylint: disable=locally-disabled,unused-argument
def cosh_compute(input_x, output_cosh, kernel_name="cosh"):
    """
    algorithm: cosh
    calculating data's cosh, y = (e^(x)+e^(-x))/2

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_cosh: TVM tensor
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "cosh"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype
    shape = input_x.shape
    has_improve_precision = False
    if dtype != "float32" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    data_mul = tbe.vmuls(input_x, tvm.const(SCALER_NEGATIVE_ONE, dtype))
    data_exp = tbe.vexp(data_mul)
    data_exp_x = tbe.vmuls(data_exp, tvm.const(SCALER_ZERO_POINT_FIVE, dtype))

    tensor_two = tbe.broadcast(tvm.const(SCALAR_TWO, dtype), shape)
    data_ln2 = tbe.vlog(tensor_two)
    data_neg_ln2 = tbe.vmuls(data_ln2, tvm.const(SCALER_NEGATIVE_ONE, dtype))
    data_x = tbe.vadd(input_x, data_neg_ln2)
    data_exp_data = tbe.vexp(data_x)

    res = tbe.vadd(data_exp_x, data_exp_data)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@tbe_base.register_operator("Cosh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def cosh(input_x, output_cosh, kernel_name="cosh"):
    """
    algorithm: cosh
    calculating data's cosh, y = (e^(2x)+e^(-x))/2

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_cosh: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "cosh"

    Returns
    --------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)
    for (_input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([_input_x])
            fuseshape = [1]
            fuseshape[0] = functools.reduce(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype=input_dtype,
                                         name="data_input")
            res = cosh_compute(data_input, output_cosh, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
