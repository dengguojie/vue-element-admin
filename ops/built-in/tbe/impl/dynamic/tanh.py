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
tanh
"""
import te
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te import tvm
from topi import generic
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from functools import reduce as reduceIns
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def tanh_compute(input_x, output_y, kernel_name="tanh"):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

        For x > 0, to avoid overflow in exp(x), we reformulate the above

          (exp(2x) - 1) / (exp(2x) + 1)
        = (1 - exp(-2x)) / (1 + exp(-2x))

        = sign(x)* ((1 - exp(-2x)) / (1 + exp(-2x)))
        = (x / abs(x)) * ((1 - exp(-2x)) / (1 + exp(-2x)))

        avoid value divide by zero, so abs(x) -> (abs(x) + min_value)


    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype
    # positive min float32 value
    MIN_FP_DATA = 2 ** (-126)
    CONST_DTYPE = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        MIN_FP_DATA = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        CONST_DTYPE = "float32"

    input_abs = tbe.vabs(input_x)
    power_val = tbe.vmuls(input_abs, tvm.const(-2, CONST_DTYPE))

    if input_dtype == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        power_val = tbe.cast_to(power_val, "float16")

    exp_val = tbe.vexp(power_val)

    if input_dtype == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        exp_val = tbe.cast_to(exp_val, "float32")

    up_val_tmp = tbe.vmul(exp_val, input_x)
    up_val = tbe.vsub(input_x, up_val_tmp)

    input_x_tmp = tbe.vadds(input_abs, MIN_FP_DATA)
    down_val_tmp = tbe.vadds(exp_val, tvm.const(1, CONST_DTYPE))
    down_val = tbe.vmul(down_val_tmp, input_x_tmp)

    res = tbe.vdiv(up_val, down_val)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@tbe_base.register_operator("Tanh")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def tanh(input_x, output_y, kernel_name="tanh"):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is tanh

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()

    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="input_x")

    ins = classify([input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe_base.compute():
            shape_x = variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, shape_x[0])
            data_input = tvm.placeholder(fuseshape, name="data_input",
                                         dtype=input_dtype)
            res = tanh_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
