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
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check


@tbe_platform.fusion_manager.fusion_manager.register("mish")
def mish_compute(input_x, output_y, kernel_name="mish"):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is mish

    Returns
    -------
    res : tvm.tensor
        the result of mish
    """
    dtype = input_x.dtype
    const_dtype = dtype
    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"
    exp_val = tbe.vexp(input_x)
    add_exp_val = tbe.vadds(exp_val, tvm.const(1, const_dtype))
    pow_var = tbe.vmul(add_exp_val, add_exp_val)
    add_val = tbe.vadds(pow_var, tvm.const(1, const_dtype))
    rec_val = tbe.vrec(add_val)
    mul_val = tbe.vmuls(rec_val, tvm.const(-2, dtype=const_dtype))
    add_val2 = tbe.vadds(mul_val, tvm.const(1, dtype=const_dtype))
    res = tbe.vmul(input_x, add_val2)
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def mish(input_x, output_y, kernel_name="mish"):
    """
    algorithm: mish
    calculating data's mish,y= x*(1 - 2/(1+(1+exp(x))^2))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is mish

    Returns
    -------
    None
    """

    input_shape = input_x.get("shape")
    input_format = input_x.get("format")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_shape(input_shape, param_name="input_x")
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    para_check.check_format(input_format)

    # fuse single axis
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, input_shape)

    data_x = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_x")
    res = mish_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, res]}
    tbe.cce_build_code(schedule, config)