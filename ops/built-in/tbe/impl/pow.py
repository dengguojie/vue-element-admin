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
pow
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm


def _less_compute(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = tbe.broadcast(tvm.const(2 ** (-126), dtype=dtype),
                                     shape, dtype)
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = tbe.vsub(input_y, input_x)
    res_min = tbe.vmin(res_sub, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62/62/2 to adaptor 126
    res_mul_val = tbe.vmuls(res_max, tvm.const(2 ** 62, dtype=dtype))
    res_mul = tbe.vmuls(res_mul_val, tvm.const(2 ** 62, dtype=dtype))
    res = tbe.vmuls(res_mul, tvm.const(2 ** 2, dtype=dtype))

    return res


def _less_compute_fp16(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = tbe.broadcast(tvm.const(2 ** (-24), dtype=dtype),
                                     shape, dtype)
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = tbe.vsub(input_y, input_x)
    res_min = tbe.vmin(res_sub, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    # max num of float32 is 2**24
    # but cce can only support 2**24, so use 12/12 to adaptor 24
    res_mul_val = tbe.vmuls(res_max, tvm.const(2 ** 12, dtype=dtype))
    res_mul = tbe.vmuls(res_mul_val, tvm.const(2 ** 12, dtype=dtype))
    res = tbe.vmuls(res_mul, tvm.const(1, dtype=dtype))

    return res


def _positive_compute(input_x, input_y):
    """
    compute result of pow when data_x is more than 0,
    use exp(y * ln(x)).
    """
    input_x = tbe.vabs(input_x)
    log_value = tbe.vlog(input_x)
    mul_value = tbe.vmul(input_y, log_value)
    res = tbe.vexp(mul_value)

    return res


def _negative_compute(input_x, input_y):
    """
    compute result of pow when data_x is less than 0,
    use [-2 * (|y| % 2) + 1] * exp(y * ln|x|)
    """
    dtype = input_x.dtype
    shape = input_x.shape
    abs_value = tbe.vabs(input_y)

    if not tbe_platform.api_check_support("te.lang.cce.vmod",
                                                   "float32"):
        dtype = "float16"
        abs_value = tbe.cast_to(abs_value, "float16")

    data_two = tbe.broadcast(tvm.const(2, dtype), shape, dtype)
    mod_value = tbe.vmod(abs_value, data_two)
    mul_value = tbe.vmuls(mod_value, tvm.const(-2, dtype))
    add_value = tbe.vadds(mul_value, tvm.const(1, dtype))

    if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        add_value = tbe.cast_to(add_value, "float32")

    abs_data_x = tbe.vabs(input_x)
    log_value = tbe.vlog(abs_data_x)
    mul_value = tbe.vmul(input_y, log_value)
    exp_value = tbe.vexp(mul_value)
    res = tbe.vmul(add_value, exp_value)

    return res


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("pow")
def pow_compute(input_x, input_y, output_z, kernel_name="pow"):
    """
    pow compute
    calculating data pow, res =x ^ y,
    x > 0: use exp(y*ln(x))
    x < 0: use [-2*(|y|%2)+1]*exp(y*ln|x|)
    x = 0: 0^0=1 & 0^y=0

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    res: TVM tensor
        the result of pow compute
    """
    input_dtype = input_x.dtype.lower()
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    shape_list = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                             param_name_input2="input_y")

    has_improve_precision = False
    data_x_cast = input_x
    data_y_cast = input_y
    cast_dtype = "float16"
    if tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        data_x_cast = tbe.cast_to(input_x, "float32")
        data_y_cast = tbe.cast_to(input_y, "float32")
        has_improve_precision = True
        cast_dtype = "float32"

    data_x = tbe.broadcast(data_x_cast, shape_list[2])
    data_y = tbe.broadcast(data_y_cast, shape_list[2])

    data_zero = tbe.broadcast(tvm.const(0, cast_dtype), shape_list[2], cast_dtype)
    if has_improve_precision:
        data_x_negative = _less_compute(data_x, data_zero)
    else:
        data_x_negative = _less_compute_fp16(data_x, data_zero)

    # compute result of pow when data_x is more than 0
    res_val_positive = _positive_compute(data_x, data_y)
    data_one = tbe.broadcast(tvm.const(1, cast_dtype), shape_list[2], cast_dtype)
    sub_one_val = tbe.vsub(data_x_negative, data_one)
    abs_val = tbe.vabs(sub_one_val)
    res_positive = tbe.vmul(res_val_positive, abs_val)

    # compute result of pow when data_x is less than 0
    res_val_negative = _negative_compute(data_x, data_y)
    res_negative = tbe.vmul(res_val_negative, data_x_negative)

    res = tbe.vadd(res_positive, res_negative)
    if input_dtype == "int32":
        res = tbe.round(res)
    else:
        res = tbe.cast_to(res, input_dtype)

    return res

# pylint: disable=locally-disabled,redefined-builtin
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def pow(input_x, input_y, output_z, kernel_name="pow"):
    """
    algorithm: pow
    calculating data pow, res =x ** y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    if len(shape_x) == 0:
        shape_x = (1,)
    if len(shape_y) == 0:
        shape_y = (1,)
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")
    shape_list = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                             param_name_input2="input_y")

    input_x_dtype = input_x.get("dtype").lower()
    input_y_dtype = input_y.get("dtype").lower()
    if input_x_dtype != input_y_dtype:
        raise RuntimeError("Dtype of input_x and input_y must be the same.")
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    para_check.check_dtype(input_x_dtype, check_list, param_name="input_x")

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=input_x_dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_y_dtype, name="data_y")
    res = pow_compute(data_x, data_y, output_z, kernel_name="pow")

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res],
              "bool_storage_as_1bit": False}
    tbe.cce_build_code(sch, config)
