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
dynamic floordiv_add_sqrt_mul_relu
"""
import tbe
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
import te
from te import platform as tbe_platform
from te import tvm
from impl.dynamic.floor_div import floor_div_compute
from impl.dynamic.add import add_compute
from impl.dynamic.sqrt import sqrt_compute
from impl.dynamic.mul import mul_compute
from impl.dynamic.relu import relu_compute
from impl.util.platform_adapter import OpPatternMode

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
SIZE_SIXTEEN = 16
CONST_ZERO = 0


def broadcast_shape(shape_0, shape_1, shape_2, shape_3):
    shape_max = []
    for _shape_0, _shape_1, _shape_2, _shape_3 in zip(shape_0, shape_1, shape_2, shape_3):
        shape_max.append(tvm.max(_shape_0, _shape_1, _shape_2, _shape_3))
    return shape_max


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def floordiv_add_sqrt_mul_relu_compute(data_0, data_1, data_2, data_3, output,
                                       kernel_name="floordiv_add_sqrt_mul_relu"):
    """
    algorithm: floordiv_add_sqrt_mul_relu
    calculating:
    c = floordiv(data_0,data_1)
    e = add(c,data_2)
    f = sqrt(e)
    h = mul(f,data_3)
    i = relu(h)
    Parameters
    ----------
    data_0 : dict
        including shape, dtype and range, only support float16, float32, int32
    data_1 : dict
        including shape, dtype and range, only support float16, float32, int32
    data_2 : dict
        including shape, dtype and range, only support float16, float32, int32
    data_3 : dict
        including shape, dtype and range, only support float16, float32, int32
    output: dict
        including shape, dtype and range, only support float16, float32, int32
    kernel_name : str
       cce kernel name, default value is floordiv_add_sqrt_mul_relu

    Returns
    -------
    output of the op
    """
    # shape_0 = shape_util.shape_to_list(data_0.shape)
    # shape_1 = shape_util.shape_to_list(data_1.shape)
    # shape_2 = shape_util.shape_to_list(data_2.shape)
    # shape_3 = shape_util.shape_to_list(data_3.shape)
    #
    # shape_max = broadcast_shape(shape_0, shape_1, shape_2, shape_3)
    # data_0 = tbe.broadcast(data_0, shape_max)
    # data_1 = tbe.broadcast(data_1, shape_max)
    # data_2 = tbe.broadcast(data_2, shape_max)
    # data_3 = tbe.broadcast(data_3, shape_max)

    res_floor_div = floor_div_compute(data_0, data_1, output)
    res_add = add_compute(res_floor_div, data_2, output)
    res_sqrt = sqrt_compute(res_add, output)
    res_mul = mul_compute(res_sqrt, data_3, output)
    res_relu = relu_compute(res_mul, output)
    return res_relu

    # # floordiv  # if dtype_0 != "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):  #     data_0 = tbe.cast_to(data_0, 'float32')  #     data_1 = tbe.cast_to(data_1, 'float32')  #  # div_res = tbe.vdiv(data_0, data_1)  # if dtype_0 != "float16" and tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") == "Ascend310":  #     div_res = tbe.cast_to(div_res, "float16")  #  # div_res = tbe.floor(div_res)  # div_res = tbe.cast_to(div_res, dtype_0)  #  # # add  # add_res = tbe.vadd(div_res, data_2)  #  # # sqrt  # has_improve_precision = False  # if dtype_0 == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt", "float32"):  #     add_res = tbe.cast_to(add_res, "float32")  #     has_improve_precision = True  # sqrt_res = tbe.vsqrt(add_res)  #  # if has_improve_precision:  #     sqrt_res = tbe.cast_to(sqrt_res, "float16")  #  # # mul  # mul_res = tbe.vmul(sqrt_res, data_3)  #  # # relu  # compatible_dtype = dtype_0  # if dtype_0 == "int8" and tbe_platform.cce_conf.api_check_support('te.lang.cce.cast_to', 's82f16'):  #     mul_res = tbe.cast_to(mul_res, "float16")  #     compatible_dtype = "float16"  # if tbe_platform.cce_conf.api_check_support('te.lang.cce.vrelu', compatible_dtype):  #     relu_res = tbe.vrelu(mul_res)  # else:  #     tensor_zero = tbe.broadcast(tvm.const(CONST_ZERO, compatible_dtype), shape_max)  #     relu_res = tbe.vmax(mul_res, tensor_zero)  # relu_res = tbe.cast_to(relu_res, dtype_0)  #  # return relu_res


@tbe.common.register.register_operator("FLOORDIVADDSQRTMULRELU")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def floordiv_add_sqrt_mul_relu(input_0, input_1, input_2, input_3, output, kernel_name="floordiv_add_sqrt_mul_relu"):
    """
    algorithm: floordiv_add_sqrt_mul_relu
    calculating:
    c = floordiv(input_0,input_1)
    e = add(c,input_2)
    f = sqrt(e)
    h = mul(f,input_3)
    i = relu(h)
    Parameters
    ----------
    input_0 : dict
        including shape, dtype and range, only support float16, float32, int32
    input_1 : dict
        including shape, dtype and range, only support float16, float32, int32
    input_2 : dict
        including shape, dtype and range, only support float16, float32, int32
    input_3 : dict
        including shape, dtype and range, only support float16, float32, int32
    output: dict
        including shape, dtype and range, only support float16, float32, int32
    kernel_name : str
       cce kernel name, default value is floordiv_add_sqrt_mul_relu

    Returns
    -------
    None
    """
    # check input tensor data_type
    para_check.check_elewise_shape_range([input_0, input_1, input_2, input_3], support_broadcast=True)
    dtype_0 = input_0.get("dtype").lower()
    dtype_1 = input_1.get("dtype").lower()
    dtype_2 = input_2.get("dtype").lower()
    dtype_3 = input_3.get("dtype").lower()

    ins = tbe.dsl.classify([input_0, input_1, input_2, input_3], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_0, _input_1, _input_2, _input_3) in ins:
        with tbe.dsl.compute():
            shape_0, shape_1, shape_2, shape_3 = shape_util.variable_shape([_input_0, _input_1, _input_2, _input_3])

            data_0 = tvm.placeholder(shape_0, name="data_0", dtype=dtype_0)
            data_1 = tvm.placeholder(shape_1, name="data_1", dtype=dtype_1)
            data_2 = tvm.placeholder(shape_2, name="data_2", dtype=dtype_2)
            data_3 = tvm.placeholder(shape_3, name="data_3", dtype=dtype_3)
            res = floordiv_add_sqrt_mul_relu_compute(data_0, data_1, data_2, data_3, output, kernel_name)
            # return res

            tensors.append((data_0, data_1, data_2, data_3, res))
        with tvm.target.cce():
            schedule = tbe.dsl.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.dsl.build(schedules, config)
