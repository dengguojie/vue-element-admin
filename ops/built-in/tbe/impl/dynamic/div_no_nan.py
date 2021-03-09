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
dynamic tan
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator


def div_no_nan_compute(input_x, input_y, ):
    """
     div_no_nan_compute
     Returns 0 if the denominator is zero, else, like Div.
     ----------
     input_x: TVM tensor
         the placeholder of input tensor x
     input_y: TVM tensor
         the placeholder of input tensor y
     output_z: dict
         dict with keys(shape and dtype) of output
     but output_z not used
     kernel_name: str
         cce kernel name

     Returns
     -------
     res: TVM tensor
         the result of div_no_nan_compute
     """
    dtype = input_x.dtype
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                              param_name_input1="input_x",
                                                              param_name_input2="input_y")

    int_list = ("int32", "int8", "uint8")
    if dtype in int_list:
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")

    if dtype in ("float16",):
        help_min = tvm.const(2 ** (-24), "float16")
        help_rec_one = tvm.const(2 ** 12, "float16")
        help_rec_sec = tvm.const(2 ** 12, "float16")
        neg_one = tvm.const(-1, "float16")
    else:
        help_min = tvm.const(2 ** (-126), "float32")
        help_rec_one = tvm.const(2 ** 38, "float32")
        help_rec_sec = tvm.const(2 ** 44, "float32")
        neg_one = tvm.const(-1, "float32")

    y_cmp = tbe.vabs(input_y)
    if tbe_platform.api_check_support("te.lang.cce.vmins", "float32"):
        y_index_help_1 = tbe.vmins(y_cmp, help_min)
    else:
        cmp_help = tbe.broadcast(help_min, shape_y)
        y_index_help_1 = tbe.vmin(y_cmp, cmp_help)
    y_index_help_2 = tbe.vmuls(y_index_help_1, help_rec_one)
    y_index = tbe.vmuls(y_index_help_2, help_rec_sec)
    if dtype not in ("float16",):
        y_index = tbe.vmuls(y_index, help_rec_sec)

    data_x_broadcast = tbe.broadcast(input_x, shape_max)
    data_y_broadcast = tbe.broadcast(input_y, shape_max)
    index_y_broadcast = tbe.broadcast(y_index, shape_max)
    neg_index = tbe.vadds(index_y_broadcast, neg_one)
    data_y_broadcast = tbe.vadd(data_y_broadcast, neg_index)
    res_vdiv = tbe.vdiv(data_x_broadcast, data_y_broadcast)
    res = tbe.vmul(res_vdiv, index_y_broadcast)

    if dtype in int_list:
        res = tbe.floor(res)
        res = tbe.cast_to(res, dtype)

    return res


@register_operator("DivNoNan")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def div_no_nan(input_x, input_y, output_z, kernel_name="div_no_nan"):
    """
    algorithm: div_no_nan_cce4
    Returns 0 if the denominator is zero, else, like Div.
    Supports broadcasting.

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "div_no_nan"

    Returns
    -------
    None
    """
    check_list = ["float16", "float32", "int32"]
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()

    if not x_dtype in check_list or not y_dtype in check_list:
        error_detal = "sub only support float16, float32, int32"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name,
                                                               "input_x",
                                                               "input_y",
                                                               error_detal)

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([x1, x2])
            data1 = tvm.placeholder(x_shape, x_dtype, "data1")
            data2 = tvm.placeholder(y_shape, y_dtype, "data2")
            res = div_no_nan_compute(data1, data2, )
            tensors.append([data1, data2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
