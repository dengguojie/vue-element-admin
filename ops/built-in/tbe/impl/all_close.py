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
all_close
"""
import operator
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector

NUM_ONE = 1.0
NUM_ZERO = 0.0

__all__ = ["all_close"]


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("all_close")
def all_close_compute(input_x, input_y, output_num, output_diff, atol, rtol,
                              kernel_name="all_close"):
    """
    algorithm: all_close

    calculating abs(x-y) > atol + rtol * abs(y)

    Parameters
    ----------
    input_x : the placeholders of input data
    input_y : the placeholders of input data
    atol: default 1e-5
    rtol: default 1e-3
    output_num: shape and dtype of output
    output_diff: shape and dtype of output
    kernel_name: cce kernel name, default value is "all_close"
    Returns
    -------
    the function of _all_close_compute
    """

    input_dtype = input_x.dtype
    shape = input_x.shape
    shape_list = []
    for i in range(len(shape)):
        shape_list.append(i)

    if input_dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")

    res_vsub = tbe.vsub(input_x, input_y)
    res_vabs = tbe.vabs(res_vsub)

    atol_tensor = tbe.broadcast(tvm.const(atol, input_x.dtype),
                                       input_x.shape)
    rtol_tensor = tbe.broadcast(tvm.const(rtol, input_x.dtype),
                                       input_x.shape)

    y_vabs = tbe.vabs(input_y)
    res_mul = tbe.vmul(rtol_tensor, y_vabs)
    res_vadd = tbe.vadd(atol_tensor, res_mul)

    res_cmp = tbe.vcmp(res_vabs, res_vadd, 'gt')

    zero_rb_tensor = tbe.broadcast(tvm.const(NUM_ZERO, "float16"), input_x.shape)
    one_rb_tensor = tbe.broadcast(tvm.const(NUM_ONE, "float16"), input_x.shape)
    res_sel = tbe.vsel(res_cmp, one_rb_tensor, zero_rb_tensor)

    res_sel = tbe.cast_to(res_sel, "float32")
    res_num = tbe.sum(res_sel, axis = shape_list)
    res_num = tbe.cast_to(res_num, "int32")

    res_div = tbe.vdiv(res_vabs, y_vabs)
    res_mul = tbe.vmul(res_div, res_sel)
    res_diff = tbe.reduce_max(res_mul, axis = shape_list)

    return res_num, res_diff


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def all_close(input_x, input_y, output_num, output_diff, atol=1e-5, rtol=1e-3,
                      kernel_name="all_close"):
    """
    abs(x-y) <= atol + rtol * abs(y)
    Parameters
    ----------
    input_x : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    input_y : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    output_num : dict, include shape and dtype, reserve
    output_diff : dict, include shape and dtype, reserve

    atol: default 1e-5
    rtol: default 0.001

    kernel_name : str
        cce kernel name, default value is "all_close"

    Returns
    ------
    None
    """

    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    in_dtype = input_x.get("dtype")
    in_y_dtype = input_y.get("dtype")

    if atol < 0:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "atol", \
                                                           ">= 0", atol)
    if atol < 0:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "rtol", \
                                                           ">= 0", rtol)

    # check shape
    if not operator.eq(shape_x, shape_y):
        error_detail = "shape of input_x and input_y should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_x", \
                                                               "input_y", error_detail)
    para_check.check_shape(shape_x, param_name="input_x")
    shape_x, _ = shape_util.refine_shape_axes(shape_x, [])
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])

    # check input tensor data_type
    check_list = ("float16", "float32")
    para_check.check_dtype(in_dtype, check_list, param_name="input_x")
    para_check.check_dtype(in_y_dtype, check_list, param_name="input_y")
    in_dtype = input_x.get("dtype").lower()
    in_y_dtype = input_y.get("dtype").lower()
    if not operator.eq(in_dtype, in_y_dtype):
        error_detail = "dtype of input_x and input_y should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_x", \
                                                               "input_y", error_detail)

    in_data_x = tvm.placeholder(shape_x, name="shape_x", dtype=in_dtype)
    in_data_y = tvm.placeholder(shape_y, name="shape_y", dtype=in_dtype)

    res_num, res_diff = all_close_compute(in_data_x, in_data_y, output_num, output_diff,
                                          atol, rtol, kernel_name)

    res = [res_num, res_diff]
    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    tensor_list = [in_data_x, in_data_y] + res

    config = {"name": kernel_name,
              "tensor_list": tensor_list,
              "bool_storage_as_1bit": False}
    tbe.cce_build_code(auto_sch, config)
