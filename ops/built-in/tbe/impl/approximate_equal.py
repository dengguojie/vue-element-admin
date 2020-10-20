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
approximate_equal
"""
import operator

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util

NUM_ONE = 1.0
NUM_ZERO = 0.0

__all__ = ["approximate_equal"]


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("approximate_equal")
def approximate_equal_compute(input_x, input_y, output_z, tolerance,
                              kernel_name="approximate_equal"):
    """
    algorithm: approximate_equal

    calculating abs(x-y) <= tolerance

    Parameters
    ----------
    input_x : the placeholders of input data
    input_y : the placeholders of input data
    tolerance: default 1e-5
    output_z: shape and dtype of output
    kernel_name: cce kernel name, default value is "approximate_equal"
    Returns
    -------
    the function of _approximate_equal_compute
    """

    input_dtype = input_x.dtype
    if input_dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")

    res_vsub = tbe.vsub(input_x, input_y)
    res_vabs = tbe.vabs(res_vsub)

    res_vabs = tbe.cast_to(res_vabs, input_x.dtype)
    tol_tensor = tbe.broadcast(tvm.const(tolerance, input_x.dtype),
                                       input_x.shape)

    res_cmp = tbe.vcmp(res_vabs, tol_tensor, 'le')
    zero_rb_tensor = tbe.broadcast(tvm.const(NUM_ZERO, "float16"), input_x.shape)
    one_rb_tensor = tbe.broadcast(tvm.const(NUM_ONE, "float16"), input_x.shape)
    res = tbe.vsel(res_cmp, one_rb_tensor, zero_rb_tensor)

    res = tbe.cast_to(res, "int8")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def approximate_equal(input_x, input_y, output_z, tolerance=1e-5,
                      kernel_name="approximate_equal"):
    """
    abs(x-y) <= tolerance
    Parameters
    ----------
    input_x : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    input_y : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    output_z : dict, include shape and dtype, reserve

    tolerance: default 1e-5

    kernel_name : str
        cce kernel name, default value is "approximate_equal"

    Returns
    ------
    None
    """

    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    in_dtype = input_x.get("dtype")
    in_y_dtype = input_y.get("dtype")

    if tolerance < 0:
        raise RuntimeError("tolerance should >= 0")


    # check shape
    if not operator.eq(shape_x, shape_y):
        raise RuntimeError("all input shape must same")
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
        raise RuntimeError("all input type must same.")

    in_data_x = tvm.placeholder(shape_x, name="shape_x", dtype=in_dtype)
    in_data_y = tvm.placeholder(shape_y, name="shape_y", dtype=in_dtype)
    res = approximate_equal_compute(in_data_x, in_data_y, output_z,
                                    tolerance, kernel_name)

    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [in_data_x, in_data_y, res],
              "bool_storage_as_1bit": False}
    tbe.cce_build_code(auto_sch, config)
