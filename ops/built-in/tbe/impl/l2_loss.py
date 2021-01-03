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
l2_loss
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector


# pylint: disable=invalid-name, unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("l2_loss")
def l2_loss_compute(x, y, kernel_name="l2_loss"):
    """
     Reduce a tensor on a certain axis, and scale output with coeff

     Parameters
     ----------
     shape : shape of data
     dtype : source data type, only support float16, float32
     kernel_name : kernel name, default value is "l2_loss"

     Returns
     -------
     res: TVM tensor
        the result of compute
     """
    coeff_dtype = x.dtype
    _, axis = shape_util.simplify_axis_shape(x.shape, range(len(x.shape)))
    if x.dtype == "float16" and tbe_platform.api_check_support("tik.set_atomic_add")\
            and tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
        coeff_dtype = "float32"

    coeff_sqrt = tvm.const(1.0 / (2 ** (0.5)), dtype=coeff_dtype)

    data_mul = tbe.vmuls(x, coeff_sqrt)
    data_sqr = tbe.vmul(data_mul, data_mul)
    res = tbe.sum(data_sqr, axis)

    return res


# pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def l2_loss(x, y, kernel_name="l2_loss"):
    """
    Reduce a tensor on a certain axis, and scale output with coeff

    Parameters
    ----------
    shape : shape of data
    dtype : source data type, only support float16, float32
    kernel_name : kernel name, default value is "l2_loss"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")

    para_check.check_shape(shape, param_name="x")

    check_list = ("float16", "float32")
    if not dtype.lower() in check_list:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, "x", \
                                                                 "float16,float32", dtype.lower())

    shape, axis = shape_util.simplify_axis_shape(shape, range(len(shape)))

    inp_dtype = dtype.lower()
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    res = l2_loss_compute(data_input, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    tbe.cce_build_code(sch, config)

