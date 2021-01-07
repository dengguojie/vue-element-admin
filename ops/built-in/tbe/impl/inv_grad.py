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
inv_grad
"""
from te import tvm
import te.lang.cce as tbe
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector

# define a scalar , value = -1
SCALAR_NEGATIVE_ONE = -1


# pylint: disable=locally-disabled,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("inv_grad")
def inv_grad_compute(input_y, input_dy, output_z, kernel_name="inv_grad"):
    """
    compute inv_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input y
    input_dy: TVM tensor
        the placeholder of input dy
    output_z: TVM tensor
        shape and dtype of output
    kernel_name: str
        kernel name, default value is "inv_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_y = shape_util.shape_to_list(input_y.shape)
    dtype = input_y.dtype

    inv_const = tvm.const(SCALAR_NEGATIVE_ONE, dtype=dtype)
    has_improve_precision = False
    if dtype in ("float16", "int8"):
        if tbe_platform.cce_conf.api_check_support("te.lang.cce.vmuls",
                                                   "float32"):
            inv_const = tvm.const(SCALAR_NEGATIVE_ONE, dtype="float32")
            input_y = tbe.cast_to(input_y, "float32")
            input_dy = tbe.cast_to(input_dy, "float32")
            has_improve_precision = True
        const_res = tbe.vmuls(input_y, inv_const)
    elif dtype in ("int32",):
        inv_const = tbe.broadcast(inv_const, shape_y, "int32")
        const_res = tbe.vmul(inv_const, input_y)
    else:
        const_res = tbe.vmuls(input_y, inv_const)
    vmul_res = tbe.vmul(const_res, input_y)
    res = tbe.vmul(vmul_res, input_dy)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def inv_grad(input_y, input_dy, output_z, kernel_name="inv_grad"):
    """
    algorithm: inv_grad
    calculating data's reciprocal grad,dx = -1*dy*y*y, where `y = 1/x`, and `dy`
    is the corresponding input gradient.

    Parameters
    ----------
    input_y: dict
        shape and dtype of input_y, only support float16, float32, int32, int8
    input_dy: dict
        shape and dtype of input_dy, should be same shape and type as input_y
    output_z: dict
        shape and dtype of output, should be same shape and type as input_y
    kernel_name: str
        kernel name, default value is "inv_grad"

    Returns
    -------
    None
    """
    shape_input_y = input_y.get("shape")
    shape_input_dy = input_dy.get("shape")
    dtype_input_y = input_y.get("dtype")
    dtype_input_dy = input_dy.get("dtype")

    para_check.check_shape(shape_input_y, param_name="input_y")
    para_check.check_shape(shape_input_dy, param_name="input_dy")

    shape_input_y = shape_util.shape_refine(shape_input_y)
    shape_input_dy = shape_util.shape_refine(shape_input_dy)

    if list(shape_input_y) != list(shape_input_dy):
        error_detail = "the shape of input_y and input_dy must be equal!"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_y", \
                                                               "input_dy", error_detail)

    dtype_input_y = dtype_input_y.lower()
    dtype_input_dy = dtype_input_dy.lower()

    if dtype_input_dy != dtype_input_y:
        error_detail = "dtype of input_y and input_dy must be the same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_y", \
                                                               "input_dy", error_detail)

    check_list = ("float16", "float32", "int32", "int8")
    para_check.check_dtype(dtype_input_y, check_list, param_name="input_y")

    shape_input_dy, shape_input_y = shape_util.refine_shapes_for_broadcast(shape_input_dy,
                                                                           shape_input_y)
    data_dy = tvm.placeholder(shape_input_dy, name="data_dy",
                              dtype=dtype_input_dy)
    data_y = tvm.placeholder(shape_input_y, name="data_y", dtype=dtype_input_y)

    res = inv_grad_compute(data_y, data_dy, output_z, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    tbe.cce_build_code(sch, config)
