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
broadcast_to_d

  Op_description :
    Broadcast an array for a compatible shape.

    # broadcast_to_d(
    #   x,
    #   y,
    #   shape,
    #   kernel_name='broadcast_to_d')

  Supportive_dtype_format :
    ['float32', 'float16', 'int8', 'uint8', 'int32']
    ['ND', 'NCHW', 'NHWC']

  Constraint :
    [1] All : `shape` must be an 1-D 'int' tensor.
    [2] All : shape size limit is 2147483648.
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector

NUM_ONE = 1


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("broadcast_to_d")
def broadcast_to_compute(x, y, shape, kernel_name='broadcast_to_d'):
    """
    Process broadcast_to operator.

    Parameters:
    ----------
    x : the input tensor.

    y : the dict of output.

    shape : the desired output shape.

    kernel_name : cce kernel name, default value is "broadcast_to_d".

    Returns:
    -------
    output_tensor : tensor after broadcast_to.
    """
    dtype = x.dtype
    shape_in = x.shape

    # tbe.broadcast supports float16, float32, int32.
    # so convert int8, uint8 to float16
    if dtype in ('int8', 'uint8'):
        x = tbe.cast_to(x, 'float16')

    python_shape_in = shape_util.shape_to_list(shape_in)
    if list(python_shape_in) == list(shape):
        if dtype == "int32":
            # tbe.vmuls supports float16, float32. int8, uint8, int32 will
            # be converted to float16. This will cause the data to be truncated.
            # so use tbe.vmul.
            value_one = tvm.const(NUM_ONE, dtype=dtype)
            value_one_tensor = tbe.broadcast(value_one, shape)
            output_tensor = tbe.vmul(x, value_one_tensor)
        else:
            output_tensor = tbe.vmuls(x, NUM_ONE)
    else:
        output_tensor = tbe.broadcast(x, shape, dtype)

    # convert float16 back to int8, uint8
    if dtype in ('int8', 'uint8'):
        return tbe.cast_to(output_tensor, dtype, f1628IntegerFlag=True)

    return output_tensor


def _check_shape_compatibility(shape_in, shape):
    """
    Check if the shape of input tensor is compatible with output tensor.

    Parameters:
    ----------
    shape_in : shape of input tensor.

    shape : shape of output tensor.

    Returns:
    -------
    comp_shape_in : new shape_in compatible with shape.
    """

    try:
        comp_shape_in, comp_shape, shape_max = shape_util.broadcast_shapes(
            shape_in, shape, param_name_input1="x", param_name_input2="shape")
        if comp_shape != shape_max:
            error_detail = "shape_in is not compatible with shape_out"
            error_manager_vector.raise_err_two_input_shape_invalid("broadcast_to_d", "shape_in", "shape_out",
                                                                   error_detail)
    except RuntimeError:
        error_detail = "shape_in is not compatible with shape_out"
        error_manager_vector.raise_err_two_input_shape_invalid("broadcast_to_d", "shape_in", "shape_out",
                                                               error_detail)

    return comp_shape_in


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def broadcast_to_d(x, y, shape, kernel_name="broadcast_to_d"):
    """
    Broadcast an array for a compatible shape.

    Parameters:
    ----------
    x : the dict of input. support data type: float32, float16, int8, uint8, int32.

    y : the dict of output.

    shape : shape of output tensor.

    kernel_name : cce kernel name, default value is "broadcast_to_d".

    Returns:
    -------
    None
    """
    inp_dtype = x.get('dtype').lower()
    check_list = ('float32', 'float16', 'int8', 'uint8', 'int32')
    para_check.check_dtype(inp_dtype, check_list, param_name="x")

    shape_x = x.get('shape')
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape, param_name="shape")

    compatible_shape_in = _check_shape_compatibility(shape_x, shape)

    var = tvm.placeholder(compatible_shape_in, inp_dtype, name='data_x')

    res = broadcast_to_compute(var, y, shape, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [var, res]}
    tbe.cce_build_code(sch, config)
