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
expand_d
"""
from te import tvm
import te.lang.cce as tbe
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from te.utils.shape_util import broadcast_shapes

NUM_ONE = 1


@fusion_manager.register("expand_d")
def expand_compute(x,
                   y,
                   shape,
                   kernel_name='expand_d'):
    """
    Process expand operator.

    Parameters:
    ----------
    x : the input tensor.
    y : the dict of output.
    shape : the desired output shape.
    kernel_name : cce kernel name, default value is "expand_d".

    Returns:
    -------
    output_tensor : tensor after expand.
    """
    dtype = x.dtype
    shape_in = x.shape

    # te.lang.cce.broadcast supports float16, float32, int32.
    # so convert int8, uint8 to float16
    if dtype in ('int8', 'uint8'):
        x = tbe.cast_to(x, 'float16')

    python_shape_in = [int(x) for x in shape_in]
    if list(python_shape_in) == list(shape):
        if dtype == "int32":
            # te.lang.cce.vmuls supports float16, float32. int8, uint8, int32 will
            # be converted to float16. This will cause the data to be truncated.
            # so use te.lang.cce.vmul.
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
        comp_shape_in, comp_shape, shape_max = broadcast_shapes(
            shape_in, shape, param_name_input1="x", param_name_input2="shape")
    except RuntimeError:
        raise ValueError('shape_in is not compatible with shape_out.')

    return comp_shape_in, comp_shape, shape_max


def expand_d(x,
             y,
             shape,
             kernel_name="expand_d"):
    """
    Broadcast an array for a compatible shape.

    Parameters:
    ----------
    x : the dict of input. support data type: float32, float16, int8, uint8, int32.
    y : the dict of output.
    shape : the other shape which needed to be broadcasted .
    kernel_name : cce kernel name, default value is "expand_d".

    Returns:
    -------
    None
    """
    check_list = ('float32', 'float16', 'int8', 'uint8', 'int32')
    x_dtype = x.get('dtype').lower()
    para_check.check_dtype(x_dtype, check_list, param_name="x")

    shape_in = x.get('shape')
    para_check.check_shape(shape_in, param_name="x")
    para_check.check_shape(shape, param_name="shape")

    compatible_shape_in, _, shape_max = _check_shape_compatibility(shape_in, shape)
    var = tvm.placeholder(compatible_shape_in, x_dtype, name='data_input')

    res = expand_compute(var, y, shape_max, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [var, res]}
    tbe.cce_build_code(sch, config)