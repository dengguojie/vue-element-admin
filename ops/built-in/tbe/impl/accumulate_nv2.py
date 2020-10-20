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
accumulate_nv2
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util

MIN_TENSOR_NUM = 1
MAX_TENSOR_NUM = 40
NUM_ONE = 1


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("accumulate_nv2")
def _accumulate_nv2_compute(x, y, num, kernel_name='accumulate_nv2'):
    """
    Process accumulate_nv2 operator.

    Parameters:
    ----------
    x : the list of input tensor.

    y : the dict of output.

    num : the size of input.

    kernel_name : cce kernel name, default value is "accumulate_nv2".

    Returns:
    -------
    result : result of accumulate.
    """

    dtype = x[0].dtype
    shape = x[0].shape
    length = len(x)

    result = x[0]
    # in order to improve the accuracy, convert float16 to float32
    if dtype == 'float16' and length > 1 and \
       tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        result = tbe.cast_to(result, 'float32')

    for i in range(1, length):
        rhs = x[i]
        if dtype == 'float16' and \
           tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
            rhs = tbe.cast_to(x[i], 'float32')
        result = tbe.vadd(result, rhs)

    if length == 1:
        # tbe.vmuls supports float16, float32. int8, uint8, int32 will
        # be converted to float16. This will cause the data to be truncated.
        # so use tbe.vmul.
        if dtype == "int32":
            value_one = tvm.const(NUM_ONE, dtype=dtype)
            value_one_tensor = tbe.broadcast(value_one, shape)
            result = tbe.vmul(result, value_one_tensor)
        else:
            result = tbe.vmuls(result, NUM_ONE)

    # in order to improve the accuracy, convert float32 back to float16
    if dtype == 'float16' and length > 1:
        result = tbe.cast_to(result, 'float16')

    return result


def _get_shape_and_dtype(input_dict):
    """
    Get shape and data type from dictionary.

    Parameters:
    ----------
    input_dict : input dictionary.

    Returns:
    -------
    shape: the shape of input tensor.

    inp_dtype: the lower data type of input tensor.
    """

    shape = input_dict.get('shape')
    para_check.check_shape(shape, param_name="x")

    check_list = ('float32', 'float16', 'int8', 'uint8', 'int32')
    dtype = input_dict.get('dtype')
    para_check.check_dtype(dtype, check_list, param_name="x")
    inp_dtype = dtype.lower()

    return shape, inp_dtype


def _check_all_shape_and_dtype_same(input_list):
    """
    Check shape and data type of inputs are all same, and return shape and dtype.

    Parameters:
    ----------
    input_list : the list of input dict.

    Returns:
    -------
    shape: the shape of input tensor.

    dtype: the data type of input tensor.
    """

    if input_list is None or len(input_list) < MIN_TENSOR_NUM:
        raise ValueError(
            'inputs must be a list of at least one Tensor with the same dtype and shape'
        )

    # ccec compiler does not support more than 40 parameters, so limit it
    if len(input_list) > MAX_TENSOR_NUM:
        raise RuntimeError('tensor_num need in range [1, 40].')

    shape, dtype = _get_shape_and_dtype(input_list[0])

    if not all((shape, dtype) == _get_shape_and_dtype(x) for x in input_list):
        raise ValueError(
            'inputs must be a list of at least one Tensor with the same dtype and shape'
        )

    return shape, dtype


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def accumulate_nv2(x, y, num, kernel_name="accumulate_nv2"):
    """
    Returns the element-wise sum of a list of tensors.

    Parameters:
    ----------
    x : the list of input dict. support dtype: float16, float32, int8, uint8, int32.

    y : the dict of output.

    num : the size of input.

    kernel_name : cce kernel name, default value is "accumulate_nv2".

    Returns:
    -------
    None
    """
    shape, dtype = _check_all_shape_and_dtype_same(x)
    if len(x) != num:
        raise RuntimeError(
            'The size of input and num must be same.'
        )
    shape, _ = shape_util.refine_shape_axes(shape, [])

    tensor_list = [None]*len(x)
    for i in range(len(x)):
        data_name = 'data%d' % i
        data = tvm.placeholder(shape, name=data_name, dtype=dtype)
        tensor_list[i] = data

    with tvm.target.cce():
        res = _accumulate_nv2_compute(tensor_list, y, kernel_name)
        sch = tbe.auto_schedule(res)

    tensor_list.append(res)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
