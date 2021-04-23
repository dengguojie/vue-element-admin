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
from te.utils.shape_util import refine_shape_axes
from te.utils.shape_util import broadcast_shapes
from impl.util.platform_adapter import error_manager_vector

MIN_TENSOR_NUM = 1
MAX_TENSOR_NUM = 40


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("accumulate_nv2")
def _accumulate_nv2_compute(tensor_list, out_shape, out_dtype, num, shape_list, same_shape, reduce_shape):
    """
    Process accumulate_nv2 operator.

    Parameters:
    ----------
    tensor_list : the list of input tensor.
    out_shape : the shape of output.
    out_dtype : the data type of output.
    num : the size of input.
    ----------
    """

    if num > 1:
        if same_shape: 
            result = tensor_list[0]
            # in order to improve the accuracy, convert float16 to float32
            if out_dtype == 'float16' and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
                result = tbe.cast_to(result, 'float32')
                for i in range(1, num):
                    tmp = tbe.cast_to(tensor_list[i], 'float32')
                    result = tbe.vadd(result, tmp)
            else:
                for i in range(1, num):
                    result = tbe.vadd(result, tensor_list[i])
        else:
            if shape_list[0] != out_shape:
                result = tbe.broadcast(tensor_list[0], out_shape)
            else:
                result = tensor_list[0]
            # in order to improve the accuracy, convert float16 to float32
            if out_dtype == 'float16' and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
                result = tbe.cast_to(result, 'float32')
                for i in range(1, num):
                    if shape_list[i] != out_shape:
                        tmp = tbe.broadcast(tensor_list[i], out_shape)
                    else:
                        tmp = tensor_list[i]
                    tmp = tbe.cast_to(tmp, 'float32')
                    result = tbe.vadd(result, tmp)
            else:
                for i in range(1, num):
                    if shape_list[i] != out_shape:
                        tmp = tbe.broadcast(tensor_list[i], out_shape)
                    else:
                        tmp = tensor_list[i]
                    result = tbe.vadd(result, tmp)

    else:
        result = tbe.broadcast(0, reduce_shape)
        # in order to improve the accuracy, convert float16 to float32
        if out_dtype == 'float16' and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
            result = tbe.cast_to(result, 'float32')
            tmp = tbe.cast_to(tensor_list[0], 'float32')
            result = tbe.vadd(result, tmp)
        else:
            result = tbe.vadd(result, tensor_list[0])

    # in order to improve the accuracy, convert float32 back to float16
    if out_dtype == 'float16' and tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
        result = tbe.cast_to(result, 'float16')

    return result


def _check_all_shape_and_dtype_same(x, y, num):
    """
    Check shape and data type of inputs are all same, and return shape and dtype.

    Parameters:
    ----------
    x : the list of input dict.
    num : the size of input.
    ----------
    """
    same_shape = True
    # ccec compiler does not support more than 40 parameters, so limit it
    if num > MAX_TENSOR_NUM or num < MIN_TENSOR_NUM:
        error_manager_vector.raise_err_input_param_not_in_range("accumulate_nv2", "num", MIN_TENSOR_NUM, MAX_TENSOR_NUM, num)
    
    check_list = ('float32', 'float16', 'int8', 'uint8', 'int32')
    shape_list = []
    out_dtype = y.get('dtype').lower()
    para_check.check_dtype(out_dtype, check_list)

    for i in range(num):
        shape = x[i].get('shape') 
        para_check.check_shape(shape)
        shape_list.append(shape)
        
        dtype = x[i].get('dtype').lower()
        if dtype != out_dtype:
            error_detail = "The input and output data types should be the same."
            error_manager_vector.raise_err_two_input_dtype_invalid("accumulate_nv2", "x", "y", error_detail)
    out_shape = shape_list[0]

    for i in range(1, num):
        if out_shape != shape_list[i]:
            _, _, out_shape = broadcast_shapes(out_shape, shape_list[i])
            same_shape = False
    
    back_dict = [shape_list, out_shape, out_dtype, same_shape]

    return back_dict


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
    ----------
    """
    if len(x) != num:
        error_detail = "The size of input and num must be same."
        error_manager_vector.raise_err_two_input_shape_invalid("accumulate_nv2", len(x), num, error_detail)
    
    back_dict = _check_all_shape_and_dtype_same(x, y, num)
    shape_list, out_shape, out_dtype, same_shape = back_dict[0], back_dict[1], back_dict[2], back_dict[3]

    para_check.check_kernel_name(kernel_name)
    reduce_shape, _ = refine_shape_axes(out_shape, [])

    tensor_list = []
    if same_shape:
        for i in range(num):
            data_name = 'data%d' % i
            data = tvm.placeholder(reduce_shape, name=data_name, dtype=out_dtype)
            tensor_list.append(data)
    else:
        for i in range(num):
            data_name = 'data%d' % i
            data = tvm.placeholder(shape_list[i], name=data_name, dtype=out_dtype)
            tensor_list.append(data)
        
    res = _accumulate_nv2_compute(tensor_list, out_shape, out_dtype, num, shape_list, same_shape, reduce_shape)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list.append(res)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
