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
arg_max_grad
"""
# pylint: disable=too-many-lines
import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from te.utils import para_check


# pylint: disable=invalid-name,unused-argument,too-many-statements
@fusion_manager.register("arg_max_grad_d")
def arg_max_grad_d_compute(var, indices, updates, assist, y, dimension=0, kernel_name="arg_max_grad_d"):
    """
    Returns the grad of argmax tensor
    """
    data_shape = y.get("shape")
    data_indices = tbe.broadcast(indices, data_shape)
    data_updates = tbe.broadcast(updates, data_shape)

    data_eq = tbe.vcmp(data_indices, assist, 'eq', 'bool')
    res = tbe.vsel(data_eq, data_updates, var)
    return res

# pylint: disable=invalid-name,unused-argument,too-many-statements
def arg_max_grad_d_check_param(var, indices, updates, assist, y, kernel_name="arg_max_grad_d"):
    """
    check input param
    """
    indices_shape = indices.get("shape")
    updates_shape = updates.get("shape")
    var_shape = var.get("shape")
    assist_shape = assist.get("shape")
    
    var_dtype = var.get("dtype")
    indices_dtype = indices.get("dtype")
    updates_dtype = updates.get("dtype")
    assist_dtype = assist.get("dtype")

    # check input param
    if list(indices_shape) != list(updates_shape):
        expected_value = "should be equal"
        real_value = "not equal"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "shape of indices and updates",
                                                           expected_value, real_value)
    if list(var_shape) != list(assist_shape):
        expected_value = "should be equal"
        real_value = "not equal"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "shape of var and assist",
                                                           expected_value, real_value)
    check_list = ("float16", "float32", "int8", "int32")
    dtype_input_var = var_dtype.lower()
    para_check.check_dtype(dtype_input_var, check_list, param_name="var")
    dtype_updates = updates_dtype.lower()
    para_check.check_dtype(dtype_updates, check_list, param_name="updates")
    check_list = ("int32")
    dtype_indices = indices_dtype.lower()
    para_check.check_dtype(dtype_indices, check_list, param_name="indices")
    dtype_assist = assist_dtype.lower()
    para_check.check_dtype(dtype_assist, check_list, param_name="assist")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def arg_max_grad_d(var, indices, updates, assist, y, dimension=0, kernel_name="arg_max_grad_d"):
    """
    Returns the grad of argmax tensor

        Parameters
        ----------
        var: dict
            the dict of input tensor.
        indices: dict
            the dict of input tensor. support dtype is int32
        updates: dict
            the dict of input tensor.
        assist: dict
           the dict of input tensor. support dtype is int32
        y: dict
            the dict of output tensor.
        dimension: int8
            the value of  argmax forward dimension.
        kernel_name: str
            cce kernel name, default value is "arg_max_grad_d".

        Returns
        -------
        None
    """
    # check input param
    arg_max_grad_d_check_param(var, indices, updates, assist, y, kernel_name)

    #set dimension of shape = 1 for broadcast proc
    var_shape = var.get("shape")
    data_shape = shape_util.shape_to_list(var_shape)
    if dimension < 0:
        dimension = dimension + len(var_shape)
    data_shape[dimension] = 1
    
    data_var = tvm.placeholder(var.get("shape"), dtype=var.get("dtype"), name="data_var")
    data_indexs = tvm.placeholder(data_shape, dtype=indices.get("dtype"), name="data_indexs")
    data_updates = tvm.placeholder(data_shape, dtype=updates.get("dtype"), name="data_updates")
    data_assist = tvm.placeholder(assist.get("shape"), dtype=assist.get("dtype"), name="data_assist")
    
    res = arg_max_grad_d_compute(data_var, data_indexs, data_updates, data_assist, y, dimension, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_var, data_indexs, data_updates, data_assist, res],
              "bool_storage_as_1bit":False}
    tbe.cce_build_code(schedule, config)
