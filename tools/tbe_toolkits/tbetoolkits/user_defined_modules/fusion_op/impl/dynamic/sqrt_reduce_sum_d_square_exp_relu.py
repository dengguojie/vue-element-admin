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
dynamic sqrt_reduce_sum_d_square_exp_relu
"""
import tbe
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
import te
from te import platform as tbe_platform
from te import tvm
from te.utils.op_utils import check_dtype
from impl.util.platform_adapter import OpPatternMode
from tbe.dsl.base.operation import add_compile_info
from impl.dynamic.sqrt import sqrt_compute
from impl.dynamic.reduce_sum_d import reduce_sum_d_compute
from impl.dynamic.square import square_compute
from impl.dynamic.relu import relu_compute
from impl.dynamic.exp import exp_compute

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
SIZE_SIXTEEN = 16
CONST_ZERO = 0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def sqrt_reduce_sum_d_square_exp_relu_compute(x, y, axis=None, keepdims=False,
                                              kernel_name="sqrt_reduce_sum_d_square_exp_relu"):
    """
    algorithm: sqrt_reduce_sum_d_square_exp_relu
    calculating:
    res_0 = sqrt(x)
    res_1 = reduce_sum_d(res_0, y, axis)
    res_2 = square(res_1)
    res_3 = exp(res_2)
    res = relu(res_3)
    Parameters
    ----------
    x : dict
        including shape, dtype and range, only support float16, float32, int32
    y: dict
        including shape, dtype and range, only support float16, float32, int32
    axis: int, list, tuple or NONETYPE
        the axis for reduce.\
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name : str
       cce kernel name, default value is sqrt_reduce_sum_d_square_exp_relu

    Returns
    -------
    y of the op
    """

    res_sqrt = sqrt_compute(x, y)
    res_reduce = reduce_sum_d_compute(res_sqrt, y, axis, keepdims)
    res_square = square_compute(res_reduce, y,"float16")
    #res_exp = exp_compute(res_square, y)
    res = relu_compute(res_square, y)

    return res


@tbe.common.register.register_operator("SQRTREDUCESUMDSQUARERELUEXP")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sqrt_reduce_sum_d_square_exp_relu(x, y, axis=None, keepdims=False, kernel_name="sqrt_reduce_sum_d_square_exp_relu"):
    """
    algorithm: sqrt_reduce_sum_d_square_exp_relu
    calculating:
    res_0 = sqrt(x)
    res_1 = reduce_sum_d(res_0, y, axis)
    res_2 = square(res_1)
    res_3 = exp(res_2)
    res = relu(res_3)
    Parameters
    ----------
    x : dict
        input, including shape, dtype and range, only support float16, float32, int32
    y: dict
        y, including shape, dtype and range, only support float16, float32
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name : str
       cce kernel name, default value is sqrt_reduce_sum_d_square_exp_relu

    Returns
    -------
    None
    """
    axis = (1, 3)
    keepdims = False
    
    # check input tensor data_type
    dtype_x = x.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(dtype_x, check_list, param_name="x")
    x["rel_pos_to_reduce"] = "before"
    # add_compile_info("_ori_axis", axis)

    shape = x["shape"]
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)
    axis = shape_util.axis_check(shape_len, axis)
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}

    ins = tbe.dsl.classify([x, input_axis], OpPatternMode.REDUCE  ,{"keepdims": keepdims is False})
    schedules, tensors = [], []
    for (_x, _axis) in ins:
        with tbe.dsl.compute():
            shape_var_new = shape_util.variable_shape([_x, _axis], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=dtype_x)
            res = sqrt_reduce_sum_d_square_exp_relu_compute(data_input, y, _axis.get("value"), keepdims)
            # return res
            tensors.append([data_input, res])
        with tvm.target.cce():
            schedule = tbe.dsl.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.dsl.build(schedules, config)
