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
dynamic add
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_common


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("Add", op_mode="dynamic", support_fusion=True)
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x:
    left input, may be dict or tensor

    input_y:
    left input, may be dict or tensor

    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output of the data's add
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = \
        shape_util.broadcast_shapes(shape_x, shape_y,
                                    param_name_input1="input_x",
                                    param_name_input2="input_y")

    x_dtype = input_x.dtype.lower()
    if x_dtype in ("uint8", "int8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    input_x = tbe.broadcast(input_x, shape_max)
    input_y = tbe.broadcast(input_y, shape_max)
    res = tbe.vadd(input_x, input_y)

    if x_dtype in ("uint8", "int8"):
        res = util_common.uint8_int8_overflow_proc(res, x_dtype)

    return res


# 'pylint: disable=too-many-locals
@register_operator("Add")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def add(input_x, input_y, output_z, kernel_name="add"):
    """
    algorithm: add
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x : dict
       including shape, dtype and range, only support float16, float32, int32, uint8, int8
    input_y : dict
       including shape, dtype and range, only support float16, float32, int32, uint8, int8
    output_z: dict
       shape should be broadcast shape of input, and type equals to input
    kernel_name : str
       cce kernel name, default value is add

    Returns
    -------
    None
    """

    # check input tensor data_type
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "int8")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("add", "input_x", "input_y",
                                                              str(x_dtype), str(y_dtype))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x, _input_y) in ins:
        with tbe.compute():
            shape_x, shape_y = \
                shape_util.variable_shape([_input_x, _input_y])
            data_x = tvm.placeholder(shape_x, name="data_1", dtype=x_dtype)
            data_y = tvm.placeholder(shape_y, name="data_2", dtype=y_dtype)
            res = add_compute(data_x, data_y, output_z, kernel_name)

            tensors.append((data_x, data_y, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
