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
logical_not
"""
from functools import reduce as reduceIns
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils import shape_util


# pylint: disable=locally-disabled,invalid-name,unused-argument
def logical_not_compute(x, y, kernel_name="logical_not"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "logical_not"

    Returns
    -------
    output tensor
    """
    const_one = tvm.const(1.0, "float16")
    shape_x = shape_util.shape_to_list(x.shape)

    const_broad = tbe.broadcast(const_one, shape_x)
    x_cast = tbe.cast_to(x, "float16", True)
    x_abs = tbe.vabs(x_cast)
    x_min = tbe.vmin(x_abs, const_broad)
    y_sub = tbe.vsub(x_min, const_broad)
    y_abs = tbe.vabs(y_sub)
    res_y = tbe.cast_to(y_abs, x.dtype, True)

    return res_y


@tbe_base.register_operator("LogicalNot")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def logical_not(x, y, kernel_name="logical_not"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support int8, int32
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "logical_not"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    if dtype_x == "bool":
        dtype_x = "int8"

    check_tuple = ("int8",)
    check_dtype(dtype_x, check_tuple, param_name="x1")

    schedules, tensors = [], []
    ins = classify([x], Mode.ELEWISE)
    for (input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype=dtype_x, name="data_input")
            res = logical_not_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)

