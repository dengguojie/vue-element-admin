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
dynamic adds
"""
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import para_check
from te.utils import shape_util


# pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-locals
def adds_compute(x, scalar, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    scalar : a number of float or int
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    res = tbe.vadds(x, scalar)
    return res


@tbe_base.register_operator("Adds")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def adds(x, y, value, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    value: a number of float
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    --------
    None
    """
    dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(dtype, check_list, param_name="x")
    ins = classify([x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe_base.compute():
            shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(shape, name="data_input", dtype=dtype)
            scalar = tvm.const(value, dtype=dtype)
            res = adds_compute(data_input, scalar)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
