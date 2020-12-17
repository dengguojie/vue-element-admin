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
abs
"""
from functools import reduce as reduceIns
import te.platform as tbe_platform
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
from te.utils.op_utils import variable_shape


# pylint: disable=invalid-name,unused-argument
def abs_compute(x, y, kernel_name="abs"):
    """
    algorithm: abs

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    y: dict
        dict info of y
    kernel_name: str
        kernel name, default value is "abs"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    inp_dtype = x.dtype
    if not tbe_platform.api_check_support("te.lang.cce.vabs", inp_dtype):
        x = tbe.cast_to(x, "float16")

    res = tbe.vabs(x)
    if inp_dtype == "int32":
        res = tbe.round(res)

    if not tbe_platform.api_check_support("te.lang.cce.vabs", inp_dtype):
        res = tbe.cast_to(res, inp_dtype)

    return res


# pylint: disable=redefined-builtin
@tbe_base.register_operator("Abs")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def abs(x, y, kernel_name="abs"):
    """
    algorithm: abs

    calculating data's abs,y= |x|

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is abs

    Returns
    -------
    None
    """
    dtype_input = x.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    check_dtype(dtype_input, check_list, param_name="x")

    ins = classify([x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe_base.compute():
            x_shape = variable_shape([_x])

            fuse_shape = [1]
            fuse_shape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuse_shape, name="data_input",
                                         dtype=dtype_input)
            res = abs_compute(data_input, y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

