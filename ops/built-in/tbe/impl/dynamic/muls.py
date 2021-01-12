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
dynamic muls
"""
from functools import reduce as reduceIns
import te.lang.cce
from te import tvm
import te.lang.base as tbe_base
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import para_check
from te.utils import shape_util

def muls_compute(x, scalar, kernel_name="muls"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    scalar : a number of float or int
    kernel_name : str
        kernel name, default value is "muls"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    res = te.lang.cce.vmuls(x, scalar)
    return res


@tbe_base.register_operator("Muls")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def muls(x, y, value, kernel_name="muls"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as x
    value : float
        scale
    kernel_name : str
        kernel name, default value is "muls"

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype")
    input_dtype = x_dtype.lower()
    
    check_list = ["float16", "float32", "int32", "int16"]
    para_check.check_dtype(x_dtype, check_list)
    ins = classify([x], Mode.ELEWISE)
    schedules, tensors = [], []
    scalar = tvm.const(value, dtype=input_dtype)
    for (input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)
            res = muls_compute(data_input, scalar)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = te.lang.cce.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    te.lang.cce.build(schedules, config)
