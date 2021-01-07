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
muls
"""
import te.lang.cce
from te import tvm
from te.utils import para_check
from te.platform.fusion_manager import fusion_manager
from topi import generic


# pylint: disable=invalid-name,unused-argument
@fusion_manager.register("muls")
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


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
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
    shape = x.get("shape")
    dtype = x.get("dtype")
    input_dtype = dtype.lower()

    check_list = ["float16", "float32", "int32", "int16"]
    para_check.check_dtype(input_dtype, check_list)

    scalar = tvm.const(value, dtype=input_dtype)
    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = muls_compute(data_input, scalar)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)
