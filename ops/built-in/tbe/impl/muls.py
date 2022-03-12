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
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.platform.fusion_manager import fusion_manager
from te.utils import shape_util
from impl.util.util_compute import check_batchmatmul_fuse


def _muls_compute_with_batchmatmul(lhs_tensor, const_tensor):
    """
    calculating data's muls, c = a * b
    Parameters
    ----------
    lhs_tensor: TVM tensor
        the placeholder of first input data
    const_value: Const tensor
    Returns
    -------
    res : output of the lhs_tensor + const_tensor
    """
    if "para_name" in lhs_tensor.op.attrs:
        para_name = lhs_tensor.op.attrs["para_name"].value
        para_name += "_muls"
    else:
        para_name = "muls"
    batch_shape = shape_util.shape_to_list(lhs_tensor.op.attrs["batch_shape"])
    res = tbe.vmuls(lhs_tensor, const_tensor)
    res.op.attrs["batch_shape"] = batch_shape
    res.op.attrs["para_name"] = para_name

    return res


# 'pylint: disable=invalid-name,unused-argument
@fusion_manager.register("muls")
def muls_compute(x, y, value, kernel_name="muls"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    y : TVM tensor
        output
    value : a number of float
    kernel_name : str
        kernel name, default value is "muls"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    const_tensor = tvm.const(value, dtype=x.dtype)
    batch_matmul_fuse_flag = check_batchmatmul_fuse(x)
    if batch_matmul_fuse_flag:
        return _muls_compute_with_batchmatmul(x, const_tensor)
    res = tbe.vmuls(x, const_tensor)
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

    input_x = tvm.placeholder(shape, name="x", dtype=input_dtype)
    res = muls_compute(input_x, y, value, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [input_x, res]}

    tbe.cce_build_code(schedule, config)
