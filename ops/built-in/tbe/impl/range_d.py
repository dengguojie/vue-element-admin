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
range_d
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("range_d")
def range_d_compute(x, y, start, limit, delta, kernel_name="range_d"):
    """
    algorithm: range_d
    Description of calculating process with TE api, the computational formula
    is as follows.
    res = input_assist * delta + start

    Parameters
    ---------
    x: TVM tensor
        contains assist data
    start: scalar int float
        contains the data of start
    limit: scalar int float
        contains the data of limit
    delta: scalar int float
        contains the data of delta
    y: dict
        dict of output, which contains shape and dtype
    kernel_name: str
        cce kernel name, default value is "range_d"

    Returns
        ------
    res: TVM tensor
        the result of range_d compute
    """
    if isinstance(start, int) and isinstance(delta, int) \
       and isinstance(limit, int):
        res_start = tbe.broadcast(tvm.const(start, dtype="int32"), x.shape)
        res_delta = tbe.broadcast(tvm.const(delta, dtype="int32"), x.shape)
        mid_res = tbe.vmul(res_delta, x)
        res = tbe.vadd(mid_res, res_start)

        return res

    res_start = tbe.broadcast(tvm.const(start, dtype="float32"), x.shape)
    res_delta = tbe.broadcast(tvm.const(delta, dtype="float32"), x.shape)
    mid_res = tbe.vmul(x, res_delta)
    res = tbe.vadd(res_start, mid_res)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def range_d(x, y, start, limit, delta, kernel_name="range_d"):
    """
    algorithm: range_d
    Generates values in an interval
    A sequence of delta evenly-spaced values are generated beginning at start
    so that the last one is exactly limit
    For example:
    range_d(1, 10.0, 2) => [ 1.0,3.0,5.0,7.0,9.0]
    range_d(1, 5)=> [1,2,3,4]
    range_d(5)=> [0,1,2,3,4]

    Parameters
    ----------
    x: dict
        dict of input, which contains shape and dtype
    y: dict
        dict of output, which contains shape and dtype
    start: scalar
        scalar of start, which contains int or float
    limit: scalar
        scalar of limit, which contains int or float
    delta: scalar
        scalar of delta, which contains int or float
    kernel_name: str
        kernel name, default value is "range_d"

    Returns
    -------
    None
    """
    shape_assist = x.get("shape")
    dtype_assist = x.get("dtype").lower()

    para_check.check_shape(shape_assist, param_name="x")
    para_check.check_dtype(dtype_assist.lower(), ("int32", "float32"), param_name="x")

    if limit == start:
        raise RuntimeError("start can not equal to limit")
    if delta == 0:
        raise RuntimeError("the input of delta can not equal to zero")
    if (start > limit) and (delta > 0):
        raise RuntimeError("requires limit should more than start "
                           "when delta is more than zero")
    if (start < limit) and (delta < 0):
        raise RuntimeError("requires start should more than limit "
                           "when delta is less than zero")

    # check shape of assist,only support 1dim
    if len(shape_assist) != 1:
        raise RuntimeError(
            "range_d only support rank=1 while length of assist shape is %d"\
            % (len(shape_assist)))

    data = tvm.placeholder(shape_assist, name="data", dtype=dtype_assist)
    res = range_d_compute(data, y, start, limit, delta, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data, res]}
    tbe.cce_build_code(sch, config)
