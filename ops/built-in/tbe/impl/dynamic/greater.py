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
greater
"""
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.utils.op_utils import check_op_params
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_elewise_shape_range
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import shape_util
from impl.util.platform_adapter import register_operator


# min float32 value
MIN_FP32 = 2 ** (-126)
# min float16 value
MIN_FP16 = 2 ** (-24)


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments,unused-argument,invalid-name
def _greater_compare(data, shape, dtype, data_min):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    data : tuple
        two input data
    shape : list or tuple
        shape of input data
    dtype : str
        source data type, support float16,float32,int32,int8,uint8
    data_min : tvm.const
        the minimal data according to dtype

    Returns
    -------
    the compare result
    """
    data_zero = tbe.broadcast(tvm.const(0, dtype), shape, dtype)
    if dtype == "int32":
        data_one = tbe.broadcast(tvm.const(1, "float16"), shape, "float16")
    else:
        data_one = tbe.broadcast(tvm.const(1, dtype), shape, dtype)

    res_sub = tbe.vsub(data[1], data[0])
    # to amend sub zero result
    res_sub_zero = tbe.vadd(res_sub, data_min)
    res_min = tbe.vmin(res_sub_zero, data_min)
    res_max = tbe.vmax(res_min, data_zero)

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        max_support_fp32 = tvm.const(2 ** 62, dtype=dtype)
        res_mul1 = tbe.vmuls(res_max, max_support_fp32)
        res_mul2 = tbe.vmuls(res_mul1, max_support_fp32)
        res_mul = tbe.vmuls(res_mul2, tvm.const(2 ** 2, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        max_support_fp16 = tvm.const(2 ** 12, dtype=dtype)
        res_mul1 = tbe.vmuls(res_max, max_support_fp16)
        res_mul = tbe.vmuls(res_mul1, max_support_fp16)
    else:
        res_mul = tbe.cast_to(res_max, "float16")
    res = tbe.vsub(data_one, res_mul)

    return tbe.cast_to(res, "uint8", True)


def greater_compute(x, y, z, kernel_name="greater"):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : Tensor
        input data_x
    y : Tensor
        input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    the result
    """
    shape_x = shape_util.shape_to_list(x.shape)
    shape_y = shape_util.shape_to_list(y.shape)
    dtype = x.dtype.lower()
    shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, shape_y,
                                                          param_name_input1="x",
                                                          param_name_input2="y")

    if dtype in ("int8", "uint8"):
        x = tbe.cast_to(x, "float16")
        y = tbe.cast_to(y, "float16")
        dtype = "float16"

    data_x = tbe.broadcast(x, shape)
    data_y = tbe.broadcast(y, shape)

    if dtype == "float32":
        # minimun num of float32 2**(-126)
        data_min = tbe.broadcast(tvm.const(MIN_FP32, dtype=dtype),
                                 shape, dtype)
    elif dtype == "float16":
        # minimun num of float16 2**(-24)
        data_min = tbe.broadcast(tvm.const(MIN_FP16, dtype=dtype),
                                 shape, dtype)
    else:
        data_min = tbe.broadcast(tvm.const(1, dtype=dtype),
                                 shape, dtype)

    return _greater_compare((data_x, data_y), shape, dtype, data_min)


# pylint: disable=invalid-name
@register_operator("Greater")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def greater(x, y, z, kernel_name="greater"):
    """
    do element-wise greater operation between two input tensors

    Parameters:
    ----------
    x : dict
        shape and dtype of input data_x
    y : dict
        shape and dtype of input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_y = y.get("dtype").lower()

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    check_dtype(dtype_x, check_list, param_name="x")

    check_elewise_shape_range([x, y], support_broadcast=True)

    ins = classify([x, y], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe_base.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([x1, x2])
            data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype_x)
            data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype_y)
            res = greater_compute(data_x1, data_x2, z, kernel_name)

            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
