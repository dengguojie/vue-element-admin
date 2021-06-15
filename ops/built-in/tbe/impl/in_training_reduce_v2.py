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
in_training_reduce_v2
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tuple_sum
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=locally-disabled,unused-argument,invalid-name,redefined-builtin
def op_select_format(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    select format dynamically
    """
    input0 = gen_param(classify="input0",
                       name="x",
                       datatype="float16,float,float16,float",
                       format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output0 = gen_param(classify="output0",
                        name="sum",
                        datatype="float,float,float,float",
                        format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
    output1 = gen_param(classify="output1",
                        name="square_sum",
                        datatype="float,float,float,float",
                        format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _reduce_compute(x, format_x):
    """
    algorithm: part of instance_norm_v2
    The first step of instance_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.

    Returns
    -------
    res: TVM tensor list
        the result of in_training_reduce compute
    """
    if format_x in ("NDC1HWC0",):  # only support NDC1HWC0 and NC1HWC0
        axis = [1, 3, 4]
    else:
        axis = [2, 3]

    square_x = tbe.vmul(x, x)
    sum_x, square_sum_x = tuple_sum([x, square_x], axis, True)
    res = [sum_x, square_sum_x]

    return res


def in_training_reduce_compute(x, format_x):
    """
    algorithm: part of instance_norm_v2
    The first step of instance_norm_v2
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    kernel_name: str
        kernel name, default value is "in_training_reduce_v2"

    Returns
    -------
    res: TVM tensor list
        the result of in_training_reduce_v2 compute
    """
    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")
    res = _reduce_compute(x, format_x)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def in_training_reduce_v2(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    algorithm: part of instance_norm_v2
    The first step of instance_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "in_training_reduce_v2"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    format_x = x.get("format")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x.lower())

    res = in_training_reduce_compute(data_x, format_x)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = [data_x] + list(res)
    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.build(sch, config)
