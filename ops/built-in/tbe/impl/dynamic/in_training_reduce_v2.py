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
dynamic in_training_reduce_v2
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tuple_sum
from impl.in_training_reduce_v2 import op_select_format as in_op_select_format


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-locals, too-many-statements,redefined-builtin
def op_select_format(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    select format dynamically
    """
    return in_op_select_format(x, sum, square_sum, kernel_name="in_training_reduce_v2")


@register_operator_compute("INTrainingReduceV2", op_mode="dynamic", support_fusion=True)
def in_training_reduce_v2_compute(x, sum, square_sum, kernel_name="in_training_reduce_v2", reduce_axis=None):
    """
    algorithm: part of instance_norm
    The first step of instance_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "in_training_reduce_v2"
    reduce_axis: list
        reduce axis of input shape

    Returns
    -------
    res: TVM tensor list
        the result of in_training_reduce_v2 compute
    """
    dtype = x.dtype.lower()
    if dtype == "float16":
        x = tbe.cast_to(x, "float32")

    data_format = sum.get("format").upper()
    if not reduce_axis and data_format in ("NC1HWC0",):
        axis = [2, 3]
    elif not reduce_axis and data_format in ("NDC1HWC0",):
        axis = [1, 3, 4]
    else:
        axis = reduce_axis
    square_x = tbe.vmul(x, x)
    sum_x, square_sum_x = tuple_sum([x, square_x], axis, True)
    res = [sum_x, square_sum_x]

    return res


@register_operator("INTrainingReduceV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def in_training_reduce_v2(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    algorithm: part of instance_norm
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
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
    dtype_x = x.get("dtype")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    data_format = x.get("format")
    if data_format in ("NC1HWC0",):
        list_axis = [2, 3]
    else:
        list_axis = [1, 3, 4]

    ins = classify([x, list_axis], OpPatternMode.TUPLE_REDUCE)
    schedules, tensors = [], []
    for (_x, _reduce_axis) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])
            input_x = tvm.placeholder(shape_x[0], name="input_x", dtype=dtype_x)
            res = in_training_reduce_v2_compute(input_x, sum, square_sum, kernel_name, _reduce_axis)

            tensor_list = [input_x] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
