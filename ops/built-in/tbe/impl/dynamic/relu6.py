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
relu6
f(x) = min(max(0,x), 6)
"""
from functools import reduce as reduceIns
import te.platform as tbe_platform
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils import shape_util
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from impl.util.platform_adapter import register_operator


# pylint: disable=unused-argument,too-many-locals
def relu6_compute(input_x, output_y, kernel_name="relu6"):
    """
    compute of relu6

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output,should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "relu6"

    Returns
    -------
    compute result of relu6
    """
    tmp_res = tbe.vmaxs(input_x, tvm.const(0, input_x.dtype))
    final_res = tbe.vmins(tmp_res, tvm.const(6, input_x.dtype))

    return final_res


@register_operator("Relu6")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def relu6(input_x, output_y, kernel_name="relu6"):
    """
       f(x)= 6(x >= 6)
       f(x)= 0(x <= 0)
       f(x)= x(0<x<6)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output_y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6"

    Returns
    ------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    vmaxs_support = tbe_platform.api_check_support("te.lang.cce.vmaxs", "float32")
    if not vmaxs_support:
        check_dtype(dtype_input, ("int32", "float16"), param_name="input_x")

    check_list = ("int32", "float16", "float32")
    check_dtype(dtype_input, check_list, param_name="input_x")

    ins = classify([input_x], Mode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe_base.compute():
            x_shape = shape_util.variable_shape([_input_x])

            fuse_shape = [1]
            fuse_shape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuse_shape, name="data_input",
                                         dtype=dtype_input)
            res = relu6_compute(data_input, output_y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
