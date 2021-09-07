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
ones_like
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te import tvm


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
def ones_like_compute(input_x, output_y, kernel_name="ones_like"):
    """
    Given a tensor, this operation returns a tensor of the same
    type and shape as `tensor` with all elements set to 1.

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: TVM tensor
        the placeholder of output data
    kernel_name : str
        cce kernel name, default value is "ones_like"

    Returns
    -------
    res: TVM tensor
        the result of ones_like_compute
    """
    src_dtype = input_x.dtype.lower()
    dst_type = src_dtype
    src_type_list = ("int8", "uint8")
    dst_type_list = ("int8", "uint8")
    if src_dtype in src_type_list:
        src_dtype = "float16"
    one = tvm.const(1, dtype=src_dtype)
    one_src = tbe.broadcast(one, input_x.shape)
    if src_dtype in dst_type_list:
        one_src = tbe.cast_to(one_src, dst_type,
                                       f1628IntegerFlag=True)
    else:
        one_src = tbe.cast_to(one_src, dst_type)
    with tvm.tag_scope("elewise_binary_phony"):
        res = tvm.compute(input_x.shape,
                             lambda *indices: one_src[indices] + input_x[indices],
                             name="elewise_binary_phony_output")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ones_like(x, y, kernel_name="ones_like"):
    """
    output a tensor of all one, shape and dtype is same of input

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32,
        int32,int8,uint8
    y: dict
        shape and dtype of output data
    kernel_name: str
        cce kernel name, default value is "ones_like"

    Returns
    ------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")
    para_check.check_shape(shape_input, param_name="x")
    check_list_src = ("float16", "float32", "int32", "int8", "uint8")
    src_dtype = dtype_input.lower()
    para_check.check_dtype(src_dtype, check_list_src, param_name="x")
    shape_input = (functools.reduce(lambda x, y: x * y, shape_input[:]),)
    input_x = tvm.placeholder(shape_input, name="input_x", dtype=src_dtype)
    res = ones_like_compute(input_x, y, kernel_name=kernel_name)
    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_x, res]}
    tbe.cce_build_code(auto_sch, config)
