#!/usr/bin/env python
# coding: utf-8
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
neg
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    SCALER_NEGATIVE_ONE = -1
    SCALER_NEGATIVE_ONE_FLOAT = -1.0


# 'pylint: disable=locally-disabled,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("neg")
def neg_compute(input_x, output_y, kernel_name="neg"):
    """
    compute neg

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype.lower()
    shape = shape_util.shape_to_list(input_x.shape)

    if dtype == "int32":
        data_tmp = tbe.broadcast(-1, shape)
        res = tbe.vmul(input_x, data_tmp)
    else:
        res = tbe.vmuls(input_x, Constant.SCALER_NEGATIVE_ONE_FLOAT)

    if dtype == "int8":
        res = tbe.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def neg(input_x, output_y, kernel_name="neg"):
    """
    Computes numerical negative value element-wise, y = -x.

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same type as input
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    para_check.check_shape(shape_input, param_name="input_x")

    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32", "int32", "int8")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    shape_input = shape_util.shape_refine(shape_input)
    shape_input = (functools.reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(shape_input, name="data_input", dtype=dtype_input)

    res = neg_compute(data_input, output_y, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    tbe.cce_build_code(sch, config)
