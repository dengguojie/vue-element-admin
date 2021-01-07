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
sign
"""
from functools import reduce as reduceIns

import te.lang.cce as tbe
from te import tvm
import te.lang.base as tbe_base
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
import te.platform as tbe_platform
from topi import generic


# pylint: disable=unused-argument,redefined-argument-from-local
def sign_compute(input_x, output_y, kernel_name="sign"):
    """
    compute for sign
    """
    inp_dtype = input_x.dtype
    fp16_max = tvm.const(32768, dtype=inp_dtype)
    fp16_min = tvm.const(2 ** (-15), dtype=inp_dtype)
    data_tmp = input_x
    if inp_dtype == "float16":
        data_tmp = tbe.round_to(input_x, 0.5, -0.5)

    if inp_dtype == "int32":
        data_tmp = tbe.cast_to(data_tmp, "float16")

    new_data = tbe.vmuls(data_tmp, fp16_max)
    tmp2 = tbe.vabs(new_data)
    anuminate = tbe.vadds(tmp2, fp16_min)
    rec = tbe.vrec(anuminate)
    fp16_res = tbe.vmul(new_data, rec)

    if not tbe_platform.api_check_support("te.lang.cce.round", "float32"):
        fp16_res = tbe.cast_to(fp16_res, "float16")

    int_res = tbe.round(fp16_res)

    res = tbe.cast_to(int_res, inp_dtype)

    return res


@tbe_base.register_operator("Sign")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sign(input_x, output_y, kernel_name="sign"):
    """
                                 x*32768
    algrithm: sign = round(-------------------------)
                            2 ** (-15) + |x*32768|

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sign

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype")
    check_list = ("float16", "float32", "int32")
    input_dtype = dtype.lower()
    check_dtype(input_dtype, check_list, param_name="input_x")
    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)
    for (input_x,) in ins:
        with tbe_base.compute():
            x_shape = variable_shape([input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype=input_dtype,
                                         name="data_input")
            res = sign_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
