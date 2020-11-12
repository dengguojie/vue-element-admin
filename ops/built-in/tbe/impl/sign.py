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
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.utils.error_manager import error_manager_vector

SHAPE_SIZE_LIMIT = 2147483648  # shape limit


# pylint: disable=unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("sign")
def sign_compute(input_x, output_y, kernel_name="sign"):
    """
    compute for sign
    """
    inp_dtype = input_x.dtype
    fp16_max = tvm.const(32768, dtype=inp_dtype)
    fp16_min = tvm.const(2**(-15), dtype=inp_dtype)
    data_tmp = input_x
    if inp_dtype == "float16":
        data_tmp = tbe.round_to(input_x, 0.5, -0.5)

    new_data = tbe.vmuls(data_tmp, fp16_max)
    tmp2 = tbe.vabs(new_data)
    anuminate = tbe.vadds(tmp2, fp16_min)
    rec = tbe.vrec(anuminate)
    fp16_res = tbe.vmul(new_data, rec)
    int_res = tbe.round(fp16_res)
    res = tbe.cast_to(int_res, inp_dtype)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
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
    shape = input_x.get("shape")
    para_check.check_shape(shape, param_name="input_x")

    check_list = ["float16", "float32", "int32"]
    inp_dtype = input_x.get("dtype").lower()
    if not inp_dtype in check_list:
        excepted_dtype_list = "float16, float32, int32"
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, "input_x", \
                                                                 excepted_dtype_list, inp_dtype)

    shape = shape_util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=inp_dtype)

    res = sign_compute(data, output_y, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}
    tbe.cce_build_code(sch, config)
