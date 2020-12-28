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
masked_scale
"""

from te import tvm
import te.lang.cce as tbe
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def masked_scale(x, mask, y, value=1.0, kernel_name="masked_scale"):
    """
    algorithm: masked_scale
    calculating data's reciprocal, y = x * mask * value

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32
    mask: dict
        shape and dtype of input, only support int8
    value: scaler
        dtype is float, default value is 1.0
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "masked_scale"

    Returns
    -------
    None
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype")

    mask_shape = mask.get("shape")
    mask_dtype = mask.get("dtype")

    # check_dtype
    x_dtype_list = ("float16", "float32")
    para_check.check_dtype(x_dtype, x_dtype_list)

    mask_dtype_list = ("int8", "float16", "float32")
    para_check.check_dtype(mask_dtype, mask_dtype_list)

    # check_shape
    para_check.check_shape(x_shape)
    para_check.check_shape(mask_shape)

    if x_shape != mask_shape:
       error_manager_vector.raise_err_two_input_shape_invalid("masked_scale", "x", 
                                                              "mask", "shou1d be same shape")

    # check_kernel_name
    para_check.check_kernel_name(kernel_name)

    # do compute
    data_x = tvm.placeholder(x_shape, name="data_x", dtype=x_dtype)
    data_mask = tvm.placeholder(mask_shape, name="data_mask", dtype=mask_dtype)
    data_value = tvm.const(value, dtype="float32")

    # mask_dst_dtype="float16" case cast_to only support int8->float16
    data_mask_0 = data_mask
    if mask_dtype == "int8":
        mask_dtype = "float16"
        data_mask_0 = tbe.cast_to(data_mask, dtype=mask_dtype)
        
    data_mask_1 = data_mask_0
    if x_dtype != mask_dtype:
       data_mask_1 = tbe.cast_to(data_mask_0, dtype=x_dtype)

    res_vmul = tbe.vmul(data_x, data_mask_1)
    res = tbe.vmuls(res_vmul, data_value)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, data_mask, res]}
    tbe.cce_build_code(sch, config)
