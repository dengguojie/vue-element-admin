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
reciprocal
"""
import json
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm

SHAPE_SIZE_LIMIT = 2147483648  # shape limit


# pylint: disable=redefined-builtin,unused-argument
def op_select_format(input_x, output_y, kernel_name="reciprocal"):
    """
    Get support format according to input_x
    """
    shape = input_x.get("shape")
    shape_len = len(shape)
    format = input_x.get("ori_format")
    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    support_format = "ND,ND,NCHW,NCHW,NHWC,NHWC,HWCN,HWCN"
    ini_dict = {"input0": {"name": "x", "format": "ND",
                           "dtype": "float,float16"},
                "output0": {"name": "y", "format": "ND",
                            "dtype": "float,float16"}}

    # whether support format NC1HWC0、FRACTAL_Z、C1HWNCoC0
    if shape_len == 4 and format in format_4d_list:
        if format == "NCHW":
            n_dim = shape[0]
            c_dim = shape[1]
        if format == "NHWC":
            n_dim = shape[0]
            c_dim = shape[3]
        if format == "HWCN":
            n_dim = shape[3]
            c_dim = shape[2]
        # whether support format NC1HWC0
        if c_dim % 16 == 0:
            support_format += ("," + "NC1HWC0") * 2
        # whether support format FRACTAL_Z and C1HWNCoC0
        if n_dim % 16 == 0 and c_dim % 16 == 0:
            support_format += ("," + "FRACTAL_Z") * 2
            support_format += ("," + "C1HWNCoC0") * 2

    ini_dict["input0"]["format"] = support_format
    ini_dict["input0"]["dtype"] = "float,float16," *\
            (len(support_format.split(",")) // 2 - 1) + "float,float16"
    ini_dict["output0"]["format"] = support_format
    ini_dict["output0"]["dtype"] = "float,float16," *\
            (len(support_format.split(",")) // 2 - 1) + "float,float16"

    return json.dumps(ini_dict, indent=4)


@tbe_platform.fusion_manager.fusion_manager.register("reciprocal")
def reciprocal_compute(input_x, output_y, kernel_name="reciprocal"):
    if tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        dtype = input_x.dtype
        shape = shape_util.shape_to_list(input_x.shape)
        if dtype == "float16":
            input_x = tbe.cast_to(input_x, "float32")
        data_one = tbe.broadcast(tvm.const(1, "float32"), shape)
        res = tbe.vdiv(data_one, input_x)
        if dtype == "float16":
            res = tbe.cast_to(res, "float16")
    else:
        res = tbe.vrec(input_x)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def reciprocal(input_x, output_y, kernel_name="reciprocal"):
    """
    algorithm: reciprocal

    calculating data's reciprocal,y= 1 / x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is reciprocal

    Returns
    -------
    None
    """
    shape = shape_util.scalar2tensor_one(input_x.get("shape"))
    para_check.check_shape(shape, param_name="input_x")

    check_list = ["float16", "float32"]
    inp_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(inp_dtype, check_list, param_name="input_x")

    shape = shape_util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=inp_dtype)

    res = reciprocal_compute(data, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    tbe.cce_build_code(sch, config)
