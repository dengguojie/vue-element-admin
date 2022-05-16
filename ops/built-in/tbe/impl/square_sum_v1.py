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
square_sum_v1
"""
from te import tvm
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.dynamic.square_sum_v1 import get_new_format_axis
from impl.dynamic.square_sum_v1 import get_op_support_info as dynamic_get_op_support_info
from impl.dynamic.square_sum_v1 import op_select_format as dynamic_op_select_format

MIN_FP32 = 2**(-126)
# min float16 value
MIN_FP16 = 2**(-24)
VALUE_ONE = 1

SHAPE_SIZE_LIMIT = 200000000


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output1, attr1, attr2=True, kernel_name="square_sum_v1"):
    """
    get_op_support_info
    """

    return dynamic_get_op_support_info(input_x, output1, attr1, attr2=True, kernel_name="square_sum_v1")


def op_select_format(input_x, output1, attr1, attr2, kernel_name="square_sum_v1"):
    """
    select format dynamically
    op_select_format support desc:
    1. input_format always support 'ND'
    2. when ori_format is 'HWCN', input_format support 'FRACTAL_Z' or 'FRACTAL_NZ' in compile_static process
        for example:
            ori:
                input_x              shape = [5,5,16,16]           format = 'HWCN'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [25,1,16,16]          format = 'FRACTAL_Z'
                output1              shape = []                    format = 'ND'
            ---------------------------------------------------------------------------
            ori:
                input_x              shape = [16,16]               format = 'ND'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [1,1,16,16]          format = 'FRACTAL_NZ'
                output1              shape = []                    format = 'ND'

    """

    return dynamic_op_select_format(input_x, output1, attr1, attr2, kernel_name="square_sum_v1")


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
@tbe_platform.fusion_manager.fusion_manager.register("square_sum_v1")
def square_sum_v1_compute(input_x, output1, axis, attr2, kernel_name="square_sum_v1", impl_mode=None):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """
    ori_dtype = input_x.dtype

    if impl_mode == "high_precision":
        input_x = tbe.cast_to(input_x, "float32")
    square_res = tbe.vmul(input_x, input_x)

    if square_res.dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        square_res = tbe.cast_to(square_res, "float32")

    sum_res = tbe.sum(square_res, axis=axis, keepdims=attr2)

    res = tbe.cast_to(sum_res, ori_dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def square_sum_v1(input_x, output1, attr1, attr2=True, kernel_name="square_sum_v1", impl_mode=None):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    ori_shape = input_x.get("ori_shape")
    x_format = input_x.get("format")
    x_ori_format = input_x.get("ori_format")
    input_dtype = dtype.lower()

    axis_d = []

    if not attr1:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    elif x_format in ["FRACTAL_NZ", "FRACTAL_Z"]:
        axis_d = get_new_format_axis(ori_shape, attr1, x_format, x_ori_format)
    else:
        axis_d = attr1

    para_check.check_shape(shape, param_name="input_x")

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)

    res = square_sum_v1_compute(data_input, output1, axis_d, attr2, kernel_name, impl_mode)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    tbe.cce_build_code(sch, config)
