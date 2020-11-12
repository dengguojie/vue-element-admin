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
diag_d
"""

import te.lang.cce as tbe
from te import tvm
import te.platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# pylint: disable = unused-argument
def get_op_support_info(x, assist, y, kernel_name="diag_d"):
    format_x = x.get("format").upper()
    shape_x_len = len(x.get("shape"))
    if format_x == "ND":
        axis_split_matrix=[]
        for i in range(0, shape_x_len):
            split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
            axis_split_matrix.append(split_0)
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# pylint: disable = locally-disabled,invalid-name,unused-argument,no-member
@tbe_platform.fusion_manager.fusion_manager.register("diag_d")
def diag_d_compute(x, assit, y, kernel_name="diag_d"):
    """
    diag_d compute
    calculating diag_d(x,help):
    returns a diagonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diagonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    assit: TVM tensor
        the placeholder of assit
    y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "diag_d"

    Returns
    -------
    res: TVM tensor
        the result of diag compute
    """
    list_shape = shape_util.shape_to_list(assit.shape)
    x_broad = tbe.broadcast(x, list_shape)
    res = tbe.vmul(x_broad, assit)

    return res

# pylint: disable =too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                           para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def diag_d(x, assist, y, kernel_name="diag_d"):
    """
    algorithm: diag_d
    calculating diag_d(x,help):
    returns a diagonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diagonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of x
    assist: dict
        dict with keys(shape and dtype) of assist
    y: dict
        dict with keys(shape and dtype) of y
    kernel_name: str
        kernel name, default value is "diag"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype = x.get("dtype")

    if len(shape_x) > 4:
        error_detail = "length of x'shape should be less than 5 but got: %d" %len(shape_x)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    shape_help = assist.get("shape")
    dtype_help = assist.get("dtype")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_help, param_name="assist")

    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(dtype.lower(), check_list, param_name="x")
    para_check.check_dtype(dtype_help.lower(), check_list, param_name="assist")

    shape_list = shape_util.broadcast_shapes(shape_x, shape_help, param_name_input1="x", param_name_input2="assist")
    for i, element in enumerate(shape_x):
        if element != shape_help[i] or \
                element != shape_help[i + len(shape_x)] or \
                len(shape_help) != 2 * len(shape_x):
            raise RuntimeError(
                "shape mismatch of x and assist : "
                "the correct shapes should be "
                "x.shape = [D1,...,Dn],"
                "assist.shape = [D1,...,Dn,D1,...Dn]")
    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=dtype.lower(), name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_help.lower(),
                             name="data_y")
    res = diag_d_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    tbe.cce_build_code(sch, config)
