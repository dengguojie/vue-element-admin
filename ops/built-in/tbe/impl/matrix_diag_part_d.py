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
matrix_diag_part_d
"""

from te import tvm
import te.lang.cce as tbe
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util

# define a scaler, value = -2
SCALER_NEGATIVE_TWO = -2


# pylint: disable=locally-disabled,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("matrix_diag_part_d")
def matrix_diag_part_d_compute(input_diagonal, input_help, output_diagonal,
                               kernel_name="matrix_diag_part_d"):
    """
    compute for matrix_diag_part_d

    Parameters
    ----------
    input_diagonal: TVM tensor
        the placeholder of input diagonal
    input_help: TVM tensor
        the placeholder of input help
    output_diagonal: dict
        dict of output_diagonal
    kernel_name: str
        cce kernel name, default value is "matrix_diag_part_d"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_input_diagonal = shape_util.shape_to_list(input_diagonal.shape)
    dtype_input_diagonal = input_diagonal.dtype

    res_vmul = tbe.vmul(input_diagonal, input_help)
    if shape_input_diagonal[-2] < shape_input_diagonal[-1]:
        if dtype_input_diagonal == "int32":
            res_vmul = tbe.cast_to(res_vmul, "float32")
        res = tbe.sum(res_vmul, -1)
        if dtype_input_diagonal == "int32":
            res = tbe.cast_to(res, "int32")
    else:
        res = tbe.sum(res_vmul, SCALER_NEGATIVE_TWO)

    if dtype_input_diagonal in ("int8", "uint8"):
        res = tbe.cast_to(res, dtype_input_diagonal,
                                  f1628IntegerFlag=True)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def matrix_diag_part_d(input_diagonal, input_help,
                       output_diagonal, kernel_name="matrix_diag_part_d"):
    """
    Returns the batched diagonal part of a batched tensor

    Parameters
    ----------
    input_diagonal: dict
        dict of input_diagonal, include keys(shape and dtype)
    input_help: dict
        dict of help Matrix, Its Diagonal Line value is 1 else value is 0
    output_diagonal: dict
        dict of output
    kernel_name: str
        cce kernel name, default value is "matrix_diag_part_d"

    Returns
    -------
    None
    """
    shape_input_diagonal = input_diagonal.get("shape")
    dtype_input_diagonal = input_diagonal.get("dtype")
    shape_input_help = input_help.get("shape")
    dtype_input_help = input_help.get("dtype")

    para_check.check_shape(shape_input_diagonal, param_name="input_diagonal")
    para_check.check_shape(shape_input_help, param_name="input_help")

    if len(shape_input_diagonal) < 2:
        raise RuntimeError("Input tensors of rank>=2 are supported!")
    if list(shape_input_diagonal) != list(shape_input_help):
        raise RuntimeError("the shape of data must be equal!")

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    dtype_input_diagonal = dtype_input_diagonal.lower()
    para_check.check_dtype(dtype_input_diagonal, check_list, param_name="input_diagonal")
    dtype_input_help = dtype_input_help.lower()
    para_check.check_dtype(dtype_input_help, check_list, param_name="input_help")

    data_input_diagonal = tvm.placeholder(shape_input_diagonal,
                                          name="data_input_diagonal",
                                          dtype=dtype_input_diagonal)
    data_input_help = tvm.placeholder(shape_input_help, name="data_input_help",
                                      dtype=dtype_input_help)

    res = matrix_diag_part_d_compute(data_input_diagonal, data_input_help,
                                     output_diagonal, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_diagonal, data_input_help, res]}
    tbe.cce_build_code(sch, config)
