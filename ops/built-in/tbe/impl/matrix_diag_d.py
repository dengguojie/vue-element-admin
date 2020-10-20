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
matrix_diag_d
"""
from te import tvm
import te.lang.cce as tbe
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util


# define a scalar, value = 2
SCALAR_TWO = 2

# pylint: disable=locally-disabled,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("matrix_diag_d")
def matrix_diag_d_compute(input_diagonal, input_help, output_diagonal,
                          kernel_name="matrix_diag_d"):
    """
    compute for matrix_diag_d

    Parameters
    ----------
    input_diagonal: TVM tensor
        the placeholder of input diagonal
    input_help: TVM tensor
        the placeholder of input help
    output_diagonal: dict
        dict info of output_diagonal
    kernel_name: str
        cce kernel name, default value is "matrix_diag_d"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    input_help_shape = shape_util.shape_to_list(input_help.shape)
    input_diagonal_dtype = input_diagonal.dtype

    if input_diagonal_dtype in ("int8", "uint8"):
        input_diagonal = tbe.cast_to(input_diagonal, "float16")
    res_broadcast = tbe.broadcast(input_diagonal, input_help_shape)
    res = tbe.vmul(res_broadcast, input_help)
    if input_diagonal_dtype in ("int8", "uint8"):
        res = tbe.cast_to(res, input_diagonal_dtype,
                                  f1628IntegerFlag=True)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def matrix_diag_d(input_diagonal, input_help, output_diagonal,
                  kernel_name="matrix_diag_d"):
    """
    Returns a batched diagonal tensor with a given batched diagonal values

    Parameters
    ----------
    input_diagonal: dict
        dict of input_diagonal, include keys(shape and dtype)
    input_help: dict
        dict of input_help, include keys(shape and dtype),
        Its diagonal Line value is 1 else value is 0
    output_diagonal: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "matrix_diag_d"

    Returns
    -------
    None
    """
    input_diagonal_shape = input_diagonal.get("shape")
    input_diagonal_dtype = input_diagonal.get("dtype")
    input_help_shape = input_help.get("shape")
    input_help_dtype = input_help.get("dtype")

    para_check.check_shape(input_diagonal_shape, param_name="input_diagonal")
    para_check.check_shape(input_help_shape, param_name="input_help")

    input_diagonal_shape_chgshape = list(input_diagonal_shape)
    # The penultimate dimension of the input_diagonal_shape is
    # extended for broadcast
    input_diagonal_shape_chgshape.insert(-1, 1)
    para_check.check_shape(input_diagonal_shape_chgshape, param_name="input_diagonal")

    if len(input_help_shape) < SCALAR_TWO:
        raise RuntimeError("Only the rank of input tensors >= 2 are supported!")
    if input_help_shape[-1] != input_help_shape[-2]:
        raise RuntimeError("the last two dimensions of shape_b"
                           "must be the same!")

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_diagonal_dtype = input_diagonal_dtype.lower()
    para_check.check_dtype(input_diagonal_dtype, check_list, param_name="input_diagonal")
    input_help_dtype = input_help_dtype.lower()
    para_check.check_dtype(input_help_dtype, check_list, param_name="input_help")

    input_diagonal = tvm.placeholder(input_diagonal_shape_chgshape,
                                     name="input_diagonal",
                                     dtype=input_diagonal_dtype)
    input_help = tvm.placeholder(input_help_shape, name="input_help",
                                 dtype=input_help_dtype)

    res = matrix_diag_d_compute(input_diagonal, input_help, output_diagonal,
                                kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_diagonal, input_help, res]}
    tbe.cce_build_code(sch, config)
