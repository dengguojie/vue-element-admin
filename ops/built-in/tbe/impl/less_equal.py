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
less_equal
"""
from te import tvm
import te.lang.cce as tbe
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util

# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2**(-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2**(50)
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2**(26)
# define a scalar, value = 2**(-24), minimun num of float16 2**(-24)
SCALAR_MIN_FP16 = 2**(-24)
# define a scalar, value = 2**(12)
SCALAR_MUL_FP16 = 2**(12)
# define a scalar, value = 1
SCALAR_ONE = 1

# limit of input shape
MAX_SHAPE_NUM = 10000000

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("less_equal")
def less_equal_compute(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    compute for less_equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_x = input_x.dtype
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_broadcast = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                    param_name_input1="input_x",
                                                                    param_name_input2="input_y")

    if dtype_x == "float32":
        tensor_min = tbe.broadcast(tvm.const(SCALAR_MIN_FP32,
                                             dtype="float32"),
                                   shape_broadcast)
        tensor_mul = tbe.broadcast(tvm.const(SCALAR_MUL_FP32,
                                             dtype="float32"),
                                   shape_broadcast)
        tensor_mul1 = tbe.broadcast(tvm.const(SCALAR_MUL2_FP32,
                                              dtype="float32"),
                                    shape_broadcast)
        tensor_one = tbe.broadcast(tvm.const(SCALAR_ONE,
                                             dtype="float32"),
                                   shape_broadcast)
    else:
        tensor_min = tbe.broadcast(tvm.const(SCALAR_MIN_FP16,
                                             dtype="float16"),
                                   shape_broadcast)
        tensor_mul = tbe.broadcast(tvm.const(SCALAR_MUL_FP16,
                                             dtype="float16"),
                                   shape_broadcast)
        tensor_one = tbe.broadcast(tvm.const(SCALAR_ONE,
                                             dtype="float16"),
                                   shape_broadcast)

    if dtype_x in ("int8", "uint8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    input_x = tbe.broadcast(input_x, shape_broadcast)
    input_y = tbe.broadcast(input_y, shape_broadcast)
    res_max = tbe.vmax(input_x, input_y)
    res_vsub = tbe.vsub(input_y, res_max)
    res_vabs = tbe.vabs(res_vsub)
    res_min = tbe.vmin(res_vabs, tensor_min)
    res_vmul = tbe.vmul(res_min, tensor_mul)
    res_vmul1 = tbe.vmul(res_vmul, tensor_mul)

    if dtype_x == "float32":
        res_vmul2 = tbe.vmul(res_vmul1, tensor_mul1)
        res_vsub1 = tbe.vsub(res_vmul2, tensor_one)
        res_vabs1 = tbe.vabs(res_vsub1)
    else:
        res_vsub1 = tbe.vsub(res_vmul1, tensor_one)
        res_vabs1 = tbe.vabs(res_vsub1)

    res = tbe.cast_to(res_vabs1, "int8", True)

    return res


# pylint: disable=unused-variable
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def less_equal(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    Returns the truth value of (x <= y) element-wise

    Parameters
    ----------
    input_x: dict
        dict of input_x, include keys(shape and dtype)
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    output_z: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    shape_x, shape_y, shape_broadcast = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                    param_name_input1="input_x",
                                                                    param_name_input2="input_y")

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    dtype_x = dtype_x.lower()
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    dtype_y = dtype_y.lower()
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")
    shape_util.compare_tensor_dict_key(input_x, input_y, "dtype")

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_x)
    data_input_y = tvm.placeholder(shape_y, name="data_input_y", dtype=dtype_y)

    res = less_equal_compute(data_input_x, data_input_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_x, data_input_y, res]}
    tbe.cce_build_code(sch, config)
