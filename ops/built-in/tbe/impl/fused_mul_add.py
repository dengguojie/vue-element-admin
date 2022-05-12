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
fused_mul_add
"""

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from tbe.dsl import broadcast
from impl.util.util_compute import batchmatmul_elem_nd2nz
from impl.util.util_compute import batchmatmul_elem_reshape
from impl.util.util_compute import check_batchmatmul_fuse
from te.utils import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.fused_mul_add import get_op_support_info as static_get_op_support_info
from impl.dynamic.fused_mul_add import op_select_format as static_op_select_format


# 'pylint: disable=locally-disabled,unused-variable,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals,too-many-statements
# 'pylint: disable=locally-disabled,too-many-branches,unused-variable
def get_op_support_info(input0, input1, input2, output,
                        kernel_name="fused_mul_add"):
    """
    get_op_support_info
    """
    return static_get_op_support_info(input0, input1, input2, output,
                                      kernel_name="fused_mul_add")


def op_select_format(input0, input1, input2, output,
                     kernel_name="fused_mul_add"):
    """
    _division_sixteen : judge whether the last two dimensions are divided by 16
    scalar2tensor_one : convert scalar to tensor
    """
    return static_op_select_format(input0, input1, input2, output,
                                      kernel_name="fused_mul_add")


def check_format(format_input0, format_input1, format_input2):
    """
    check the format_list
    """
    list_format = [format_input0, format_input1, format_input2]
    nd_format = {"ND", "NHWC", "NCHW", "HWCN"}
    standard_format = []
    for item in list_format:
        if item in nd_format:
            standard_format.append("ND")
        else:
            standard_format.append(item)

    list_pattern = [["FRACTAL_NZ", "ND", "ND"],
                    ["ND", "FRACTAL_NZ", "ND"],
                    ["ND", "ND", "FRACTAL_NZ"],
                    ["FRACTAL_NZ", "ND", "FRACTAL_NZ"],
                    ["ND", "FRACTAL_NZ", "FRACTAL_NZ"],
                    ]
    if standard_format in list_pattern:
        format_pattern = list_pattern.index(standard_format) + 1
    else:
        format_pattern = 0

    return format_pattern


def check_ori_shape(input0, input1, input2):
    """
    check the ND shapes whether they can be broadcasted
    """
    shape_0 = list(shape_util.scalar2tensor_one(input0.get("ori_shape")))
    shape_1 = list(shape_util.scalar2tensor_one(input1.get("ori_shape")))
    shape_2 = list(shape_util.scalar2tensor_one(input2.get("ori_shape")))
    shape_input0, shape_input1, shape_max_mul = \
        shape_util.broadcast_shapes(shape_0, shape_1, param_name_input1="input0",
                                    param_name_input2="input1")
    shape_input2, shape_max_mul, shape_max_add0 = \
        shape_util.broadcast_shapes(shape_0, shape_2, param_name_input1="input0",
                                    param_name_input2="input2")


# 'pylint: disable=arguments-out-of-order
def _infer_shape_one(shape_input0, shape_input1, shape_input2, format_pattern):
    """
    shape_input0 : FRACTAL_NZ, [N,...,A,B,16,16]
    last_two_dims : [B*16, A*16]
    """
    if format_pattern == 2:
        shape_input0, shape_input1 = shape_input1, shape_input0
    if format_pattern == 3:
        shape_input0, shape_input2 = shape_input2, shape_input0

    last_two_dims = [shape_input0[-2]*shape_input0[-3],
                     shape_input0[-4]*shape_input0[-1]]
    condition2 = (len(shape_input1) == 1 and shape_input1[0] == 1)
    if not condition2:
        if len(shape_input1) == 1:
            shape_input1.insert(0, 1)
        condition0 = (shape_input1[-1] == last_two_dims[-1])
        condition1 = (shape_input1[-2] == last_two_dims[-2])

    condition5 = (len(shape_input2) == 1 and shape_input2[0] == 1)
    if not condition5:
        if len(shape_input2) == 1:
            shape_input2.insert(0, 1)
        condition3 = (shape_input2[-1] == last_two_dims[-1])
        condition4 = (shape_input2[-2] == last_two_dims[-2])

    if condition2:
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1,
                                        param_name_input1="input0",
                                        param_name_input2="input1")
    elif condition0 and not condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-4] = shape_input0[-4]
        shape_input1[-1] = shape_input0[-1]
        shape_input1[-2] = 1
        shape_input1[-3] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1,
                                        param_name_input1="input0",
                                        param_name_input2="input1")
    elif not condition0 and condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-2] = shape_input0[-2]
        shape_input1[-3] = shape_input0[-3]
        shape_input1[-4] = 1
        shape_input1[-1] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1,
                                        param_name_input1="input0",
                                        param_name_input2="input1")
    else:
        error_detail = 'shape of input1 or input0 is illegal'
        error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)

    if condition5:
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul,
                                        param_name_input1="input2",
                                        param_name_input2="shape_max_mul")
    elif condition3 and not condition4:
        shape_input2.append(1)
        shape_input2.append(1)
        shape_input2[-4] = shape_input0[-4]
        shape_input2[-1] = shape_input0[-1]
        shape_input2[-2] = 1
        shape_input2[-3] = 1
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul,
                                        param_name_input1="input2",
                                        param_name_input2="shape_max_mul")
    elif not condition3 and condition4:
        shape_input2.append(1)
        shape_input2.append(1)
        shape_input2[-2] = shape_input0[-2]
        shape_input2[-3] = shape_input0[-3]
        shape_input2[-4] = 1
        shape_input2[-1] = 1
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul,
                                        param_name_input1="input2",
                                        param_name_input2="shape_max_mul")
    else:
        error_detail = 'shape of input2 or input0 is illegal'
        error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)

    if format_pattern == 2:
        shape_input0, shape_input1 = shape_input1, shape_input0
    if format_pattern == 3:
        shape_input0, shape_input2 = shape_input2, shape_input0

    return shape_input0, shape_input1, shape_input2


def _infer_shape_two(shape_input0, shape_input1, shape_input2, format_pattern):
    """
    shape_input0 : FRACTAL_NZ, [N,...,A,B,16,16]
    last_two_dims : [B*16, A*16]
    """
    # support format_pattern == 4 or 5
    # Nz ND Nz || ND NZ NZ
    last_two_dims = [shape_input0[-2]*shape_input0[-3],
                     shape_input0[-4]*shape_input0[-1]]

    condition2 = (len(shape_input1) == 1 and shape_input1[0] == 1)
    if not condition2:
        if len(shape_input1) == 1:
            shape_input1.insert(0, 1)
        condition0 = (shape_input1[-1] == last_two_dims[-1])
        condition1 = (shape_input1[-2] == last_two_dims[-2])

    if condition2:
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
    elif condition0 and not condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-4] = shape_input0[-4]
        shape_input1[-1] = shape_input0[-1]
        shape_input1[-2] = 1
        shape_input1[-3] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
    elif not condition0 and condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-2] = shape_input0[-2]
        shape_input1[-3] = shape_input0[-3]
        shape_input1[-4] = 1
        shape_input1[-1] = 1
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
    else:
        raise RuntimeError("shape of input1 or input0 is illegal")

    shape_input2, shape_max_mul, shape_max_add0 = \
        shape_util.broadcast_shapes(shape_input2, shape_max_mul, param_name_input1="input2",
                                    param_name_input2="shape_max_mul")

    return shape_input0, shape_input1, shape_input2


def shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


@tbe_platform.fusion_manager.fusion_manager.register("fused_mul_add")
def fusion_mul_add_compute(data_input0, data_input1, data_input2,
                           output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function for ub fusion

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fused_mul_add"

    Returns
    -------
    output tensor
    """
    shape_0 = shape_util.shape_to_list(data_input0.shape)
    shape_1 = shape_util.shape_to_list(data_input1.shape)
    batch_matmul_flag_lhs = check_batchmatmul_fuse(data_input0)
    batch_matmul_flag_rhs = check_batchmatmul_fuse(data_input1)

    if batch_matmul_flag_rhs:
        data_input0, data_input1 = data_input1, data_input0
    if "para_name" in data_input0.op.attrs:
        para_name = data_input0.op.attrs["para_name"].value
        para_name += "_muladd"
    else:
        para_name = "muladd"
    batch_shape = shape_util.shape_to_list(data_input0.op.attrs["batch_shape"])
    para_dict_1 = {"format_elem": data_input1.op.attrs["format"],
                   "batch_shape": batch_shape}
    para_dict_2 = {"format_elem": data_input2.op.attrs["format"],
                   "batch_shape": batch_shape}

    if batch_matmul_flag_lhs or batch_matmul_flag_rhs:
        data_input1, shape_max = batchmatmul_elem_nd2nz(data_input0, data_input1, para_dict_1, para_name + "1")
        data_input2, _ = batchmatmul_elem_nd2nz(data_input0, data_input2, para_dict_2, para_name + "2")
        data_input1 = broadcast(data_input1, shape_max)
        data_input2 = broadcast(data_input2, shape_max)
        data_input1 = batchmatmul_elem_reshape(data_input0, data_input1, batch_shape, para_name + "1")
        data_input2 = batchmatmul_elem_reshape(data_input0, data_input2, batch_shape, para_name + "2")
        mul_result = tbe.vmul(data_input0, data_input1)
        res = tbe.vadd(mul_result, data_input2)
        res.op.attrs["batch_shape"] = batch_shape
        res.op.attrs["para_name"] = para_name
    else:
        res = fused_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name)

    return res


def fused_mul_add_compute(data_input0, data_input1, data_input2,
                          output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fuesd_mul_add"

    Returns
    -------
    output tensor
    """

    # mul
    data_input0, data_input1 = shape_broadcast(data_input0, data_input1)
    mul_result = tbe.vmul(data_input0, data_input1)

    # add
    mul_result, data_input2 = shape_broadcast(mul_result, data_input2)
    res = tbe.vadd(mul_result, data_input2)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fused_mul_add(input0, input1, input2,
                  output, kernel_name="fused_mul_add"):
    """
    function: fused for mul+add

    Parameters
    ----------
    input0: dict
         the dict of input of mul, support float16,float32,int32
    input1: dict
         the dict of input of mul, support float16,float32,int32
    input2: dict
         the dict of input of add, support float16,float32,int32
    output: dict
         the dict of output of add, support float16,float32,int32
    kernel_name: str
        cce kernel name, default value is fused_mul_add

    Returns
    -------
    None
    """
    shape_input0 = list(shape_util.scalar2tensor_one(input0.get("shape")))
    shape_input1 = list(shape_util.scalar2tensor_one(input1.get("shape")))
    shape_input2 = list(shape_util.scalar2tensor_one(input2.get("shape")))

    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()

    format_input0 = input0.get("format").upper()
    format_input1 = input1.get("format").upper()
    format_input2 = input2.get("format").upper()

    check_ori_shape(input0, input1, input2)
    format_pattern = check_format(format_input0, format_input1, format_input2)
    if format_pattern in [1, 2, 3]:
        shape_input0, shape_input1, shape_input2 = \
            _infer_shape_one(shape_input0, shape_input1,
                             shape_input2, format_pattern)
    elif format_pattern == 4:
        shape_input0, shape_input1, shape_input2 = \
            _infer_shape_two(shape_input0, shape_input1,
                             shape_input2, format_pattern)
    elif format_pattern == 5:
        shape_input1, shape_input0, shape_input2 = \
            _infer_shape_two(shape_input1, shape_input0,
                             shape_input2, format_pattern)
    else:
        shape_input0, shape_input1, shape_max_mul = \
            shape_util.broadcast_shapes(shape_input0, shape_input1, param_name_input1="input0",
                                        param_name_input2="input1")
        shape_input2, shape_max_mul, shape_max_add0 = \
            shape_util.broadcast_shapes(shape_input2, shape_max_mul, param_name_input1="input2",
                                        param_name_input2="shape_max_mul")

    data_input0 = tvm.placeholder(shape_input0,
                                  name="data_input0",
                                  dtype=dtype_input0)
    data_input1 = tvm.placeholder(shape_input1,
                                  name="data_input1",
                                  dtype=dtype_input1)
    data_input2 = tvm.placeholder(shape_input2,
                                  name="data_input2",
                                  dtype=dtype_input2)

    res = fused_mul_add_compute(data_input0, data_input1, data_input2,
                                output, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_input0, data_input1, data_input2, res)}

    tbe.cce_build_code(sch, config)
