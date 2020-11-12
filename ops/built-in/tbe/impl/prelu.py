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
prelu
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from te import tvm


def reshape(tensor_in, new_shape):
    """
    :params:
    :input: tensor to be reshaped
    :new_shape: shape after input tensor reshaped
    :return: reshape tensor
    """
    def _nd2nz_compute(tensor, indices):
        axis_0, axis_1, axis_2, axis_3 = indices
        return tensor(axis_0 * 16 + axis_3)
    return tvm.compute(new_shape, lambda *indices: _nd2nz_compute(tensor_in, indices), name='reshape')


# pylint: disable=unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("prelu")
def prelu_compute(input_x, weight_input, output_y, kernel_name="prelu"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    weight_input : TVM tensor
        the placeholder of weight_input
    kernel_name : str
        kernel name, default value is "prelu"

    Returns
    -------
    output tensor
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    if input_x.dtype == "float16":
        scalar_zero = tvm.const(0, dtype="float16")
    else:
        scalar_zero = tvm.const(0, dtype="float32")
    val_max = tbe.vmaxs(input_x, scalar_zero)
    val_min = tbe.vmins(input_x, scalar_zero)
    if "format" in input_x.op.attrs:
        format_x = input_x.op.attrs["format"].value
        shape_weight =  shape_util.shape_to_list(weight_input.shape)
        if format_x == "FRACTAL_NZ":
            target_shape = [1] * len(shape_x)
            if sum(shape_weight) != 1:
                target_shape[0] = shape_x[0]
                target_shape[-1] = shape_x[-1]
            weight_input = reshape(weight_input, target_shape)
    weight_input = tbe.broadcast(weight_input, shape_x)
    val_prod = tbe.vmul(val_min, weight_input)
    res = tbe.vadd(val_max, val_prod)
    return res


# pylint: disable=too-many-locals,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def prelu(input_x, input_A, output_y, kernel_name="prelu"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_A : dict
        shape and dtype of input_A, should be same type as input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    kernel_name : str
        kernel name, default value is "prelu"
    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_format = input_x.get("format")
    input_dtype = dtype.lower()
    para_check.check_shape(shape, param_name="x")

    check_list = ("float16", "float32")

    para_check.check_dtype(input_dtype, check_list, param_name="x")
    weight_shape = input_A.get("shape")
    weight_dtype = input_A.get("dtype").lower()
    para_check.check_shape(weight_shape, param_name="weight")
    para_check.check_dtype(weight_dtype, check_list, param_name="weight")

    if weight_dtype != input_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal('prelu', 'input_dtype', 'weight_dtype',
                                                              str(weight_dtype), str(input_dtype))

    weight_dim = len(weight_shape)
    feature_dim = len(shape)

    if input_format == "NC1HWC0":
        if weight_dim == 5 and (shape[1] != weight_shape[1] or
                                shape[4] != weight_shape[4]):
            detail = "weight dim only support two values: 1 or 5, when feature_dim is 5, channel(C1/C0) dim" \
                     " for input_x and input_A must be matched, " \
                     "while feature [C1, C0]:[%d, %d], weight [C1, C0]:[%d, %d]" % (
                         shape[1], shape[4], weight_shape[1], weight_shape[4])
            error_manager_vector.raise_err_two_input_shape_invalid('prelu', 'input_x', 'input_A', detail)
        if weight_dim != 5:
            if weight_dim == 1 and weight_shape[0] == 1:
                weight_shape_new = [1] * 5
            else:
                detail = "weight_dim only support two values: 1 or 5," \
                    " when feature_dim is 5(NC1HWC0) and weight_dim is not equal to 5, both weight_shape[0] " \
                    "and weight_dim must be 1, while weight_shape[0] is {0}, weight_dim is {1}".format(
                        weight_shape[0], weight_dim)
                error_manager_vector.raise_err_input_shape_invalid('prelu', 'input_A', detail)
        else:
            weight_shape_new = [1] * feature_dim
            weight_shape_new[1] = weight_shape[1]
            weight_shape_new[-1] = weight_shape[-1]
    elif input_format == "FRACTAL_NZ":
        if weight_dim == 1 and weight_shape[0] == 1:
            weight_shape_new = [1] * feature_dim
        else:
            weight_shape_new = [1] * feature_dim
            weight_shape_new[0] = shape[0]
            weight_shape_new[-1] = shape[-1]
    elif input_format == "NHWC" and feature_dim == 4:
        if (weight_shape[0] != shape[-1] and weight_shape[0] != 1) or weight_dim != 1:
            detail = "channel dim of input_x and input_A must be matched, and weight_dim must be 1, " \
                     "while channel dim of input_A is {0}, channel dim of input_x is {1}, " \
                     "weight_dim is {2}".format(weight_shape[0], shape[-1], weight_dim)
            error_manager_vector.raise_err_two_input_shape_invalid('prelu', 'input_x', 'input_A', detail)

        weight_shape_new = [1] * feature_dim
        weight_shape_new[3] = weight_shape[0]
    elif feature_dim == 1:
        if weight_shape[0] != 1 or weight_dim != 1:
            detail = "when feature_dim is 1, both weight_shape[0] and weight_dim must be 1, " \
                     "while weight_shape[0] is {0}, weight_dim is {1}".format(weight_shape[0], weight_dim)
            error_manager_vector.raise_err_input_shape_invalid('prelu', 'input_A', detail)
        weight_shape_new = [1]
    # input_x:DIM = 2,3,4,5,6,7...
    else:
        if (weight_shape[0] != shape[1] and weight_shape[0] != 1) or weight_dim != 1:
            detail = "channel dim of input_x and input_A must be matched, and weight_dim must be 1, " \
                     "while channel dim of input_A is {0}, channel dim of input_x is {1}," \
                     " weight_dim is {2}".format(weight_shape[0], shape[1], weight_dim)
            error_manager_vector.raise_err_two_input_shape_invalid('prelu', 'input_x', 'input_A', detail)
        weight_shape_new = [1] * feature_dim
        weight_shape_new[1] = weight_shape[0]

    if len(weight_shape_new) == sum(weight_shape_new):
        weight_shape_new = [1]
        total_calc_num = 1
        for i, _ in enumerate(shape):
            total_calc_num = total_calc_num * shape[i]
        shape_new = [total_calc_num]
        data_input = tvm.placeholder(
            shape_new, name="data_input", dtype=input_dtype)
        weight_input = tvm.placeholder(
            weight_shape_new, name="weight_input", dtype=input_dtype)
        weight_input1 = tbe.broadcast(
            weight_input, shape_new, output_dtype=input_dtype)
    else:
        data_input = tvm.placeholder(
            shape, name="data_input", dtype=input_dtype)
        weight_input = tvm.placeholder(
            weight_shape_new, name="weight_input", dtype=input_dtype)
        weight_input1 = tbe.broadcast(
            weight_input, shape, output_dtype=input_dtype)

    res = prelu_compute(data_input, weight_input1, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_input, weight_input, res]
    }

    tbe.cce_build_code(sch, config)
