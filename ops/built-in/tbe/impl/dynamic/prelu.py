# Copyright 2021 Huawei Technologies Co., Ltd
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
dynamic prelu
"""
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_select_op_base import get_op_cal_info


# pylint: disable=locally-disabled,too-many-branches,too-many-statements,invalid-name,unused-argument,too-many-locals
def op_select_format(x, weight, y, kernel_name="prelu"):
    """ calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of x
    weight : dict
        shape and dtype of weight, should be same type as x
    y : dict
        shape and dtype of y, should be same shape and type as x
    kernel_name : str
        kernel name, default value is "prelu"
    Returns
    -------
    None.
    """

    x_shape = x.get("ori_shape")
    weight_shape = weight.get("ori_shape")

    product_version = tbe_platform.get_soc_spec("SOC_VERSION")
    if product_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_base = ["float16"]
        format_x = ["NCHW", "ND"]
        format_weight = ["ND", "ND"]
        dtype_base_out = ["float16", "float16"]
    else:
        dtype_base = ["float16", "float"]
        format_x = ["NCHW", "NCHW", "ND", "ND"]
        format_weight = ["ND", "ND", "ND", "ND"]
        dtype_base_out = ["float16", "float", "float16", "float"]

    if len(x_shape) >= 2 and len(weight_shape) >= 2 and x_shape[-1] == weight_shape[-1] \
            and x_shape[-2] == weight_shape[-2] and not (x_shape[-1] == -1 or x_shape[-2] == -1) \
            and not (weight_shape[-2] == -1 or weight_shape[-2] == -1) \
            and not (weight_shape[-2] == -2 or weight_shape[-2] == -2):
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["FRACTAL_NZ"] * len(dtype_base)
        format_weight = format_weight + ["FRACTAL_NZ"] * len(dtype_base)
    if len(x_shape) == 2 and len(weight_shape) == 1:
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["FRACTAL_NZ"] * len(dtype_base)
        format_weight = format_weight + ["ND"] * len(dtype_base)
    if not (len(weight_shape) == 3 and weight_shape[-1] == 1 and len(weight_shape) != sum(weight_shape)):
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["NC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["NC1HWC0"] * len(dtype_base)
    if not (len(weight_shape) == 4 and weight_shape[-1] == 1 and len(weight_shape) != sum(weight_shape)):
        dtype_base_out = dtype_base_out + dtype_base
        format_x = format_x + ["NDC1HWC0"] * len(dtype_base)
        format_weight = format_weight + ["NDC1HWC0"] * len(dtype_base)

    dtype_str = ','.join(dtype_base_out)
    format_x_str = ','.join(format_x)
    format_weight_str = ','.join(format_weight)

    input0 = gen_param(classify="input0", name="x", datatype=dtype_str,
                                           format=format_x_str, unknownshape_format=format_x_str)
    input1 = gen_param(classify="input1", name="weight", datatype=dtype_str,
                                           format=format_weight_str, unknownshape_format=format_weight_str)
    output0 = gen_param(classify="output0", name="y", datatype=dtype_str,
                                            format=format_x_str, unknownshape_format=format_x_str)
    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


#pylint: disable=invalid-name,too-many-branches,too-many-statements
def broadcast_inputs_shape(x, weight):
    """
    :params:
    x: dict
    weight: dict
    """
    shape_x = x.get("shape")
    format_x = x.get("format")
    shape_w = weight.get("shape")
    x_dim = len(shape_x)
    w_dim = len(shape_w)
    if format_x == "NC1HWC0":
        if w_dim != 5:
            if w_dim == 1:
                weight_shape_new = [1] * 5
            else:
                shape_list = shape_util.broadcast_shapes(shape_x, shape_w, param_name_input1="x",
                                                        param_name_input2="weight")
                shape_x, weight_shape_new = shape_list[0], shape_list[1]
        else:
            c1 = shape_x[1]
            c0 = shape_x[4]
            weight_shape_new = [1, c1, 1, 1, c0]
    elif format_x == "NDC1HWC0":
        if w_dim != 6:
            if w_dim == 1:
                weight_shape_new = [1] * 6
            else:
                shape_list = shape_util.broadcast_shapes(shape_x, shape_w, param_name_input1="x",
                                                         param_name_input2="weight")
                shape_x, weight_shape_new = shape_list[0], shape_list[1]
        else:
            c1 = shape_x[2]
            c0 = shape_x[5]
            weight_shape_new = [1, 1, c1, 1, 1, c0]
    elif format_x == "FRACTAL_NZ":
        if w_dim == 1 and shape_w[0] ==1:
            weight_shape_new = [1] * x_dim
        elif w_dim == 1:
            weight_shape_new = [1] * x_dim
            weight_shape_new[0] = shape_x[0]
            weight_shape_new[-1] = shape_x[-1]
        else:
            shape_list = shape_util.broadcast_shapes(shape_x, shape_w, param_name_input1="x",
                                                     param_name_input2="weight")
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
    elif format_x == "NHWC" and x_dim == 4:
        if (w_dim == 1 and shape_w[0] != shape_x[-1] and shape_w[0] != 1) or (w_dim not in (1, 3)):
            shape_list = shape_util.broadcast_shapes(shape_x, shape_w, param_name_input1="x",
                                                     param_name_input2="weight")
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
        elif w_dim == 1:
            weight_shape_new = [1] * x_dim
            weight_shape_new[3] = shape_x[-1]
        else:
            weight_shape_new = list(shape_w)
            weight_shape_new.insert(0, 1)
    elif x_dim == 1:
        if shape_w[0] != 1 or w_dim != 1:
            shape_list = shape_util.broadcast_shapes(shape_x, shape_w, param_name_input1="x",
                                                     param_name_input2="weight")
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
        else:
            weight_shape_new = [1]
    # input_x:DIM = 2,3,4,5,6,7...
    else:
        if (shape_w[0] != shape_x[1] and shape_w[0] != 1) or (w_dim not in (1, x_dim - 1)):
            shape_list = shape_util.broadcast_shapes(shape_x, shape_w, param_name_input1="x",
                                                     param_name_input2="weight")
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
        elif w_dim == 1:
            weight_shape_new = [1] * x_dim
        elif w_dim == x_dim - 1:
            weight_shape_new = list(shape_w)
            weight_shape_new.insert(0, 1)

    return shape_x, weight_shape_new


# pylint: disable=unused-variable,too-many-branches,too-many-statements
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
@register_operator_compute("PRelu", op_mode="dynamic", support_fusion=True)
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
    shape_weight = shape_util.shape_to_list(weight_input.shape)
    if input_x.dtype == "float16":
        scalar_zero = tvm.const(0, dtype="float16")
    else:
        scalar_zero = tvm.const(0, dtype="float32")
    if "format" in input_x.op.attrs and "format" in weight_input.op.attrs:
        format_x = input_x.op.attrs["format"].value
        format_weight = weight_input.op.attrs["format"].value
        if format_x == "FRACTAL_NZ" and format_weight != "FRACTAL_NZ": 
            target_shape = [1] * len(shape_x)
            if sum(shape_weight) != 1:
                target_shape[0] = shape_x[0]
                target_shape[-1] = shape_x[-1]
            weight_input = reshape(weight_input, target_shape)

    shape_list = shape_util.broadcast_shapes(shape_x, shape_weight, param_name_input1="input_x",
                                             param_name_input2="weight_input")
    input_x = tbe.broadcast(input_x, shape_list[2])
    weight_input = tbe.broadcast(weight_input, shape_list[2])
    val_max = tbe.vmaxs(input_x, scalar_zero)
    val_min = tbe.vmins(input_x, scalar_zero)
    val_prod = tbe.vmul(val_min, weight_input)
    res = tbe.vadd(val_max, val_prod)

    return res


# pylint: disable=too-many-locals,invalid-name
@register_operator("PRelu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def prelu(input_x, input_a, output_y, kernel_name="prelu"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_a : dict
        shape and dtype of input_a, should be same type as input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    kernel_name : str
        kernel name, default value is "prelu"
    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    input_format = input_x.get("format")
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    weight_dtype = input_a.get("dtype").lower()
    para_check.check_dtype(weight_dtype, check_list, param_name="weight")

    if weight_dtype != input_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal('prelu', 'input_dtype', 'weight_dtype',
                                                              str(weight_dtype), str(input_dtype))

    shape_x, weight_shape = broadcast_inputs_shape(input_x, input_a)
    input_x["shape"] = shape_x
    input_a["shape"] = weight_shape

    tbe_context.get_context().add_compile_info("broadcast_weight_shape", weight_shape)

    weight_range = []
    for i, _range in enumerate(input_x["range"]):
        _range = (weight_shape[i], weight_shape[i]) if weight_shape[i] != -1 else _range
        weight_range.append(_range)
    input_a["range"] = tuple(weight_range)

    ins = classify([input_x, input_a], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []

    for (_input_x, _input_a) in ins:
        with tbe.compute():
            x_shape, input_a_shape = shape_util.variable_shape([_input_x, _input_a])
            data_input = tvm.placeholder(
                x_shape, name="data_input", dtype=input_dtype)
            weight_input = tvm.placeholder(
                input_a_shape, name="weight_input", dtype=input_dtype)

            res = prelu_compute(data_input, weight_input, output_y, kernel_name)
            tensors.append([data_input, weight_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {
        "name": kernel_name,
        "tensor_list": tensors}
    tbe.build(schedules, config)
