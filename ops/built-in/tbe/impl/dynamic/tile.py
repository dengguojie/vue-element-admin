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
dynamic tile
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
def tile_compute(data, multiples, output_x, kernel_name="tile"):
    """
    TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    multiples : list or tuple.
        Number of the axis replicates.
    output_x: dict.
        dict of output.

    kernel_name : str.
        Cce kernel name, default value is "tile_d".

    Returns
    -------
    res
    """
    src_dtype = data.dtype
    if src_dtype == "int8":
        data = tbe.cast_to(data, "float16")
    res = tbe.broadcast(data, multiples)
    if src_dtype == "int8":
        res = tbe.cast_to(res, "int8")
    return res


# 'pylint: disable=too-many-locals,too-many-statements
@register_operator("Tile")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def tile(input_x, input_m, output_x, kernel_name="tile"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is compatible with the tensorflow operator Tile
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_m : dict
        shape and dtype of multiples
    output_x: dict
        dict of output.
    kernel_name : str.
        kernel name, default value is "tile".

    Returns
    -------
    None
    """

    input_x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int8", "uint8", "int32", "int16", "uint16", "bool")
    para_check.check_dtype(input_x_dtype, check_list, param_name="input_x")
    if input_x_dtype == "bool":
        input_x_dtype = "int8"

    # multiples : A Tensor. Must be one of the following types: int32, int64
    input_m_dtype = input_m.get("dtype").lower()
    check_list = ("int32", "int64")
    para_check.check_dtype(input_m_dtype, check_list, param_name="input_multiples")

    # multiples : A Tensor. Must be 1-D
    input_m_shape = list(input_m.get("shape"))
    if len(input_m_shape) > 1:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'tile'
        error_info['param_name'] = 'multiples'
        error_info['real_value'] = str(len(input_m_shape))
        raise RuntimeError(error_info, "In op[%s], input[%s] should be 1-D, but actually is [%s]-D." % (
            error_info['op_name'], error_info['param_name'], error_info['real_value']))

    input_x_shape = list(input_x.get("shape"))
    compile_shape = input_x_shape.copy()
    input_x_range = list(input_x.get("range"))

    dims_value = input_m_shape[0]
    if dims_value < -1:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'tile'
        error_info['param_name'] = 'multiples'
        error_info['real_value'] = str(dims_value)
        raise RuntimeError(error_info, "In op[%s], input[%s]'s shape value [%s] is invalid. "
                                       "It should be more than -1." % (
                                           error_info['op_name'], error_info['param_name'], error_info['real_value']))
    if dims_value == -1:
        dims_value = len(input_x_shape)

    if len(input_x_shape) > dims_value:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_012
        error_info['op_name'] = 'tile'
        error_info['param_name'] = 'input_x'
        error_info['real_value'] = str(len(input_x_shape))
        error_info['max_value'] = str(dims_value)
        raise RuntimeError(error_info, "In op[%s], the dimensions of input[%s] is [%s], should not be bigger than "
                                       "multiples values [%s]. " % (
                                           error_info['op_name'], error_info['param_name'], error_info['real_value'],
                                           error_info['max_value']))

    if len(input_x_shape) < dims_value:
        len_diff = dims_value - len(input_x_shape)
        input_x_shape = [1] * len_diff + input_x_shape
        input_x_range = [(1, 1)] * len_diff + input_x_range

    shape_adapt = []
    multiples_adapt = []
    range_adapt = []
    multiples_range_adapt = []
    for shape_i, range_i in zip(input_x_shape, input_x_range):
        if shape_i == 1:
            shape_adapt.append(1)
            range_adapt.append((1, 1))
            multiples_adapt.append(-1)
            multiples_range_adapt.append((1, None))
        else:
            shape_adapt.append(1)
            range_adapt.append((1, 1))
            shape_adapt.append(shape_i)
            range_adapt.append(range_i)
            multiples_adapt.append(-1)
            multiples_range_adapt.append((1, None))
            if shape_i == -1:
                multiples_adapt.append(-1)
                multiples_range_adapt.append((1, None))
            else:
                multiples_adapt.append(shape_i)
                multiples_range_adapt.append(range_i)

    input_x["shape"] = shape_adapt
    input_x["range"] = range_adapt
    input_m["shape"] = multiples_adapt
    input_m["range"] = multiples_range_adapt

    extra_params = {"disable_optimization": True}
    ins = classify([input_m, input_x], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_input_m, _input_x) in ins:
        with tbe.compute():
            shape_mul = shape_util.variable_shape([_input_m])[0]
            shape = [shape_mul[i] if shape_adapt[i] != 1 else shape_adapt[i] for i in range(len(shape_mul))]
            data = tvm.placeholder(shape, name="input_x", dtype=input_x_dtype)
            input_mul = tvm.placeholder(shape_mul, name="multiples", dtype=input_m_dtype)

            res = tile_compute(data, shape_mul, output_x, kernel_name)
            tensors.append([data, input_mul, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)

    tbe_context.get_context().add_compile_info("compile_shape", compile_shape)
