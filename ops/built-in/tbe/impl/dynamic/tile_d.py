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
dynamic tile_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-statements
def tile_d_compute(data, output_x, multiples, kernel_name="tile_d"):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    output_x: dict.
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        Cce kernel name, default value is "tile_d".

    Returns
    -------
    res
    """
    shape = shape_util.shape_to_list(data.shape)
    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i * multiples_i
        out_shape.append(out_shape_i)
    if data.dtype == "int8":
        data = tbe.cast_to(data, "float16")
        res_tmp = tbe.broadcast(data, out_shape)
        res = tbe.cast_to(res_tmp, "int8")
    else:
        res = tbe.broadcast(data, out_shape)

    return res


# 'pylint: disable=too-many-locals
@register_operator("TileD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def tile_d(input_x, output_x, multiples, kernel_name="tile_d"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is different from tf.tile, tile of TBE use broadcast
    api, and only support that at least an axis in shape is 1.The '1' axis
    is to be multipled.
    For example, if shape = [51, 1] and multiples = [1, 77], after computation,
    the output shape will be [51, 77].
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.
    2. The type of kernel_name is not string.
    3. The shape is neither list nor tuple.
    4. The dtype is not float32, float16, or int32.
    5. All of the axises of the multiples is 1.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple
        Number of the axis replicates.
    kernel_name : str.
        kernel name, default value is "tile_d".

    Returns
    -------
    None
    """
    # Check not all shape is 1
    origin_multiples = list(multiples)
    axis_not_multiple = 0
    for multiples_i in origin_multiples:
        if multiples_i == 1:
            axis_not_multiple += 1
    if axis_not_multiple == len(origin_multiples):
        error_manager_vector.raise_err_input_param_range_invalid("tile_d", "axis_not_multiple",
                                                                 "1", str(len(origin_multiples) - 1),
                                                                 str(axis_not_multiple))

    # Check support dtype
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "int8", "bool")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    if input_dtype == "bool":
        input_dtype = "int8"

    input_range = list(input_x.get("range"))
    input_shape = list(input_x.get("shape"))

    # Write unknown dim index into tiling_info
    tiling_info = []
    for idx, shape_i in enumerate(input_shape):
        if shape_i == -1:
            tiling_info.append(idx)
    tiling_info.insert(0, len(tiling_info))

    # Check len between input and multiples, the multiples len must not be less than input len
    if len(input_shape) > len(multiples):
        error_manager_vector.raise_err_input_param_range_invalid("tile_d", "input_x_shape",
                                                                 "1", str(len(multiples)),
                                                                 str(input_shape))
    if len(input_shape) < len(multiples):
        len_diff = len(multiples) - len(input_shape)
        input_shape = [1] * len_diff + input_shape
        input_range = [(1, 1)] * len_diff + input_range

    shape_adapt = []
    multiples_adapt = []
    range_adapt = []
    multiples_align = []
    for shape_i, multiples_i, range_i in zip(input_shape, multiples, input_range):
        if multiples_i != 1 and shape_i != 1:
            shape_adapt.extend([1, shape_i])
            range_adapt.extend([(1, 1), range_i])
            multiples_adapt.extend([multiples_i, 1])
            multiples_align.extend([multiples_i, shape_i])
        else:
            shape_adapt.append(shape_i)
            range_adapt.append(range_i)
            multiples_adapt.append(multiples_i)
            multiples_align.append(multiples_i * shape_i)

    tiling_info.extend(shape_adapt)
    tiling_info.extend(multiples_align)

    input_x["shape"] = input_x["ori_shape"] = shape_adapt
    input_x["range"] = range_adapt

    extra_params = {"disable_optimization": True}
    ins = classify([input_x], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            shape = shape_util.variable_shape([_input_x])[0]
            data = tvm.placeholder(shape, name="data", dtype=input_dtype)
            res = tile_d_compute(data, output_x, multiples_adapt, kernel_name)
            tensors.append([data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)

    tbe_context.get_context().add_compile_info("tiling_info", tiling_info)
