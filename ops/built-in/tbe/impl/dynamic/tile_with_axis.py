# Copyright 2021Huawei Technologies Co., Ltd
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
tile_with_axis
"""
# 'pylint: disable=import-error
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=C0103
@register_operator_compute("TileWithAxis", op_mode="dynamic", support_fusion=False)
def tile_with_axis_compute(x, shape_y):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    shape_y: tuple or list.
        The shape of output.

    Returns
    -------
    res the compute results
    """
    res = tbe.broadcast(x, shape_y)
    return res


# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements
# 'pylint: disable=C0103
@register_operator("TileWithAxis")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def tile_with_axis(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    algorithm: tile.
    Expanding the input tensor according to a specified dimension,
    and the expansion multiple is specified by the tiles param.
    For example, tiling [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11,
    12]]], which shape is (2, 3, 2), by axis:1 and tiles:2 produces
    [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]]
    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    axis: int
         The index of the axis to tile
    tiles: int
        The number of copies (tiles) of the blob to output.
    kernel_name : str
        kernel name, default value is "tile_with_axis"

    Returns
    -------
    tik_instance
    """
    # check dtype
    dtype_x = input_x.get("dtype").lower()
    check_list = ["int8", "int16", "int32", "uint8", "uint16", "uint32", "float16", "float32"]
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")

    # check tiles
    if tiles <= 0:
        error_manager_vector.raise_err_input_value_invalid("tile_with_axis", "tiles", "more than 1", str(tiles))

    # check shape for 5HD
    shape_x = list(input_x.get("shape"))
    shape_range_x = input_x.get("range")
    shape_x_len = len(shape_x)

    input_format = input_x.get("format")
    if input_format == "NC1HWC0":
        shape_x_ori = input_x.get("ori_shape")
        ori_format = input_x.get("ori_format")
        length_x_ori = len(shape_x_ori)
        if ori_format not in ("NCHW", "NHWC"):
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "input_x's ori_format \
                                                          is invalid for 5D Tensor")
        if shape_x_len != 5:
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "input_x's shape is \
                                                          invalid for 5D Tensor")
        if length_x_ori != 4:
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "input_x's ori_shape \
                                                          is invalid for 5D Tensor")
        axis = shape_util.axis_check(length_x_ori, axis)
        axis = shape_util.axis_transform_5d(axis, ori_format)
        if axis in (1, 4):
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "axis is invalid for 5D Tensor")
    else:
        if axis >= shape_x_len or axis < -shape_x_len:
            error_manager_vector.raise_err_input_value_invalid("tile_with_axis", "axis",
                                                               "in range of [ {} , {} ]".format(-shape_x_len, \
                                                               shape_x_len - 1), str(axis))
        if axis < 0:
            axis += shape_x_len

    # modify output shape
    if axis < 0:
        axis += shape_x_len

    shape_y = shape_x.copy()

    ori_axis_value = shape_x[axis]
    tbe_context.get_context().add_compile_info("ori_axis_value", ori_axis_value)
    tbe_context.get_context().add_compile_info("attr_axis", axis)
    tbe_context.get_context().add_compile_info("attr_tiles", tiles)
    if shape_x[axis] != 1:
        shape_x.insert(axis, 1)
        shape_range_x.insert(axis, [1, 1])
        shape_y.insert(axis, tiles)
    else:
        shape_y[axis] = tiles

    input_x["shape"] = shape_x
    input_x["range"] = shape_range_x

    # classify
    schedules, tensors = [], []

    extra_params = {"disable_optimization": True}
    ins = classify([input_x], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)

    for (x_,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([x_])[0]

            if ori_axis_value != 1:
                shape_y[axis + 1] = shape_x[axis + 1]

            for i, shape_value in enumerate(shape_y):
                if shape_value == -1:
                    shape_y[i] = shape_x[i]

            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)

            if tiles > 1:
                res = tile_with_axis_compute(data_x, shape_y)
            else:
                data_zero = tvm.const(0, dtype=dtype_x)
                zero_broadcast = tbe.broadcast(data_zero, shape_x)
                res = tbe.vadd(data_x, zero_broadcast)

            tensors.append([data_x, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
