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
tile_d
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from impl.util import util_select_op_base
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# pylint: disable = unused-argument
def get_op_support_info(input_x, output_x, multiples, kernel_name="tile_d"):
    shape_x = input_x.get("shape")
    shape_x = list(shape_x)
    multiples = list(multiples)
    format_x = input_x.get("format").upper()
    format_output = output_x.get("format").upper()
    if format_x == "ND":
        axis_split_matrix = []
        if len(shape_x) < len(multiples):
            len_error = len(multiples) - len(shape_x)
            shape_x = [1]*len_error + shape_x
        for i, shape_i in enumerate(shape_x):
            multiples_i = multiples[i]
            if multiples_i == 1 and shape_i != 1:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None

    elif format_x in ("NCHW", "NHWC") and format_output == "NC1HWC0":
        shape_x = shape_x + [1]
        if format_x == "NCHW":
            multiples = [multiples[0], multiples[1] // 16, multiples[2], multiples[3], 16]
        if format_x == "NHWC":
            multiples = [multiples[0], multiples[3] // 16, multiples[1], multiples[2], 16]
        if len(shape_x) < len(multiples):
            len_error = len(multiples) - len(shape_x)
            shape_x = [1]*len_error + shape_x
        axis_split_matrix = []
        for i, shape_i in enumerate(shape_x):
            multiples_i = multiples[i]
            if multiples_i == 1 and shape_i != 1:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
# pylint: disable=locally-disabled,too-many-branches
def op_select_format(input_x, output_x, multiples, kernel_name="tile_d"):
    """TileD: to do boradcast with multiples

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str
        kernel name, default value is "tile_d".

    Returns
    -------
    param_dynamic_in_json
    """
    input_shape = list(input_x.get("shape"))
    input_format = input_x.get("format")
    inputdtype = input_x.get("dtype")
    # ND dtype
    dtype_base = ["float16", "float", "int32"]
    dtype_list = ["float16", "float", "int32", "bool"]
    # default support ND for dtype_base
    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)
    format_base_in = ["ND"] * len(dtype_base)

    # check whether support 4D to 5HD
    is_support_5hd = True
    if inputdtype == "bool":
        is_support_5hd = False
        dtype_base_out = dtype_list.copy()
        format_base_out = ["ND"] * len(dtype_list)
        format_base_in = ["ND"] * len(dtype_list)
    elif input_format not in ("NCHW", "NHWC") or len(input_shape) != 4 or len(multiples) != 4:
        is_support_5hd = False
    elif input_shape[1] != 1 or input_shape[2] != 1 or input_shape[3] != 1:
        is_support_5hd = False
    elif input_format in ("NCHW",) and multiples[1] % 16 != 0:
        is_support_5hd = False
    elif input_format in ("NHWC",) and multiples[3] % 16 != 0:
        is_support_5hd = False
    if is_support_5hd:
        dtype_base_out = dtype_base_out + dtype_base + dtype_base
        format_base_in = format_base_in + ["NCHW"] * len(dtype_base) + ["NHWC"] * len(dtype_base)
        format_base_out = format_base_out + ["NC1HWC0"] * len(dtype_base) + ["NC1HWC0"] * len(dtype_base)

    dtype_str = ','.join(dtype_base_out)
    format_input_str = ','.join(format_base_in)
    format_output_str = ','.join(format_base_out)

    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_input_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_output_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@tbe_platform.fusion_manager.fusion_manager.register("tile_d")
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
    src_dtype = data.dtype.lower()
    shape = shape_util.shape_to_list(data.shape)
    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i*multiples_i
        out_shape.append(out_shape_i)
    if src_dtype == "int8":
        data = tbe.cast_to(data, "float16")
    res = tbe.broadcast(data, out_shape)
    if src_dtype == "int8":
        res = tbe.cast_to(res, "int8")

    return res


# pylint: disable=too-many-locals
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
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        kernel name, default value is "tile_d".

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    para_check.check_shape(shape, param_name="input_x")
    para_check.check_shape(multiples, param_name="multiples")
    para_check.check_dtype(dtype.lower(), ("float16", "float32", "int32", "int8"), param_name="input_x")
    shape = list(shape)
    multiples = list(multiples)
    input_format = input_x.get("format")
    output_format = output_x.get("format")
    if input_format in ("NCHW", "NHWC") and output_format in ("NC1HWC0",):
        # branch: 4D tile to 5HD ((N, 1, 1, 1) to (N, C1, H, W, C0)) and output C is 16 align
        # change input shape from (N, 1, 1, 1) to (N, 1, 1, 1, 1)
        shape = shape + [1]
        if input_format == "NCHW":
            # change multiples from (1, C, H, W) to (1, C1, H, W, C0)
            multiples = [multiples[0], multiples[1] // 16, multiples[2], multiples[3], 16]
        else:
            # change multiples from (1, H, W, C) to (1, C1, H, W, C0)
            multiples = [multiples[0], multiples[3] // 16, multiples[1], multiples[2], 16]

    if len(shape) > len(multiples):
        error_detail = "The lenth of multiples must be greater or equal to length of input shape"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_x", \
                                                               "multiples", error_detail)
    if len(shape) < len(multiples):
        len_error = len(multiples) - len(shape)
        shape = [1]*len_error + shape

    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i*multiples_i
        out_shape.append(out_shape_i)
    para_check.check_shape(out_shape, param_name="output_x")

    shape_adapt = []
    multiples_adapt = []
    for i, shape_i in enumerate(shape):
        multiples_i = multiples[i]
        if multiples_i != 1 and shape_i != 1:
            shape_adapt.append(1)
            multiples_adapt.append(multiples_i)
            multiples_i = 1
        shape_adapt.append(shape_i)
        multiples_adapt.append(multiples_i)

    shape = shape_adapt
    multiples = multiples_adapt

    for shape_i, multiples_i in zip(shape, multiples):
        if not (shape_i == 1 or multiples_i == 1):
            error_detail = "In tile of TBE, any axis of either shape or multiples have to be 1"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_x", \
                                                               "multiples", error_detail)

    axis_not_multiple = 0
    for multiples_i in multiples:
        if multiples_i == 1:
            axis_not_multiple += 1
    if axis_not_multiple == len(multiples):
        error_detail = "In tile of TBE, the axis of multiples can't all be 1"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "multiples", error_detail)

    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    res = tile_d_compute(data, output_x, multiples, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    tbe.cce_build_code(sch, config)
