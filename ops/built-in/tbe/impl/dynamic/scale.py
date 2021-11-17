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
dynamic scale
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import get_current_build_config


def param_scale_check(shape_x, shape_scale, kernel_name="scale"):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if not (length_scale == 1 and shape_scale[0] == 1):
        if length_x != length_scale:
            error_detail = "length_x and length_scale must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "length_x",
                                                                   "length_scale", error_detail)

        for i in range(length_scale):
            if shape_scale[i] != shape_x[i] and shape_scale[i] != 1:
                error_detail = "the inputs could not be broadcast together"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "shape_x",
                                                                       "shape_scale", error_detail)


def get_param_scale_shape(shape_x, shape_scale):
    """
    Function to calculate the shape of scale.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_scale)

    return shape


def _check_dtype(input_dtype, name):
    """
    Function to check dtype of input data.

    Parameters
    ----------

    input_dtype: str
        dtype of input data
    Returns
    -------
    None
    """

    vadd_support = tbe_platform.api_check_support("te.lang.cce.vadd", "float32")
    if not vadd_support:
        if input_dtype == "float32":
            rule_desc = "float32 is not support in HISI"
            error_manager_vector.raise_err_check_params_rules("scale", rule_desc, "input_dtype", input_dtype)
        para_check.check_dtype(input_dtype, ["float16"], param_name=name)
    else:
        para_check.check_dtype(input_dtype, ["float16", "float32"], param_name=name)


# 'pylint: disable=too-many-branches,too-many-arguments
def _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob, kernel_name="scale"):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes:
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if (axis >= length_x) or (axis < (-length_x)):
        error_detail = "axis should be greater than the length of shape_x"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "axis", "shape_x",
                                                               error_detail)

    if num_axes < -1:
        expected_value = "greater than -1"
        real_value = "less than or equal -1"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "num_axes",
                                                           expected_value, real_value)

    if axis < 0:
        axis_ = length_x + axis
    else:
        axis_ = axis

    # from blob
    if scale_from_blob:
        if num_axes == -1:
            scale_num = length_x - axis_
            if length_scale != scale_num:
                error_detail = "length_scale and scale_num must be equal"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "length_scale",
                                                                       "scale_num", error_detail)
            for i in range(scale_num):
                if shape_x[axis_ + i] != shape_scale[i]:
                    error_detail = "Dimensions shape_x and shape_scale must be equal"
                    error_manager_vector.raise_err_two_input_shape_invalid(kernel_name,
                                                                           "shape_x[axis_ + i]", "shape_scale[i]",
                                                                           error_detail)

        if num_axes == 0:
            if length_scale != 1 or shape_scale[0] != 1:
                error_detail = "scale must be a scalar"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "length_scale",
                                                                       "shape_scale", error_detail)
        if num_axes > 0:
            num_axis = axis_ + num_axes
            if num_axis > length_x:
                error_detail = "scale shape extends x shape when applied"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "num_axis",
                                                                       "length_x", error_detail)
            if length_scale != num_axes:
                error_detail = "length_scale and num_axes must be equal"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "length_scale",
                                                                       "num_axes", error_detail)

    # from bottom
    if not scale_from_blob:
        if not (length_scale == 1 and shape_scale[0] == 1):
            scale_num = axis_ + length_scale
            if scale_num > length_x:
                error_detail = "scale shape extends x shape when applied"
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "scale_num",
                                                                       "length_x", error_detail)
            for i in range(length_scale):
                if shape_x[axis_ + i] != shape_scale[i]:
                    error_detail = "Dimensions shape_x and shape_scale must be equal"
                    error_manager_vector.raise_err_two_input_shape_invalid(kernel_name,
                                                                           "shape_x[axis_ + i]", "shape_scale[i]",
                                                                           error_detail)


def get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob):
    """
    Function to calculate shape of scale.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape
    axis_ : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes:
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    shape: list or tuple
        the shape of scale
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)
    if scale_from_blob:
        if num_axes == -1:
            shape_left = [1] * axis_
            shape = shape_left + list(shape_scale)
        elif num_axes == 0:
            shape = [1] * length_x
        else:
            left_length = length_x - num_axes - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right
    else:
        if length_scale == 1 and shape_scale[0] == 1:
            shape = [1] * length_x
        else:
            left_length = length_x - length_scale - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right

    return shape


# 'pylint: disable=too-many-locals,redefined-argument-from-local,invalid-name
def get_fusion_params(x_tensor, scale_tensor, bias_tensor, y):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x_tensor : tensor of input data
    scale_tensor : tensor of input data
    bias_tensor : tensor of input data
    y : dict of output data
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    in_l1_flag_list = []
    in_valid_shape_list = []
    in_slice_offset_list = []
    in_select_read_flag_list = []
    is_l1_depth_fusion = False

    input_tensor = [x_tensor, scale_tensor, bias_tensor]
    for x_tensor in input_tensor:
        if x_tensor is not None:
            l1_fusion_type = -1
            if not get_current_build_config("enable_op_prebuild"):
                l1_fusion_type = x_tensor.op.attrs["L1_fusion_type"].value \
                    if "L1_fusion_type" in x_tensor.op.attrs else -1
                if l1_fusion_type == 1:
                    error_manager_vector.raise_err_specific_reson("scale",
                                                                  "Scale does not support l1 width fusion")
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
            in_l1_flag = x_tensor.op.attrs["addr_type"].value == 1 \
                if "addr_type" in x_tensor.op.attrs else False
            in_l1_flag_list.append(in_l1_flag)
            in_valid_shape = x_tensor.op.attrs["valid_shape"] \
                if "valid_shape" in x_tensor.op.attrs else []
            in_valid_shape_list.append(in_valid_shape)
            in_slice_offset = x_tensor.op.attrs["slice_offset"] \
                if "slice_offset" in x_tensor.op.attrs else []
            in_slice_offset_list.append(in_slice_offset)
            in_select_read_flag = x_tensor.op.tag == "read_select_5d"
            in_select_read_flag_list.append(in_select_read_flag)

    l1_fusion_type = 0 if is_l1_depth_fusion is True else -1
    if l1_fusion_type != -1 and y.get("format").upper() != 'NC1HWC0':
        shape_rule = "the input format must be 5HD when l1 fusion"
        error_manager_vector.raise_err_check_params_rules(
            "scale", shape_rule, "x", y.get("format").upper())

    out_l1_flag = False
    out_valid_shape = []
    out_slice_offset = []
    out_select_write_flag = False
    if y is not None:
        out_l1_flag = y.get("addr_type", 0) == 1
        out_valid_shape = y.get("valid_shape", [])
        out_slice_offset = y.get("slice_offset", [])
        out_select_write_flag = bool(out_valid_shape)

    fusion_params = {
        "is_l1fusion": is_l1_depth_fusion,
        "l1_fusion_type": l1_fusion_type,
        "in_l1_flag": in_l1_flag_list,
        "in_select_read_flag": in_select_read_flag_list,
        "in_valid_shape": in_valid_shape_list,
        "in_slice_offset": in_slice_offset_list,
        "out_l1_flag": out_l1_flag,
        "out_select_write_flag": out_select_write_flag,
        "out_valid_shape": out_valid_shape,
        "out_slice_offset": out_slice_offset
    }
    return fusion_params


# 'pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_compute(x, scale):
    """
    algorithm: Scale
    y = scale*x

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype_x == "float16":
            is_cast = True
            x = tbe.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = tbe.cast_to(scale, 'float32')

    shape_x = shape_util.shape_to_list(x.shape)
    scale_broad = tbe.broadcast(scale, shape_x)

    res = tbe.vmul(x, scale_broad)

    if is_cast:
        res = tbe.cast_to(res, dtype_x)

    return res


# 'pylint: disable=invalid-name,redefined-outer-name,too-many-locals
def _fused_scale_bias_compute(x, scale, bias):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data
    bias : TVM tensor
        contains bias data
    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype
    dtype_bias = bias.dtype

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype_x == "float16":
            is_cast = True
            x = tbe.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = tbe.cast_to(scale, 'float32')
        if dtype_bias == "float16":
            bias = tbe.cast_to(bias, 'float32')

    shape_x = shape_util.shape_to_list(x.shape)
    shape_scale = shape_util.shape_to_list(scale.shape)
    shape_x, shape_scale, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                  shape_scale,
                                                                  param_name_input1="x",
                                                                  param_name_input2="scale")
    x_broad = tbe.broadcast(x, shape_max)

    scale_broad = tbe.broadcast(scale, shape_max)
    bias_broad = tbe.broadcast(bias, shape_max)

    res_tmp = tbe.vmul(x_broad, scale_broad)
    res = tbe.vadd(res_tmp, bias_broad)

    if is_cast:
        res = tbe.cast_to(res, dtype_x)

    return res


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@register_operator_compute("Scale", op_mode="dynamic", support_fusion=False)
def scale_compute(x, scale, bias, y, axis, num_axes, scale_from_blob, kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data
    bias : TVM tensor
        contains bias data
    y : dict
        dict of output,
        A Tensor for output, should be same shape and type as x.
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes: int
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.
    kernel_name : str
        kernel name, default value is "scale"

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """
    tmp_y = {}
    tmp_y["addr_type"] = 0
    tmp_y["valid_shape"] = []
    tmp_y["slice_offset"] = []
    fuse_y = tmp_y if y is None else y
    fusion_params = get_fusion_params(x, scale, bias, fuse_y)

    res = None
    if bias is not None:
        res = _fused_scale_bias_compute(x, scale, bias)
    else:
        res = _fused_scale_compute(x, scale)

    res.op.attrs["ele_fusion_params"] = fusion_params
    res.op.attrs["L1_fusion_type"] = fusion_params["l1_fusion_type"]

    return res


# 'pylint: disable=too-many-locals,no-member,invalid-name,too-many-statements,line-too-long
@register_operator("Scale")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scale(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True, kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : dict
        dict of input, A Tensor for input data.
    scale : dict
        dict of scale,
        A Tensor for scaling factor, to scale the input data.
    bias : dict
        dict of bias,
        A Tensor for bias, to shift to the input data.
    y : dict
        dict of output,
        A Tensor for y, should be same shape and type as x.
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes: int
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.
    kernel_name : str
        kernel name, default value is "scale"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    _check_dtype(dtype_x.lower(), "input_x")

    shape_scale = scale.get("shape")
    dtype_scale = scale.get("dtype")
    _check_dtype(dtype_scale.lower(), "input_scale")

    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        dtype_bias = bias.get("dtype")
        para_check.check_shape(shape_bias, param_name="input_bias")
        _check_dtype(dtype_bias.lower(), "input_bias")

    shape_x_ori = x.get("ori_shape")
    length_x_ori = len(shape_x_ori)

    shape_scale_new = []
    shape_bias_new = []

    if length_x_ori == 4:
        param_scale_check(shape_x, shape_scale)
        shape_scale_new.append(get_param_scale_shape(shape_x, shape_scale))
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new
    else:
        _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob)

        length_x = len(shape_x)
        if axis < 0:
            axis_ = length_x + axis
        else:
            axis_ = axis

        shape_scale_new = get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob)
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new

    shape_bias_new = shape_scale_new
    tbe_context.get_context().add_compile_info("_boardcast_scale_shape", shape_scale_new)
    scale["shape"] = shape_scale_new
    bias["shape"] = shape_scale_new
    scale_range = []
    for i, _range in enumerate(x["range"]):
        _range = (shape_scale_new[i], shape_scale_new[i]) if shape_scale_new[i] != -1 else _range
        scale_range.append(_range)
    scale["range"] = tuple(scale_range)

    ins = classify([x, scale], OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedules, tensors = [], []
    for (x_, scale_) in ins:
        with tbe.compute():
            shape_x, shape_scale = shape_util.variable_shape([x_, scale_])
            shape_x, shape_scale = shape_util.refine_shapes_for_broadcast(shape_x, shape_scale)

            tensor_x = tvm.placeholder(shape_x, dtype_x, "tensor_x")
            tensor_scale = tvm.placeholder(shape_scale, dtype_scale, "tensor_scale")
            tensor_bias = tvm.placeholder(shape_scale, dtype_bias, "tensor_bias")
            res = scale_compute(tensor_x, tensor_scale, tensor_bias, y, axis, num_axes,
                                scale_from_blob, kernel_name)
            tensor_list = [tensor_x, tensor_scale, res]
            if len(shape_bias_new) > 0:
                tensor_list = [tensor_x, tensor_scale, tensor_bias, res]
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
