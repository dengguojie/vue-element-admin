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
bninference_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import get_current_build_config


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_bias_compute(x, mean, variance, scale, bias):
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
    dtype_bias = bias.dtype

    shape_x = shape_util.shape_to_list(x.shape)
    shape_mean = shape_util.shape_to_list(mean.shape)
    shape_x, shape_mean, shape_max = shape_util.broadcast_shapes(shape_x, shape_mean,
                                                                 param_name_input1="x", param_name_input2="mean")

    x_broadcast = tbe.broadcast(x, shape_max)
    mean_broadcast = tbe.broadcast(mean, shape_max)
    var_broadcast = tbe.broadcast(variance, shape_max)

    mean_add = tbe.vadd(x_broadcast, mean_broadcast)
    res_y = tbe.vmul(var_broadcast, mean_add)

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SOC_VERSION")
    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype_x == "float16":
            is_cast = True
            res_y = tbe.cast_to(res_y, "float32")
        if dtype_scale == "float16":
            scale = tbe.cast_to(scale, "float32")
        if dtype_bias == "float16":
            bias = tbe.cast_to(bias, "float32")

    scale_broad = tbe.broadcast(scale, shape_max)
    bias_broad = tbe.broadcast(bias, shape_max)

    res_tmp = tbe.vmul(res_y, scale_broad)
    res = tbe.vadd(res_tmp, bias_broad)

    if is_cast:
        res = tbe.cast_to(res, dtype_x)
    return res


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_compute(x, mean, variance, scale):
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

    shape_x = shape_util.shape_to_list(x.shape)
    shape_mean = shape_util.shape_to_list(mean.shape)
    shape_x, shape_mean, shape_max = shape_util.broadcast_shapes(shape_x, shape_mean,
                                                                 param_name_input1="x", param_name_input2="mean")

    x_broadcast = tbe.broadcast(x, shape_max)
    mean_broadcast = tbe.broadcast(mean, shape_max)
    var_broadcast = tbe.broadcast(variance, shape_max)

    mean_add = tbe.vadd(x_broadcast, mean_broadcast)
    res_y = tbe.vmul(var_broadcast, mean_add)

    is_cast = False
    product_version = tbe_platform.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype_x == "float16":
            is_cast = True
            res_y = tbe.cast_to(res_y, 'float32')
        if dtype_scale == "float16":
            scale = tbe.cast_to(scale, 'float32')

    scale_broad = tbe.broadcast(scale, shape_max)

    res = tbe.vmul(res_y, scale_broad)

    if is_cast:
        res = tbe.cast_to(res, dtype_x)
    return res


# pylint: disable=invalid-name,redefined-outer-name
def _fused_compute(x, mean, variance):
    """
    Parameters
    ----------
    x: dict
        contains x data.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.

    Returns
    -------
    res: dict
        the result of compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    shape_mean = shape_util.shape_to_list(mean.shape)

    shape_x, shape_mean, shape_max = shape_util.broadcast_shapes(shape_x, shape_mean,
                                                                 param_name_input1="x", param_name_input2="mean")

    x_broadcast = tbe.broadcast(x, shape_max)
    mean_broadcast = tbe.broadcast(mean, shape_max)
    var_broadcast = tbe.broadcast(variance, shape_max)

    mean_add = tbe.vadd(x_broadcast, mean_broadcast)
    res_y = tbe.vmul(var_broadcast, mean_add)

    return res_y


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name,protected-access
@register_operator_compute("bninference_d", op_mode="dynamic", support_fusion=True)
def bninference_d_compute(x, mean, variance, scale, bias, y,
                          momentum, epsilon, use_global_stats, mode):
    """
    Parameters
    ----------
    x: dict
        contains x data. A 4D or 5D Tensor of type float16 or float32.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.
    scale: dict
        no use in caffe batchnorm inference
    bias: dict
        no use in caffe batchnorm inference
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    momentum: float
        a float number of the variance and mean's scale factor
    epsilon: float
        a small float number added to the variance of x to avoid dividing by zero. Defaults to "0.00001".
    use_global_stats: bool
        means the caffe inference model, only can be True.
    mode: int
        an optional attr, no use

    Returns
    -------
    res: TVM tensor list
        the result of compute
    """

    fuse_y = y
    if y is None:
        fuse_y = {"addr_type": 0, "valid_shape": [], "slice_offset": []}

    # if l1 fusion x format must 5hd
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value if "L1_fusion_type" in x.op.attrs else -1
    if l1_fusion_type != -1 and y.get("format").upper() != 'NC1HWC0':
        shape_rule = "when L1_FUSION is enabled for the bninference operator, the input format must be 5HD"
        error_manager_vector.raise_err_check_params_rules("bninference_d",
                                                          shape_rule, "x", y.get("format").upper())

    fusion_params = get_fusion_params(x, mean, variance, scale, bias, fuse_y)

    if scale is not None and bias is not None:
        res = _fused_scale_bias_compute(x, mean, variance, scale, bias)
    elif scale is not None and bias is None:
        res = _fused_scale_compute(x, mean, variance, scale)
    else:
        res = _fused_compute(x, mean, variance)
    res.op.attrs["ele_fusion_params"] = fusion_params
    res.op.attrs["L1_fusion_type"] = fusion_params["l1_fusion_type"]

    return res


def get_fusion_params(x, mean, variance, scale, bias, y):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x : tensor of input data
    y : dict of output data
    x_tensor_num: input tensor num
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

    input_tensor = [x, mean, variance, scale, bias]
    for x in input_tensor:
        if x is not None:
            l1_fusion_type = -1
            if not get_current_build_config("enable_op_prebuild"):
                l1_fusion_type = x.op.attrs["L1_fusion_type"].value \
                    if "L1_fusion_type" in x.op.attrs else -1
                if l1_fusion_type == 1:
                    error_detail = 'bninference does not support l1 width fusion, l1_fusion_type:', l1_fusion_type
                    error_manager_vector.raise_err_specific_reson("bninference_d", error_detail)
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
            in_l1_flag = x.op.attrs["addr_type"].value == 1 \
                if "addr_type" in x.op.attrs else False
            in_l1_flag_list.append(in_l1_flag)
            in_valid_shape = x.op.attrs["valid_shape"] \
                if "valid_shape" in x.op.attrs else []
            in_valid_shape_list.append(in_valid_shape)
            in_slice_offset = x.op.attrs["slice_offset"] \
                if "slice_offset" in x.op.attrs else []
            in_slice_offset_list.append(in_slice_offset)
            in_select_read_flag = x.op.tag == "read_select_5d"
            in_select_read_flag_list.append(in_select_read_flag)

    l1_fusion_type = 0 if is_l1_depth_fusion else -1

    out_l1_flag = False
    out_valid_shape = []
    out_slice_offset = []
    out_select_write_flag = False
    if y is not None:
        out_l1_flag = y.get("addr_type", 0) == 1
        out_valid_shape = y.get("valid_shape", [])
        out_slice_offset = y.get("slice_offset", [])
        out_select_write_flag = bool(out_valid_shape)

    fusion_params = {"is_l1fusion": is_l1_depth_fusion,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag_list,
                     "in_select_read_flag": in_select_read_flag_list,
                     "in_valid_shape": in_valid_shape_list,
                     "in_slice_offset": in_slice_offset_list,
                     "out_l1_flag": out_l1_flag,
                     "out_select_write_flag": out_select_write_flag,
                     "out_valid_shape": out_valid_shape,
                     "out_slice_offset": out_slice_offset}
    return fusion_params


def brodcast_inputs_shape(x, mean, variance, scale, offset):
    """
    :param x:x tensor
    :param mean: mean tensor
    :param variance:var tensor
    :return:
    x_input:x
    mean_input:mean
    var_input:var
    scale:scale,not use
    b:not use
    """
    shape_x = x.get("shape")
    format_x = x.get("format")

    if format_x in ("ND", "NCHW"):
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 1
    elif format_x == "NHWC":
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 3
    else:
        c1 = shape_x[1]
        c0 = shape_x[4]
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")

    if format_x in ("ND", "NCHW", "NHWC"):
        shape_mean = [1] * len(shape_x[:index_c]) + list(shape_mean) \
            + [1] * len(shape_x[index_c + 1:])
        shape_variance = [1] * len(shape_x[:index_c]) + list(shape_variance) \
            + [1] * len(shape_x[index_c + 1:])
    else:
        shape_mean = [1, c1, 1, 1, c0]
        shape_variance = [1, c1, 1, 1, c0]

    shape_scale = {}
    shape_offset = {}
    if scale is not None:
        shape_scale = scale.get("shape")
    if offset is not None and bool(offset):
        shape_offset = offset.get("shape")

    is_l1_depth_fusion = False

    _, l1_fusion_type = get_l1_paras(x)
    is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
    _, l1_fusion_type = get_l1_paras(mean)
    is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
    _, l1_fusion_type = get_l1_paras(variance)
    is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion

    shape_scale_new = None
    shape_offset_new = None

    if len(shape_scale) > 0:
        shape_scale_new = get_param_scale_shape(shape_x, shape_scale)
        _, l1_fusion_type = get_l1_paras(scale)
        is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion

        if len(shape_offset) > 0:
            shape_offset_new = shape_scale_new
            _, l1_fusion_type = get_l1_paras(offset)
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion

    return shape_x, shape_mean, shape_variance, shape_scale_new, shape_offset_new, is_l1_depth_fusion


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
    new shape
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_scale)

    return shape


def get_l1_paras(x):
    l1_fusion_type = -1
    if not get_current_build_config("enable_op_prebuild"):
        l1_fusion_type = x.get('L1_fusion_type', -1)
        if l1_fusion_type == 1:
            error_detail = 'bninference does not support l1 width fusion, l1_fusion_type:', l1_fusion_type
            error_manager_vector.raise_err_specific_reson("bninference_d", error_detail)
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    attr_x = {
        "addr_type": addr_type,
        "valid_shape": valid_shape,
        "slice_offset": slice_offset,
        "L1_fusion_type": l1_fusion_type
    }
    return attr_x, l1_fusion_type


# pylint: disable=locally-disabled,no-member
@register_operator("BNInferenceD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def bninference_d(x, mean, variance, scale, offset, y, momentum, epsilon,
                  use_global_stats, mode, kernel_name="bninference"):
    """

    Parameters
    ----------
    x: dict
        contains x data. A 4D or 5D Tensor of type float16 or float32.
    mean: dict
        contains mean data.Must be 1D if input "x" Specifies the mean used for inference.
    variance: dict
        contains variance data.Must be 1D if input "x" Specifies the variance used for inference.
    scale: dict
        no use in caffe batchnorm inference
    bias: dict
        no use in caffe batchnorm inference
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    momentum: float
        a float number of the variance and mean's scale factor
    epsilon: float
        a small float number added to the variance of x to avoid dividing by zero. Defaults to "0.00001".
    use_global_stats: bool
        means the caffe inference model, only can be True.
    mode: int
        an optional attr, no use
    kernel_name: str
        kernel name

    Returns
    -------
    None
    """
    # check format
    format_x = x.get("format")
    format_data = x.get("format")
    excepted_format_list = ["ND", "NC1HWC0", "NCHW", "NHWC"]
    para_check.check_format(format_data, excepted_format_list, param_name="x")

    # check dtype
    product = tbe_platform.get_soc_spec("SOC_VERSION")
    if product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        checklist = ["float16"]
    else:
        checklist = ["float32", "float16"]

    dtype_x = x.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")
    para_check.check_dtype(dtype_x.lower(), checklist, param_name="x")
    para_check.check_dtype(dtype_mean.lower(), checklist, param_name="mean")
    para_check.check_dtype(dtype_variance.lower(), checklist, param_name="variance")

    if scale is not None:
        dtype_scale = scale.get("dtype")
        para_check.check_dtype(dtype_scale.lower(), checklist, param_name="scale")
    if offset is not None and bool(offset):
        dtype_offset = offset.get("dtype")
        para_check.check_dtype(dtype_offset.lower(), checklist, param_name="offset")

    # brodcast inputs shape
    shape_x, shape_mean, shape_variance, _, _, is_l1_depth_fusion = brodcast_inputs_shape(x, mean,
                                                                                          variance, scale, offset)

    x["shape"] = shape_x

    # compute mean shape
    mean["shape"] = shape_mean

    tbe_context.get_context().add_compile_info("broadcast_mean_shape", shape_mean)

    mean_range = []
    for i, _range in enumerate(x["range"]):
        _range = (shape_mean[i], shape_mean[i]) if shape_mean[i] != -1 else _range
        mean_range.append(_range)
    mean["range"] = tuple(mean_range)

    variance["shape"] = shape_variance

    # op compute and schedule
    ins = classify([x, mean], OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedules, tensors = [], []

    # used to be create holder
    for (x_input, mean_input) in ins:
        # op compute
        with tbe.compute():
            # get all dynamic tensor shape
            shape_x, shape_mean = shape_util.variable_shape([x_input, mean_input])

            # new all input data place holder
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
            data_mean = tvm.placeholder(shape_mean, name="data_mean", dtype=dtype_mean)
            data_variance = tvm.placeholder(shape_mean, name="data_variance", dtype=dtype_variance)

            # add all tensors
            if scale is not None and offset is not None:
                data_scale = tvm.placeholder(shape_mean, name="data_scale", dtype=dtype_scale)
                data_offset = tvm.placeholder(shape_mean, name="data_offset", dtype=dtype_offset)
                res = bninference_d_compute(data_x, data_mean, data_variance, data_scale, data_offset,
                                            y, momentum, epsilon, use_global_stats, mode)
                tensor_list = [data_x, data_mean, data_variance, data_scale, data_offset, res]
            elif scale is not None and offset is None:
                data_scale = tvm.placeholder(shape_mean, name="data_scale", dtype=dtype_scale)
                res = bninference_d_compute(data_x, data_mean, data_variance, data_scale, None,
                                            y, momentum, epsilon, use_global_stats, mode)
                tensor_list = [data_x, data_mean, data_variance, data_scale, res]
            else:
                res = bninference_d_compute(data_x, data_mean, data_variance, None, None,
                                            y, momentum, epsilon, use_global_stats, mode)
                tensor_list = [data_x, data_mean, data_variance, res]

            tensors.append(tensor_list)

        # target auto schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        # append schedule 2 schedules
        schedules.append(sch)

    # build
    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "l1_fusion_option": is_l1_depth_fusion
    }

    tbe.build(schedules, config)
