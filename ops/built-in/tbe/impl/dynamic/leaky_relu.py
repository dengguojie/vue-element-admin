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
leaky_relu
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import get_current_build_config
from impl.util.platform_adapter import is_supported_vlrelu
from impl.common_util import get_vlrelu
from impl.common_util import get_attr


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
def get_fusion_params(x_tensor, y):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x_tensor : tensor of input data
    y : dict of output data
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = -1
    if not get_current_build_config("enable_op_prebuild"):
        l1_fusion_type = x_tensor.op.attrs["L1_fusion_type"].value \
            if "L1_fusion_type" in x_tensor.op.attrs else -1
        if l1_fusion_type == 1:
            error_manager_vector.raise_err_specific_reson("leaky_relu",
                                                          "leaky_relu does not support l1 fusion")
    is_l1_depth_fusion = l1_fusion_type == 0
    in_l1_flag = x_tensor.op.attrs["addr_type"].value == 1 \
        if "addr_type" in x_tensor.op.attrs else False
    in_valid_shape = x_tensor.op.attrs["valid_shape"] \
        if "valid_shape" in x_tensor.op.attrs else []
    in_slice_offset = x_tensor.op.attrs["slice_offset"] \
        if "slice_offset" in x_tensor.op.attrs else []
    in_select_read_flag = x_tensor.op.tag == "read_select_5d"

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
                     "in_l1_flag": in_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "in_valid_shape": in_valid_shape,
                     "in_slice_offset": in_slice_offset,
                     "out_l1_flag": out_l1_flag,
                     "out_select_write_flag": out_select_write_flag,
                     "out_valid_shape": out_valid_shape,
                     "out_slice_offset": out_slice_offset}
    return fusion_params


@register_operator_compute("LeakyRelu", op_mode="dynamic", support_fusion=True)
def leaky_relu_compute(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """
    compute for caffe_relu_layer_cce
    """
    negative_slope_dtype = "float"
    dtype = x.dtype
    fusion_params = get_fusion_params(x, y)
    # check whether support vlrelu interface
    if not is_supported_vlrelu:
        res, negative_slope = get_vlrelu(x, negative_slope, "negative_slope",
                                         negative_slope_dtype)
    else:
        negative_slope = get_attr(negative_slope, "negative_slope",
                                  dtype, negative_slope_dtype)
        res = tbe.vlrelu(x, negative_slope)
    if x.op.attrs:
        if 'format' in x.op.attrs:
            res.op.attrs['format'] = x.op.attrs['format']
    res.op.attrs["negative_slope"] = negative_slope
    res.op.attrs["ele_fusion_params"] = fusion_params
    res.op.attrs["L1_fusion_type"] = fusion_params["l1_fusion_type"]

    return res


@register_operator("LeakyRelu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def leaky_relu(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """leaky_relu op for input tensor

       f(x)= x(x>=0) or negative_slope*x(x<0) equal to
       f(x)=negative_slope*x

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization

    kernel_name : str
        cce kernel name, default value is "leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    dtype = x.get("dtype").lower()

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    para_check.check_dtype(dtype, check_list, param_name="x")

    l1_fusion_type = -1
    if not get_current_build_config("enable_op_prebuild"):
        l1_fusion_type = x.get("L1_fusion_type", -1)
        if l1_fusion_type == 1:
            error_manager_vector.raise_err_specific_reson("leaky_relu",
                                                          "leaky_relu does not support l1 fusion")
    is_l1_depth_fusion = l1_fusion_type == 0
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    attr_x = {"addr_type": addr_type,
              "valid_shape": valid_shape,
              "slice_offset": slice_offset,
              "L1_fusion_type": l1_fusion_type}

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            input_data_x = tvm.placeholder(x_shape[0], name="input_data_x",
                                           dtype=dtype, attrs=attr_x)

            res = leaky_relu_compute(input_data_x, y, negative_slope, kernel_name)
            tensors.append([input_data_x, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "l1_fusion_option": is_l1_depth_fusion,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
