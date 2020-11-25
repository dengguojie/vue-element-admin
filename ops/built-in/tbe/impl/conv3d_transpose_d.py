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
conv3d_transpose_d
"""
import te.lang.cce as tbe
from te.lang.cce.te_compute import conv3d_backprop_input_compute as conv3d_bp_dx
import te.platform as tbe_platform
from te.utils import para_check
from te.utils.error_manager import error_manager_util
from te import tvm
from impl import conv3d_backprop_input_d
from impl.util import util_common
from impl.util import util_select_op_base


L1FUSION_INPUT_CTR = 2

OUT_BACKPROP_TARGET_FORMAT = "NDHWC"
OUT_BACKPROP_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]
FILTER_TARGET_FORMAT = "DHWCN"
FILTER_FORMAT_WHITE_LIST = ["DHWCN", "NDHWC", "NCDHW"]
RES_TARGET_FORMAT = "NDHWC"
RES_FORMAT_WHITE_LIST = ["NDHWC", "NCDHW"]

def _transform_shape_with_format(src_format, to_format, ori_shape, format_white_list):
    # input format is not expected
    if ((src_format not in format_white_list) or
        (to_format not in format_white_list)):
        return None
    # need not to transform
    if src_format == to_format:
        return ori_shape
    res_shape = [1 for _ in range(len(to_format))]
    for i in range(len(to_format)):
        for j in range(len(src_format)):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
                break
    return res_shape

def get_op_support_info(out_backprop,
                        filters, # pylint: disable=R0913,R0914
                        bias,
                        offset_w,
                        y_input,
                        input_sizes,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1, 1),
                        groups=1,
                        data_format="NDHWC",
                        output_padding=[0, 0, 0, 0, 0],
                        offset_x=0,
                        kernel_name="conv3d_transpose",
                        op_slice_info=""):
    """
    algorithm: get_op_support_info

    Parameters
    ----------
    out_backprop: A dict with keys(shape and dtype)
        The shape of gradients

    filters: A dict with keys(shape and dtype)
        Input weight tensor

    bias: A dict with keys(shape and dtype) or None
        Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
        Input offset_w tensor

    y_input: A dict with keys(shape and dtype)
       Conv3d_transpose output tensor, dtype must be assigned

    input_sizes: The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
        Filter move stride

    pads: A tuple/list of 6 integers
        [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
        Filter expand size of dilated conv3d_transpose, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
        Default value is 1

    data_format: The data format of the input and output data
        Default format is "NDHWC"

    output_padding: The size will be added in the output shape
        Default value is [0, 0, 0, 0, 0]

    offset_x: Int
        Input offset_x value, default value is 0

    kernel_name: Str
        Kernel name, default value is "conv3d_transpose"

    op_slice_info: Str
        Default value is ""

    Returns
    -------
    op_cal_info_in_json: A dict with keys(split_maps, reduce_maps, l1_fusion_enable
                         and min_tbe_l1_space)
    """
    def _cal_min_l1space():
        block_size = 16
        w_value = ori_shape_out_backprop[3] * strides[3]
        filter_d_dilation = (ori_shape_filters[0] - 1) * dilations[1] + 1
        filter_h_dilation = (ori_shape_filters[1] - 1) * dilations[2] + 1
        filter_w_dilation = (ori_shape_filters[2] - 1) * dilations[3] + 1
        if ori_shape_res[3] > block_size:
            h_value_max = filter_h_dilation + 1
        elif block_size % ori_shape_res[3] == 0:
            h_value_max = filter_h_dilation + block_size // ori_shape_res[3] - 1
        else:
            h_value_max = filter_h_dilation + block_size // ori_shape_res[3] + 1

        a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // strides[1] + 2) * block_size * 2
        b_l1_size = filter_h_dilation * filter_w_dilation * filter_d_dilation * block_size * block_size * 2
        return a_l1_size + b_l1_size

    def _get_slice_info():
        overlap_d = -1 if (ori_shape_filters[0] == 1 and strides_formated[1] == 1) else 0
        overlap_h = -1 if (ori_shape_filters[1] == 1 and strides_formated[2] == 1) else 0
        overlap_w = -1 if (ori_shape_filters[2] == 1 and strides_formated[3] == 1) else 0
        overlap_c = -1 if ori_shape_filters[3] <= 16 else 0

        # format
        axis_split_matrix = []
        axis_reduce_list = None
        format_out_backprop = out_backprop.get("format")
        if format_out_backprop == "NDC1HWC0":
            # cut N
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [0], [-1], [-1]]),
                                    util_select_op_base.SplitOutput([0, [0]])])
            # cut D
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [1], [overlap_d], [overlap_d]]),
                                    util_select_op_base.SplitOutput([0, [1]])])
            # cut H
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [3], [overlap_h], [overlap_h]]),
                                    util_select_op_base.SplitOutput([0, [3]])])
            # cut W
            axis_split_matrix.append([util_select_op_base.SplitInput([0, [4], [overlap_w], [overlap_w]]),
                                    util_select_op_base.SplitOutput([0, [4]])])
            # cut Cout
            if bias:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [0], [overlap_c], [overlap_c]], [2, [0], [-1], [-1]]),
                    util_select_op_base.SplitOutput([0, [2]])]
                )
            else:
                axis_split_matrix.append(
                    [util_select_op_base.SplitInput([1, [0], [overlap_c], [overlap_c]]),
                    util_select_op_base.SplitOutput([0, [2]])]
                )
        else:
            axis_split_matrix = None

        return axis_split_matrix, axis_reduce_list

    ori_shape_out_backprop = _transform_shape_with_format(out_backprop.get("ori_format"),
                                                          OUT_BACKPROP_TARGET_FORMAT,
                                                          out_backprop.get("ori_shape"),
                                                          OUT_BACKPROP_FORMAT_WHITE_LIST)
    strides_formated = _transform_shape_with_format(out_backprop.get("ori_format"),
                                                    OUT_BACKPROP_TARGET_FORMAT,
                                                    strides,
                                                    OUT_BACKPROP_FORMAT_WHITE_LIST)
    if ori_shape_out_backprop is None or strides_formated is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': ",".join(OUT_BACKPROP_FORMAT_WHITE_LIST),
            'format': out_backprop.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    ori_shape_filters = _transform_shape_with_format(filters.get("ori_format"),
                                                     FILTER_TARGET_FORMAT,
                                                     filters.get("ori_shape"),
                                                     FILTER_FORMAT_WHITE_LIST)
    if ori_shape_filters is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': ",".join(FILTER_FORMAT_WHITE_LIST),
            'format': filters.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    ori_shape_res = _transform_shape_with_format(y_input.get("ori_format"),
                                                 RES_TARGET_FORMAT,
                                                 y_input.get("ori_shape"),
                                                 RES_FORMAT_WHITE_LIST)
    if ori_shape_res is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': ",".join(RES_FORMAT_WHITE_LIST),
            'format': y_input.get("ori_format")
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    axis_split_info, axis_reduce_info = _get_slice_info()

    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_info,
                                                              axis_reduce_info,
                                                              L1FUSION_INPUT_CTR,
                                                              _cal_min_l1space())

    return op_cal_info_in_json

@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
    para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,  para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_STR,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_INT,
    para_check.KERNEL_NAME)
def conv3d_transpose_d(out_backprop, filters, # pylint: disable=R0913,R0914
                       bias, offset_w, y_input, input_sizes,
                       strides, pads, dilations=(1, 1, 1, 1, 1), groups=1,
                       data_format="NDHWC",
                       output_padding=[0, 0, 0, 0, 0],
                       offset_x=0, kernel_name="conv3d_transpose"):
    """
    algorithm: conv3d_transpose

    Parameters
    ----------
    out_backprop: A dict with keys(shape and dtype)
        The shape of gradients

    filters: A dict with keys(shape and dtype)
        Input weight tensor

    bias: A dict with keys(shape and dtype) or None
        Input bias tensor

    offset_w: A dict with keys(shape and dtype) or None
        Input offset_w tensor

    y_input: A dict with keys(shape and dtype)
       Conv3d_transpose output tensor, dtype must be assigned

    input_sizes: The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
        Filter move stride

    pads: A tuple/list of 6 integers
        [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
        Filter expand size of dilated conv3d_transpose, default value is (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
        Default value is 1

    data_format: The data format of the input and output data
        Default format is "NDHWC"

    output_padding: The size will be added in the output shape
        Default value is [0, 0, 0, 0, 0]

    offset_x: Int
        Input offset_x value, default value is 0

    kernel_name: Str
        Kernel name, default value is "conv3d_transpose"

    Returns
    -------
    None
    """

    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_sizes
    ori_shape_strides = strides
    ori_shape_dialtions = dilations
    ori_shape_output_padding = output_padding

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    if (isinstance(ori_shape_output_padding, (tuple, list)) and
        len(ori_shape_output_padding) != util_common.CONV3D_SHAPE_COMMON_DIM):
        dict_args = {
            'errCode': 'E62006',
            'error_desc': 'output_padding should be 5-dim list/tuple',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    # transform filter shape
    shape_filters = _transform_shape_with_format(ori_format_filters,
                                                 FILTER_TARGET_FORMAT,
                                                 ori_shape_filters,
                                                 FILTER_FORMAT_WHITE_LIST)
    if shape_filters is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': ",".join(FILTER_FORMAT_WHITE_LIST),
            'format': ori_format_filters
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    # transform out_backprop, strides, dilations shape
    shape_out_backprop = _transform_shape_with_format(ori_format_out_backprop,
                                                      OUT_BACKPROP_TARGET_FORMAT,
                                                      ori_shape_out_backprop,
                                                      OUT_BACKPROP_FORMAT_WHITE_LIST)

    shape_strides = _transform_shape_with_format(ori_format_out_backprop,
                                                 OUT_BACKPROP_TARGET_FORMAT,
                                                 ori_shape_strides,
                                                 OUT_BACKPROP_FORMAT_WHITE_LIST)

    shape_dilations = _transform_shape_with_format(ori_format_out_backprop,
                                                   OUT_BACKPROP_TARGET_FORMAT,
                                                   ori_shape_dialtions,
                                                   OUT_BACKPROP_FORMAT_WHITE_LIST)

    shape_output_padding = _transform_shape_with_format(ori_format_out_backprop,
                                                        OUT_BACKPROP_TARGET_FORMAT,
                                                        ori_shape_output_padding,
                                                        OUT_BACKPROP_FORMAT_WHITE_LIST)

    if (shape_out_backprop is None or shape_strides is None or shape_dilations is None or
        shape_output_padding is None):
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': ",".join(OUT_BACKPROP_FORMAT_WHITE_LIST),
            'format': ori_format_out_backprop
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    _, dilation_d, dilation_h, dilation_w, _ = shape_dilations
    _, stride_d, stride_h, stride_w, _ = shape_strides
    _, output_padding_d, output_padding_h, output_padding_w, _ = shape_output_padding

    # transform res shape
    shape_res = _transform_shape_with_format(ori_format_res,
                                             RES_TARGET_FORMAT,
                                             ori_shape_res,
                                             RES_FORMAT_WHITE_LIST)
    if shape_res is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': ",".join(RES_FORMAT_WHITE_LIST),
            'format': ori_format_res
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if output_padding_d < 0 or (output_padding_d >= dilation_d and output_padding_d >= stride_d):
        dict_args = {
            'errCode': 'E62305',
            'param_name': 'output_padding D',
            'expect_value': '[{}, {})'.format(str(0), 'max(stride D,dilation D)'),
            'value': str(output_padding_d)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if output_padding_h < 0 or (output_padding_h >= dilation_h and output_padding_h >= stride_h):
        dict_args = {
            'errCode': 'E62305',
            'param_name': 'output_padding H',
            'expect_value': '[{}, {})'.format(str(0), 'max(stride H,dilation H)'),
            'value': str(output_padding_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if output_padding_w < 0 or (output_padding_w >= dilation_w and output_padding_w >= stride_w):
        dict_args = {
            'errCode': 'E62305',
            'param_name': 'output_padding W',
            'expect_value': '[{}, {})'.format(str(0), 'max(stride W,dilation W)'),
            'value': str(output_padding_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    conv3d_transpose_cce(shape_filters,
                         shape_out_backprop,
                         shape_res,
                         shape_strides,
                         pads,
                         groups,
                         shape_dilations,
                         filters_dtype,
                         out_backprop_dtype,
                         res_dtype,
                         kernel_name)


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple), int,
                             (list, tuple), str, str, str, str)
def conv3d_transpose_cce(shape_filter, # pylint: disable=R0913,R0914
                         shape_out_backprop, input_sizes,
                         strides, pads, groups, dilations=(1, 1, 1, 1, 1),
                         filter_dtype='float16',
                         out_backprop_dtype='float16',
                         res_dtype='float16',
                         kernel_name="conv3d_transpose_cce"):
    """
    Topi interface of conv3d transpose

    Parameters:
    ----------
    shape_filter : The shape of filter
        5-D with shape [ depth, height, weight, batch, channels]

    shape_out_backprop : The shape of gradients
        5-D with shape [batch, depth, height, weight, channels]

    input_sizes : The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints or str

    groups: Int of blocked connections from input channels to output channels

    dilations : An optional list/tuple of ints. Only support (1, 1, 1, 1, 1) now

    filter_dtype : The dtype of filter data. Default value is float16

    out_backprop_dtype : The dtype of gradients data. Default value is float16

    res_dtype : The dtype of result(De/Dx) data. Default value is float16

    kernel_name : Cce kernel name. Default value is "conv3d_transpose_cce"

    Returns
    ----------
    None
    """
    def _conv3d_transpose_achieve_with_tvm():
        dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
        shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth,
                              filter_h, filter_w]

        filters = tvm.placeholder(shape_filter_frac,
                                  name="filter", dtype=filter_dtype)

        para_dict = {
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "res_dtype": res_dtype,
            "kernel_name": kernel_name,
            "group_dict": group_dict
        }

        dedx = conv3d_bp_dx.conv3d_dx(
            filter=filters,
            out_backprop=dedy,
            filter_size=shape_filter_ncdhw,
            input_size=input_sizes,
            para_dict=para_dict
        )
        tensor_list = [dedy, filters, dedx]

        with tvm.target.cce():
            sch = tbe.auto_schedule(dedx)

        config = {
            "name": kernel_name,
            "tensor_list": tensor_list
        }
        tbe.cce_build_code(sch, config)

    res = conv3d_backprop_input_d.check_conv3dbp_input_params(
        shape_filter, shape_out_backprop,
        input_sizes, strides, pads, groups, dilations,
        filter_dtype, out_backprop_dtype,
        res_dtype, kernel_name)
    (shape_filter, shape_out_backprop, input_sizes, strides, padding, dilations,
     filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter

    # Channel axis should be align with 16
    c0_size = tbe_platform.C0_SIZE
    shape_dedy = (dedy_batch,
                  dedy_deep,
                  util_common.ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)

    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]

    shape_filter_frac = (real_g * filter_depth * cin1_g * filter_h * filter_w,
                         cout_g // c0_size, c0_size, c0_size)
    _conv3d_transpose_achieve_with_tvm()
