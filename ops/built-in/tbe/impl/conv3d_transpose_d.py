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
    def _ncdhw2ndhwc(shape_ncdhw):
        shape_ndhwc = (shape_ncdhw[0], shape_ncdhw[2], shape_ncdhw[3], shape_ncdhw[4], shape_ncdhw[1])
        return shape_ndhwc

    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_sizes
    ori_shape_strides = strides
    ori_shape_dialtions = dilations

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    if ori_format_filters == "DHWCN":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NDHWC":
        shape_filters = (ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[0],
                        )
    elif ori_format_filters == "NCDHW":
        shape_filters = (ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[1],
                         ori_shape_filters[0],
                        )
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': '[{}, {}, {}]'
                                    .format('DHWCN', 'NDHWC', 'NCDHW'),
            'format': ori_format_filters
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_out_backprop == "NDHWC":
        shape_out_backprop = ori_shape_out_backprop
        shape_strides = ori_shape_strides
        shape_dilations = ori_shape_dialtions
    elif ori_format_out_backprop == "NCDHW":
        shape_out_backprop = _ncdhw2ndhwc(ori_shape_out_backprop)
        shape_strides = _ncdhw2ndhwc(ori_shape_strides)
        shape_dilations = _ncdhw2ndhwc(ori_shape_dialtions)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_out_backprop
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if ori_format_res == "NDHWC":
        shape_res = ori_shape_res
    elif ori_format_res == "NCDHW":
        shape_res = _ncdhw2ndhwc(ori_shape_res)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_res
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    conv3d_transpose_cce(shape_filters,
                         shape_out_backprop,
                         shape_res,
                         shape_strides,
                         pads,
                         shape_dilations,
                         filters_dtype,
                         out_backprop_dtype,
                         res_dtype,
                         kernel_name)


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple), (list, tuple),
                             str, str, str, str)
def conv3d_transpose_cce(shape_filter, # pylint: disable=R0913,R0914
                         shape_out_backprop, input_sizes,
                         strides, pads, dilations=(1, 1, 1, 1, 1),
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

        dedx = conv3d_bp_dx.conv3d_backprop_input_compute(
            filters=filters,
            out_backprop=dedy,
            filter_sizes=shape_filter_ncdhw,
            input_sizes=input_sizes,
            strides=strides,
            padding=padding,
            dilations=dilations,
            res_dtype=res_dtype,
            kernel_name=kernel_name
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
        input_sizes, strides, pads, dilations,
        filter_dtype, out_backprop_dtype,
        res_dtype, kernel_name)
    (shape_filter, shape_out_backprop, input_sizes, strides, padding, dilations,
     filter_dtype, out_backprop_dtype, res_dtype, kernel_name) = res

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter

    # Channel axis should be align with 16
    c0_size = tbe_platform.C0_SIZE
    shape_dedy = (dedy_batch,
                  dedy_deep,
                  util_common.ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)

    shape_filter_frac = (filter_depth,
                         util_common.ceil(filter_channel, c0_size) * filter_h * filter_w,
                         util_common.ceil(filter_batch, c0_size), c0_size, c0_size)
    _conv3d_transpose_achieve_with_tvm()
