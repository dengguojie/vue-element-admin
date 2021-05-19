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
conv3d_backprop_input_d
"""
from impl.util import util_common
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# the dim of shape in conv_backprop must be 5
_CONV_BACKPROP_SHAPE_DIM = 5
# the dim of pads in conv3d_backprop must be 6
_CONV_BACKPROP_PAD_SHAPE_DIM = 6
# the dim of strides in conv_backprop must be 5
_STRIDES_SHAPE_DIM = 5
# the dim of dilations in conv_backprop must be 5
_DILATIONS_SHAPE_DIM = 5

# fmapH, fmapW must be in [1,4096]
_FMAP_HW_MIN = 1
_FMAP_HW_MAX = 4096

# DeDy H,W must be in [1,4096]
_DEDY_HW_MIN = 1
_DEDY_HW_MAX = 4096

# filterH, filterW must be in [1,255]
_FILTER_HW_MIN = 1
_FILTER_HW_MAX = 255
_FILTER_HW_SIZE = 256
_FILTER_D_MAX = 128

# stride must be in [1,63] and h*w not lagger than 256
_STRIDE_HW_MIN = 1
_STRIDE_HW_MAX = 63
_STRIDE_SIZE_MAX = 256
_STRIDE_SIZE_HWD_MAX = 343

# special num
_KHWD_COEFF = 343

# the max num of each axis of shape
_DEFAULT_MAX_SHAPE_NUM = 1000000

# dilation must be in [1,255]
_DILATION_HW_MIN = 1
_DILATION_HW_MAX = 255

# the bytes length of several dtype
_BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
_DATA_SIZE_MAX = 9223372036854775807
_PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# align with 16 for chips
_C0_SIZE = tbe_platform.C0_SIZE
_BLOCK_SIZE = 16

def _get_ndhwc_shape(ori_format_filters, ori_shape_filters,
                     ori_format_out_backprop, ori_shape_out_backprop,
                     ori_shape_strides, ori_shape_dialtions,
                     ori_format_res, ori_shape_res):
    def _ncdhw2ndhwc(shape_ncdhw):
        shape_ndhwc = [shape_ncdhw[0], shape_ncdhw[2], shape_ncdhw[3], shape_ncdhw[4], shape_ncdhw[1]]
        return shape_ndhwc

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
        shape_out_backprop = list(ori_shape_out_backprop)
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

    shape_out_backprop[-1] = shape_filters[-1]

    return shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res


@tbe_platform.fusion_manager.register("conv3d_backprop_input")
def conv3d_backprop_input_fusion_compute(filters, #pylint: disable=R0913,R0914
                                         out_backprop, y_input, input_sizes, strides,
                                         pads, dilations=(1, 1, 1, 1, 1), groups=1,
                                         data_format="NDHWC",
                                         kernel_name="conv3d_backprop_input"):
    shape_filter = []
    for i in filters.op.attrs['ori_shape']:
        shape_filter.append(i.value)
    filter_format = filters.op.attrs['ori_format']

    shape_out_backprop = []
    for i in out_backprop.op.attrs['ori_shape']:
        shape_out_backprop.append(i.value)
    out_backprop_format = out_backprop.op.attrs['ori_format']

    shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res = \
        _get_ndhwc_shape(filter_format,
                         shape_filter,
                         out_backprop_format,
                         shape_out_backprop,
                         strides,
                         dilations,
                         data_format,
                         input_sizes)
    filter_dtype = filters.op.attrs['data_type'].value
    out_backprop_dtype = out_backprop.dtype
    res_dtype = "float16"

    res = check_conv3dbp_input_params(shape_filters,
                                      shape_out_backprop,
                                      shape_res,
                                      shape_strides,
                                      pads,
                                      groups,
                                      shape_dilations,
                                      filter_dtype,
                                      out_backprop_dtype,
                                      res_dtype,
                                      kernel_name)
    (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
     filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    pads = list(pads)

    shape_dedy = (dedy_batch,
                  dedy_deep,
                  util_common.ceil(dedy_channel, _C0_SIZE),
                  dedy_h,
                  dedy_w,
                  _C0_SIZE)

    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]

    shape_filter_frac = (real_g * filter_depth * cin1_g * filter_h * filter_w,
                         cout_g // _C0_SIZE,
                         _C0_SIZE,
                         _C0_SIZE)
    shape_filter_ncdhw = [filter_batch,
                          filter_channel,
                          filter_depth,
                          filter_h,
                          filter_w]

    para_dict = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedx = tbe.conv3d_backprop_input(filter=filters,
                                     out_backprop=out_backprop,
                                     filter_size=shape_filter_ncdhw,
                                     input_size=input_sizes,
                                     para_dict=para_dict)

    return dedx


def check_supported(filters,
                    out_backprop,
                    y_input,
                    input_sizes,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1, 1),
                    groups=1,
                    data_format="NDHWC",
                    kernel_name="conv3d_backprop_input"):
    """
    The H and W dimension of input_sizes should be in range [1, 4096]
    The H and W dimension of dilation should be in range [1, 255]
    The D,H or W dimension of the filter should be in range [1, 255]
    The padding in each dimension should be in range [0, 255]
    The D,H or W dimension of the stride should be in range [1, 63]
    The filter's H * filter 's W should < 256
    The filter's H * W * D should < 343
    The stride's H * W should < 256
    The stride's H * W * D should < 343
    The groups should <= the feature map's and the filter's channel dimension
    The feature map's channel dimension or filter's channel dimension must be divisible by groups
    The channel dimension of feature map should = the filter's channel dimension * groups
    The out_backprop's channel dimension should = the filter's batch dimension
    The feature map's batch dimension should = the out_backprop's batch dimension
    The D,H or W dimension of the feature map after padding should >= the filter's corresponding dimension after dilation
    The out_backprop's H * stride's H should < 4096
    The out_backprop's W * stride's W should < 4096
    If the output H dimension is not 1, the output W dimension should >= 2

    The data in Ubuffer should <= the chip's Ubuffer size
    The data in L1 buffer should <= the chip's L1 buffer size
    """
    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_sizes
    ori_shape_strides = strides
    ori_shape_dilations = dilations

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    try:
        shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res = \
        _get_ndhwc_shape(ori_format_filters,
                         ori_shape_filters,
                         ori_format_out_backprop,
                         ori_shape_out_backprop,
                         ori_shape_strides,
                         ori_shape_dilations,
                         ori_format_res,
                         ori_shape_res)
        check_conv3dbp_input_params(shape_filters, shape_out_backprop, shape_res, shape_strides, pads,
                                    groups, shape_dilations, filters_dtype, out_backprop_dtype, res_dtype,
                                    kernel_name)
        return True, ""
    except Exception as e:
        reason = e.args[1]
        return False, reason


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_INT,
    para_check.REQUIRED_ATTR_STR,
    para_check.KERNEL_NAME,
)
def conv3d_backprop_input_d(filters, # pylint: disable=R0913,R0914
                            out_backprop, y_input, input_size, strides,
                            pads, dilations=(1, 1, 1, 1, 1), groups=1,
                            data_format="NDHWC",
                            kernel_name="conv3d_backprop_input"):
    """
    algorithm: Conv3d_backprop_input

    Parameters
    ----------
    filters: A dict with keys(shape and dtype)
        Input weight tensor

    out_backprop: A dict with keys(shape and dtype)
        Gradients tensor

    y_input: A dict with keys(shape and dtype)
        Conv3d_backprop_input output tensor, dtype must be assigned

    input_size: The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides: A tuple/list of 5 integers
        Filter move stride

    pads: A tuple/list of 6 integers
        [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
        filter expand size of dilated conv3d_backprop_input, default value (1, 1, 1, 1, 1)

    groups: Int of blocked connections from input channels to output channels
        Default value 1

    data_format: The data format of the input and output data
        Default format "NDHWC"

    kernel_name: Str
        Kernel name, default value is "conv3d_backprop_input"

    Returns
    -------
    None
    """

    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_size
    ori_shape_strides = strides
    ori_shape_dilations = dilations

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    shape_filters, shape_out_backprop, shape_strides, shape_dilations, shape_res = \
        _get_ndhwc_shape(ori_format_filters,
                         ori_shape_filters,
                         ori_format_out_backprop,
                         ori_shape_out_backprop,
                         ori_shape_strides,
                         ori_shape_dilations,
                         ori_format_res,
                         ori_shape_res)

    _conv3d_backprop_input_cce(shape_filters,
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
def check_conv3dbp_input_params(shape_filter,# pylint:disable=R0913,R0914,R0915
                                shape_out_backprop,
                                input_sizes, strides, pads, groups, dilations,
                                filter_dtype, out_backprop_dtype,
                                res_dtype, kernel_name):
    """
    The params check function of conv3d backprop input

    Parameters
    -------------------------
    shape_filter : The shape of filter
        5-D with shape (depth, height, weight, batch, channels)

    shape_out_backprop : The shape of gradients
        5-D with shape [batch, depth, height, weight,channels]

    input_sizes : The shape of feature map
        5-D with shape [batch, depth, height, weight, channels]

    strides : A list/tuple of ints. The stride of the sliding window

    pads : A list/tuple of ints or str

    groups : Int of blocked connections from input channels to output channels

    dilations : An optional list/tuple of ints

    filter_dtype : The dtype of filter data

    out_backprop_dtype : The dtype of gradients data

    res_dtype : The dtype of result(De/Dx) data

    kernel_name : Cce kernel name

    Returns
    -----------------------
    All transformed params
    """
    def _check_attr_range(attr_name, attr_value, attr_min, attr_max):
        if attr_value < attr_min or attr_value > attr_max:
            error_manager_cube.raise_err_attr_range_invalid("conv3d",
                "[{},{}]".format(attr_min, attr_max),
                attr_name,
                str(attr_value))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype is None:
            bit_ratio = _BIT_RATIO_DICT.get("float16")
        else:
            bit_ratio = _BIT_RATIO_DICT.get(dtype)
        if attr_value * bit_ratio > _DATA_SIZE_MAX:
            dict_args = {
                'errCode': 'E60020',
                'attr_name': attr_name,
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_ub_limitation():
        w_value = dedy_w * stride_w

        aub_dedy_size_min = dedy_w * _BLOCK_SIZE * 2
        aub_filling_size_min = w_value * _BLOCK_SIZE * 2
        cub_size_min = _BLOCK_SIZE * _BLOCK_SIZE * 2
        ub_size = tbe_platform.get_soc_spec("UB_SIZE")

        if (aub_dedy_size_min + aub_filling_size_min + cub_size_min) > ub_size:
            dict_args = {
                'errCode': 'E60119'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_l1_limitation():
        w_value = dedy_w * stride_w
        if fmap_w > _BLOCK_SIZE:
            h_value_max = filter_h_dilation + 1
        elif _BLOCK_SIZE % fmap_w == 0:
            h_value_max = filter_h_dilation + _BLOCK_SIZE // fmap_w - 1
        else:
            h_value_max = filter_h_dilation + _BLOCK_SIZE // fmap_w + 1

        a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // stride_d + 2) * _BLOCK_SIZE * 2
        b_l1_size = filter_h_dilation * filter_w_dilation * filter_d_dilation * _BLOCK_SIZE * _BLOCK_SIZE * 2
        l1_size = tbe_platform.get_soc_spec("L1_SIZE")
        if (a_l1_size + b_l1_size) > l1_size:
            dict_args = {
                'errCode': 'E60026'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_shape_error():
        fmap_h_padding = fmap_h + pad_up + pad_down
        fmap_w_padding = fmap_w + pad_left + pad_right
        fmap_d_padding = fmap_deep + pad_head + pad_tail
        # Check Batch Dimension
        if fmap_channel != filter_channel * groups:
            error_manager_cube.raise_err_specific("conv3d",
                    "Shape error: Fmap's C must be equal to Filter'C * groups.")

        if dedy_channel != filter_batch:
            error_manager_cube.raise_err_specific("conv3d",
                    "Shape error: Dedy's C must be equal to Filter'N.")

        if fmap_batch != dedy_batch:
            error_manager_cube.raise_err_two_paras('E62503', 'conv3d',
                str(dedy_batch), str(fmap_batch))

        # Check HWD dimension
        if filter_h_dilation > fmap_h_padding:
            error_manager_cube.raise_err_three_paras('E62507', 'conv3d', 'H',
                str(filter_h_dilation), str(fmap_h_padding))

        if filter_w_dilation > fmap_w_padding:
            error_manager_cube.raise_err_three_paras('E62507', 'conv3d', 'W',
                str(filter_w_dilation), str(fmap_w_padding))

        if filter_d_dilation > fmap_d_padding:
            error_manager_cube.raise_err_three_paras('E62507', 'conv3d', 'D',
                str(filter_d_dilation), str(fmap_d_padding))

        if ((fmap_h_padding - filter_h_dilation) // stride_h + 1) != dedy_h:
            dict_args = {
                'errCode': 'E60024',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
        if ((fmap_w_padding - filter_w_dilation) // stride_w + 1) != dedy_w:
            dict_args = {
                'errCode': 'E60025',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))
        if ((fmap_d_padding - filter_d_dilation) // stride_d + 1) != dedy_deep:
            dict_args = {
                'errCode': 'E62508',
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    # Base check, Mainly required by interface appearance
    # ===========================================================
    # para_check check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_filter, _CONV_BACKPROP_SHAPE_DIM,
                                _CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(shape_out_backprop, _CONV_BACKPROP_SHAPE_DIM,
                                _CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(input_sizes, _CONV_BACKPROP_SHAPE_DIM,
                                _CONV_BACKPROP_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(strides, _STRIDES_SHAPE_DIM, _STRIDES_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(dilations, _DILATIONS_SHAPE_DIM,
                                _DILATIONS_SHAPE_DIM, _DEFAULT_MAX_SHAPE_NUM)

    # pads check
    if isinstance(pads, (tuple, list)) and len(pads) != _CONV_BACKPROP_PAD_SHAPE_DIM:
        error_manager_cube.raise_err_one_para('E62501', 'conv3d', 'pads')

    if isinstance(pads, str) and pads not in ['SAME', 'VALID']:
        dict_args = {
            'errCode': 'E60000',
            'param_name': 'pads',
            'expected_value':'SAME or VALID',
            'input_value': str(pads),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    _, dilation_d, dilation_h, dilation_w, _ = dilations
    if dilation_d != 1:
        error_manager_cube.raise_err_specific("conv3d",
            "dilation in D dimension only supports 1.")

    # dtype check
    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    para_check.check_dtype_rule(filter_dtype, ('float16'), "filter")
    para_check.check_dtype_rule(out_backprop_dtype, ('float16'), "out_backprop")
    para_check.check_dtype_rule(res_dtype, ('float16'), "output")

    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_deep, fmap_h, fmap_w, fmap_channel = input_sizes
    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    _, stride_d, stride_h, stride_w, _ = strides

    group_dict = util_common.calculate_group(fmap_channel,
                                             filter_batch, groups, _C0_SIZE, _C0_SIZE)

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    filter_d_dilation = (filter_depth - 1) * dilation_d + 1

    if pads == 'SAME':
        pad_h = util_common.align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = util_common.align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_d = util_common.align(fmap_deep, stride_d) - stride_d + filter_d_dilation - fmap_deep
        pad_d = max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head

        pads = [pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = _PADDING_VAILD
    # pads compute
    pads = list(pads)
    pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads

    if fmap_h != 1 and fmap_w == 1:
        # Chip Design demand fmap_w must larger than 2 when fmap != 1
        error_manager_cube.raise_err_one_para('E62006', 'conv3d',
            'Chip Design demand input_size_w must >=2 when input_size_h != 1')

    # filter value limit
    _check_attr_range("filter's H", filter_h, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's W", filter_w, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range("filter's D", filter_depth, _FILTER_HW_MIN, _FILTER_D_MAX)

    _check_attr_range("filter H*W", filter_h * filter_w, _FILTER_HW_MIN,
                      _FILTER_HW_SIZE)

    _check_attr_range("filter H*W*D", filter_h * filter_w * filter_depth,
                      _FILTER_HW_MIN, _KHWD_COEFF)

    # Fmap value limit
    _check_attr_range("Fmap's H", fmap_h, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range("Fmap's W", fmap_w, _FMAP_HW_MIN, _FMAP_HW_MAX)

    # stride value limit
    _check_attr_range("stride's H", stride_h, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's W", stride_w, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range("stride's H*W",
                      stride_h * stride_w, _STRIDE_HW_MIN, _STRIDE_SIZE_MAX)
    _check_attr_range("stride's H*W*D", stride_h * stride_w * stride_d,
                      _STRIDE_HW_MIN, _STRIDE_SIZE_HWD_MAX)

    # dilation value limit
    _check_attr_range("dilation's H", dilation_h, _DILATION_HW_MIN, _DILATION_HW_MAX)
    _check_attr_range("dilation's W", dilation_w, _DILATION_HW_MIN, _DILATION_HW_MAX)

    # Dedy value limit
    _check_attr_range("Dedy's H after expands", dedy_h * stride_h,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)
    _check_attr_range("Dedy's W after expands", dedy_w * stride_w,
                      _DEDY_HW_MIN, _DEDY_HW_MAX)

    _check_shape_error()

    if stride_h > 1 or stride_w > 1:
        _check_ub_limitation()

    _check_l1_limitation()
    # check shape size, 64 bits limitation
    # ===========================================================

    fmap_size = fmap_batch * util_common.align(fmap_channel, _C0_SIZE) * fmap_deep * fmap_h * fmap_w
    dedy_size = dedy_batch * util_common.align(dedy_channel, _C0_SIZE) * dedy_deep * dedy_h * dedy_w
    filter_size = util_common.align(filter_batch, _C0_SIZE) * util_common.align(
        filter_channel, _C0_SIZE) * filter_depth * filter_h * filter_w
    _check_64bits_limitation("input", fmap_size, dtype=res_dtype)
    _check_64bits_limitation("out_backprop", dedy_size,
                             dtype=out_backprop_dtype)
    _check_64bits_limitation("filter", filter_size, dtype=filter_dtype)

    result = (shape_filter, shape_out_backprop, input_sizes, strides,
              pads, dilations, filter_dtype, out_backprop_dtype,
              res_dtype, kernel_name, group_dict)
    return result


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple), int, (list, tuple),
                             str, str, str, str)
def _conv3d_backprop_input_cce(shape_filter, # pylint: disable=R0913,R0914
                              shape_out_backprop, input_sizes,
                              strides, pads, groups, dilations=(1, 1, 1, 1, 1),
                              filter_dtype='float16',
                              out_backprop_dtype='float16',
                              res_dtype='float16',
                              kernel_name="_conv3d_backprop_input_cce"):
    """
    Topi interface of conv3d backprop input

    Parameters
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

    kernel_name : Cce kernel name. Default value is "_conv3d_backprop_input_cce

    Returns
    ----------
    None
    """
    def _conv3dbp_input_achieve_with_tvm():
        dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
        shape_filter_ncdhw = [filter_batch,
                              filter_channel, filter_depth, filter_h, filter_w]

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

        dedx = tbe.conv3d_backprop_input(filter=filters,
                                         out_backprop=dedy,
                                         filter_size=shape_filter_ncdhw,
                                         input_size=input_sizes,
                                         para_dict=para_dict)
        tensor_list = [filters, dedy, dedx]

        with tvm.target.cce():
            sch = tbe.auto_schedule(dedx)

        config = {
            "name": kernel_name,
            "tensor_list": tensor_list,
            "dummy_placeholder": True
        }
        tbe.build(sch, config)


    res = check_conv3dbp_input_params(shape_filter, shape_out_backprop,
                                      input_sizes, strides, pads, groups,
                                      dilations, filter_dtype,
                                      out_backprop_dtype,
                                      res_dtype, kernel_name)
    (shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,
     filter_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w, filter_channel, filter_batch = shape_filter
    pads = list(pads)

    shape_dedy = (dedy_batch,
                  dedy_deep,
                  util_common.ceil(dedy_channel, _C0_SIZE), dedy_h, dedy_w, _C0_SIZE)

    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]

    shape_filter_frac = (real_g * filter_depth * cin1_g * filter_h * filter_w,
                         cout_g // _C0_SIZE, _C0_SIZE, _C0_SIZE)
    _conv3dbp_input_achieve_with_tvm()
