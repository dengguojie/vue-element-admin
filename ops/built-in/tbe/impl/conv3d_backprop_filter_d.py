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
conv3d_backprop_filter_d
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.lang.cce.te_compute import conv3d_backprop_filter_compute as conv3d_bp_dw
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.errormgr import error_manager_cube as cube_err
from te import tvm
from impl.util import util_common


# the dim of shape in CONV3D_BACKPROP must be 5
_CONV3D_BACKPROP_SHAPE_DIM = 5
# the dim of strides in CONV3D_BACKPROP must be 3
_STRIDES_SHAPE_DIM = 3
# the dim of pads in CONV3D_BACKPROP must be 6
_PADDING_SHAPE_DIM = 6
# the min x or y dim for cube mul
_C0 = 16
# fmapH, fmapW must be in [1,4096]
_FMAP_HW_MAX = 4096
_FMAP_HW_MIN = 1

# DeDy H,W must be in [1,4096]
_DEDY_HW_MAX = 4096
_DEDY_HW_MIN = 1

# filterH, filterW must be in [1,255]
_FILTER_HW_MAX = 255
_FILTER_HW_MIN = 1

# stride must be in [1,63]
_STRIDE_HW_MAX = 63
_STRIDE_HW_MIN = 1

# pad must be in [0,255]
_PAD_MAX = 255
_PAD_MIN = 0

# dilation must be in [1,255]
_DILATION_MIN = 1
_DILATION_MAX = 255
# dilation in the D dimension only supports 1 now
_DILATION_D = 1

# the max num of each axis of shape
_DEFAULT_MAX_SHAPE_NUM = 1000000

# the max size is 2**63-1
_DATA_SIZE_MAX = 9223372036854775807

# the bytes length of several dtype
_BIT_RATIO_DICT = {
    "int32": 4,
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "int8": 1,
    "uint4": 0.5,
    "int4": 0.5
}

# pads valid mode to be [0, 0, 0, 0]
_PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# If pads is string , only support "SAME" or "VALID"
_PADDING_SUPPORT = ('SAME', 'VALID')


def check_supported(x_dict,
                    out_backprop,
                    y_dict,
                    filter_size,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1, 1),
                    groups=1,
                    data_format='NDHWC',
                    kernel_name="conv3d_backprop_filter"):
    """
    The H and W dimension of dilation should be in range [1, 255]
    The D,H or W dimension of the filter should be in range [1, 255]
    The padding in each dimension should be in range [0, 255]
    The feature map's H,W and D dimension should be in [1, 4096]
    The out_backprop's H and W dimension should be in [1, 4096]
    If filter h,w in [1,11] and fmap h/w after padding equals to filter h/w, the out_backprop's h,w,d dimension should be in range [2, 4096]
    The D,H or W dimension of the stride should be in range [1, 63]

    The groups should <= the feature map's and the out_backprop's channel dimension
    Feature map's channel dimension or out_backprop's channel dimension must be divisible by groups
    The channel dimension of feature map should = the filter's channel dimension * groups
    The out_backprop's channel dimension should = the filter's batch dimension
    The feature map's batch dimension should = the out_backprop's batch dimensionss
    The D,H or W dimension of the feature map after padding should >= the filter's corresponding dimension after dilation
    The padding in each dimension should < the filter's corresponding dimension after dilation
    The out_backprop's H * stride's H should < 4096
    The out_backprop's W * stride's W should < 4096
    If the output H dimension is not 1, the output W dimension should >= 2

    The data in L1 buffer should <= the chip's L1 buffer size
    """
    
    try:
        processed_res = _process_input(x_dict, out_backprop, y_dict, filter_size,
                                       strides, pads, dilations, groups, data_format, kernel_name)
                                   
        shape_x, shape_out_backprop, shape_res, strides, pads, groups, dilations, x_dtype,\
                                    out_backprop_dtype, res_dtype, kernel_name = processed_res
        _check_conv3dbp_filter_params(shape_x, shape_out_backprop,
            shape_res, strides, pads, groups, dilations, x_dtype,
            out_backprop_dtype, res_dtype, kernel_name)
        return True
    except Exception as e:
        print(e)
        return False
    

def _process_input(x_dict,
                   out_backprop,
                   y_dict,
                   filter_size,
                   strides,
                   pads,
                   dilations=(1, 1, 1, 1, 1),
                   groups=1,
                   data_format='NDHWC',
                   kernel_name="conv3d_backprop_filter"):
    """
    Process input Data for Conv3d_backprop_filter_d
    """
    def _check_inputs_rules():
        if (not isinstance(ori_shape_out_backprop, (tuple, list))) or len(ori_shape_out_backprop) != 5:
            args_dict = {
                'errCode': 'E62002',
                'param_name': 'out_backprop_shape',
                'expected_type': '[{}, {}]'.format('tuple', 'list'),
                'expected_length': '5',
                'type': str(type(ori_shape_out_backprop)),
                'length': str(len(ori_shape_out_backprop))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if (not isinstance(ori_shape_x, (tuple, list))) or len(ori_shape_x) != 5:
            args_dict = {
                'errCode': 'E62002',
                'param_name': 'input_shape',
                'expected_type': '[{}, {}]'.format('tuple', 'list'),
                'expected_length': '5',
                'type': str(type(ori_shape_x)),
                'length': str(len(ori_shape_x))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if (not isinstance(ori_shape_res, (tuple, list))) or len(ori_shape_res) != 5:
            args_dict = {
                'errCode': 'E62002',
                'param_name': 'res_shape',
                'expected_type': '[{}, {}]'.format('tuple', 'list'),
                'expected_length': '5',
                'type': str(type(ori_shape_res)),
                'length': str(len(ori_shape_res))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if len(strides) != 3:
            args_dict = {
                'errCode': 'E60006',
                'param_name': 'strides',
                'expected_length': '3',
                'length': str(len(strides))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if len(filter_size) != 5:
            args_dict = {
                'errCode': 'E60006',
                'param_name': 'filter_size',
                'expected_length': '5',
                'length': str(len(filter_size))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if len(dilations) != 5:
            args_dict = {
                'errCode': 'E60006',
                'param_name': 'dilations',
                'expected_length': '5',
                'length': str(len(dilations))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if isinstance(pads, str) and pads not in _PADDING_SUPPORT:
            args_dict = {
                'errCode': 'E60021',
                'expected_pad_mode': '[{}, {}]'.format('SAME', 'VALID'),
                'actual_pad_mode': str(pads)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if isinstance(pads, (tuple, list)) and len(pads) != 6:
            cube_err.raise_err_one_para('E62501', 'conv3d_backprop_filter', 'pads')

    ori_shape_x = x_dict.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = y_dict.get("ori_shape")

    x_dtype = x_dict.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_dict.get("dtype")

    ori_format_x = x_dict.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_format_res = y_dict.get("ori_format")

    if len(strides) == 5:
        d_index = data_format.find('D')
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[d_index], strides[h_index], strides[w_index]]

    _check_inputs_rules()

    input_format_list = ("NDHWC", "NCDHW")
    shape_x = _normalize_shape_ndchw(ori_shape_x,
                                     ori_format_x,
                                     input_format_list,
                                     'x')
    shape_out_backprop = _normalize_shape_ndchw(
                            ori_shape_out_backprop,
                            ori_format_out_backprop,
                            input_format_list,
                            'out_backprop')
    dilations = _normalize_shape_ndchw(dilations,
                                       ori_format_out_backprop,
                                       input_format_list,
                                       'dilations')

    res_format_list = ("NDHWC", "NCDHW", "DHWCN")
    shape_res = _normalize_shape_ndchw(ori_shape_res,
                                       ori_format_res,
                                       res_format_list,
                                       'y') 
    return (shape_x, shape_out_backprop, shape_res,
            strides, pads, groups, dilations, x_dtype,
            out_backprop_dtype, res_dtype, kernel_name)


def _normalize_shape_ndchw(ori_shape, ori_format, format_list,
                           param_name='input_param'):
    """
    Normalizing the shape to NDCHW
    """
    if ori_format not in format_list:
        args_dict = {
            'errCode': 'E60008',
            'param_name': param_name,
            'expected_format_list': ','.join(format_list),
            'format': ori_format
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))

    n_index = ori_format.find('N')
    d_index = ori_format.find('D')
    c_index = ori_format.find('C')
    h_index = ori_format.find('H')
    w_index = ori_format.find('W')

    new_shape = [
        ori_shape[n_index], ori_shape[d_index],
        ori_shape[c_index], ori_shape[h_index],
        ori_shape[w_index]
    ]

    return new_shape


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
def conv3d_backprop_filter_d(x_dict,
                             out_backprop,
                             y_dict,
                             filter_size,
                             strides,
                             pads,
                             dilations=(1, 1, 1, 1, 1),
                             groups=1,
                             data_format='NDHWC',
                             kernel_name="conv3d_backprop_filter"):
    """
    algorithm: conv3d_backprop_filter

    Parameters
    ----------
    x_dict: A dict with keys(shape and dtype)
        Input feature map tensor

    out_backprop: A dict with keys(shape and dtype)
        Gradients tensor

    y_dict: A dict with keys(shape and dtype)
        Output tensor, dtype must be assigned

    filter_size: The shape of filter
        5-D with shape [batch, depth, channels, height, weight]

    strides: A tuple/list of 5 integers
        Filter move stride

    pads: A tuple/list of 6 integers
        [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]

    dilations: A tuple/list of 5 integers
        Filter expand size of dilated conv3d_backprop_filter, default value is (1, 1, 1, 1, 1)

    groups: Int
        Param for group covolution, default value is 1

    data_format: Str
        An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC"

    kernel_name: Str
        Kernel name, default value is "conv3d_backprop_filter"

    Returns
    -------
    None
    """
    processed_res = _process_input(x_dict, out_backprop, y_dict, filter_size,
                                   strides, pads, dilations, groups, data_format, kernel_name)
                                   
    shape_x, shape_out_backprop, shape_res, strides, pads, groups, dilations, x_dtype,\
                                out_backprop_dtype, res_dtype, kernel_name = processed_res

    _conv3d_backprop_filter_cce(shape_x, shape_out_backprop, shape_res,
                                strides, pads, groups, dilations, x_dtype,
                                out_backprop_dtype, res_dtype, kernel_name)


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple), int,
                             (list, tuple), str, str, str, str)
def _check_conv3dbp_filter_params(
    shape_x, shape_out_backprop,
    filter_sizes, strides, pads, groups, dilations, x_dtype,
    out_backprop_dtype, res_dtype, kernel_name):
    """
    The params check function of conv3d_backprop_filter

    Parameters:
    ----------
    shape_x : The shape of feature map
        5-D [batch, depth, channels, height, weight]

    shape_out_backprop : The shape of gradients
        5-D [batch, depth,channels, height, weight]

    filter_sizes : The shape of filter
        5-D [batch, depth, channels, height, weight]

    strides : The stride of the sliding window. A list/tuple of ints

    pads : A list/tuple of 6 integers or str

    dilations : An optional list/tuple of ints. Default value is (1, 1, 1, 1, 1)

    x_dtype : Feature map data dtype. Default value is float16

    out_backprop_dtype : Gradients data dtype. Default value is float16

    res_dtype : Result(De/Dw) data dtype. Default value is float32

    kernel_name : Kernel name of cce
        Default value is "_conv3d_backprop_filter_cce"

    Returns
    ----------
    All transformed params.
    """
    def _check_attr_range_dw(attr_name, attr_value, attr_min, attr_max):
        if attr_value < attr_min or attr_value > attr_max:
            dict_args = {
                'errCode': 'E60011',
                'range': '[{},{}]'.format(attr_min, attr_max),
                'attr_name': attr_name,
                'value': str(attr_value)
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype:
            bit_ratio = _BIT_RATIO_DICT.get(dtype)
        else:
            bit_ratio = _BIT_RATIO_DICT.get("float16")
        if attr_value * bit_ratio > _DATA_SIZE_MAX:
            args_dict = {'errCode': 'E60020', 'attr_name': attr_name}
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

    # First : Base check, Mainly required by interface appearance
    # ===========================================================
    # para_check check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x, _CONV3D_BACKPROP_SHAPE_DIM,
                                _CONV3D_BACKPROP_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(shape_out_backprop, _CONV3D_BACKPROP_SHAPE_DIM,
                                _CONV3D_BACKPROP_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(filter_sizes, _CONV3D_BACKPROP_SHAPE_DIM,
                                _CONV3D_BACKPROP_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    para_check.check_shape_rule(strides, _STRIDES_SHAPE_DIM, _STRIDES_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)

    def _check_attr_pads():
        # pads check
        if isinstance(pads, (tuple, list)) and len(pads) != _PADDING_SHAPE_DIM:
            cube_err.raise_err_one_para('E62501', 'conv3d_backprop_filter', 'pads')

        if isinstance(pads, str) and pads not in _PADDING_SUPPORT:
            args_dict = {
                'errCode': 'E60021',
                'expected_pad_mode': '[{}]'.format(_PADDING_SUPPORT),
                'actual_pad_mode': str(pads)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

    _check_attr_pads()

    # dilations check
    para_check.check_shape_rule(dilations, _CONV3D_BACKPROP_SHAPE_DIM,
                                _CONV3D_BACKPROP_SHAPE_DIM,
                                _DEFAULT_MAX_SHAPE_NUM)
    dilation_n, dilation_d, dilation_c, dilation_h, dilation_w = dilations
    _check_attr_range_dw("dilations's H", dilation_h, _DILATION_MIN,
                         _DILATION_MAX)
    _check_attr_range_dw("dilations's W", dilation_w, _DILATION_MIN,
                         _DILATION_MAX)
    _check_attr_range_dw("dilations's D", dilation_d, _DILATION_D,
                         _DILATION_D)

    if dilation_n != 1 or dilation_c != 1:
        args_dict = {
            'errCode': 'E60023',
            'dilation_n': str(dilation_n),
            'dilation_c': str(dilation_c)
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))

    # dtype check
    x_dtype = x_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    para_check.check_dtype_rule(x_dtype, ['float16'], "x")
    para_check.check_dtype_rule(out_backprop_dtype, ['float16'], "out_backprop")
    para_check.check_dtype_rule(res_dtype, ['float32', 'float16'], "output")

    # Second : Furture Check, Mainly required by SRS
    # ===========================================================
    # the relation limits between shape
    shape_x = list(shape_x)
    shape_out_backprop = list(shape_out_backprop)
    filter_sizes = list(filter_sizes)
    strides = list(strides)
    fmap_batch, fmap_d, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_d, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_d, filter_channel, filter_h, filter_w = filter_sizes
    stride_d, stride_h, stride_w = strides

    group_dict = util_common.calculate_group(fmap_channel, dedy_channel,
                                             groups, _C0, _C0)

    filter_d_dilation = (filter_d - 1) * dilation_d + 1
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    # pads compute
    if pads == 'SAME':
        pad_d = util_common.align(fmap_d, stride_d) - stride_d + filter_d_dilation - fmap_d
        pad_d = max(pad_d, 0)
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_w = util_common.align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = util_common.align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pads = [pad_front, pad_back, pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = _PADDING_VAILD
    pads = list(pads)
    pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = pads
    util_common.check_pads_value_3d(pads)

    if pad_front >= filter_d_dilation or pad_back >= filter_d_dilation:
        args_dict = {
            'errCode': 'E60013',
            'depth_of_pad': '{}, {}'.format(pad_front, pad_back),
            'depth_of_filter': '{}'.format(filter_d_dilation)
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))
    if pad_up >= filter_h_dilation or pad_down >= filter_h_dilation:
        args_dict = {
            'errCode': 'E60016',
            'h_of_filter': '{}'.format(filter_h_dilation),
            'h_of_pad': '{}, {}'.format(pad_up, pad_down)
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))
    if pad_left >= filter_w_dilation or pad_right >= filter_w_dilation:
        args_dict = {
            'errCode': 'E60017',
            'w_of_filter': '{}'.format(filter_w_dilation),
            'w_of_pad': '{}, {}'.format(pad_left, pad_right)
        }
        raise RuntimeError(args_dict,
                           error_manager_util.get_error_message(args_dict))

    fmap_w_padding = fmap_w + pad_left + pad_right
    fmap_h_padding = fmap_h + pad_up + pad_down

    # special cases
    if dedy_w < 2 and dedy_h != 1:
        # Chip Design demand dedy_w must >=2 when dedy_h != 1
        cube_err.raise_err_specific("conv3d_backprop_filter",
            "Chip Design demand dedy_w must >=2 when dedy_h != 1.")

    # Dedy value limit
    _check_attr_range_dw("Dedy's H", dedy_h, _DEDY_HW_MIN, _DEDY_HW_MAX)
    _check_attr_range_dw("Dedy's W", dedy_w, _DEDY_HW_MIN, _DEDY_HW_MAX)

    # filter value limit
    _check_attr_range_dw("filter's H", filter_h, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range_dw("filter's W", filter_w, _FILTER_HW_MIN, _FILTER_HW_MAX)
    _check_attr_range_dw("filter's D", filter_d, _FILTER_HW_MIN, _FILTER_HW_MAX)

    # Fmap value limit
    _check_attr_range_dw("Fmap's H", fmap_h, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range_dw("Fmap's W", fmap_w, _FMAP_HW_MIN, _FMAP_HW_MAX)
    _check_attr_range_dw("Fmap's D", fmap_d, _FMAP_HW_MIN, _FMAP_HW_MAX)

    # stride value limit
    _check_attr_range_dw("stride's H", stride_h, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range_dw("stride's W", stride_w, _STRIDE_HW_MIN, _STRIDE_HW_MAX)
    _check_attr_range_dw("stride's D", stride_d, _STRIDE_HW_MIN, _STRIDE_HW_MAX)

    def _check_axis_hw():
        if fmap_batch != dedy_batch:
            cube_err.raise_err_two_paras('E62503', 'conv3d_backprop_filter',
                    str(dedy_batch), str(fmap_batch))

        if dedy_channel != filter_batch:
            cube_err.raise_err_two_paras('E62504', 'conv3d_backprop_filter',
                    str(dedy_channel), str(filter_batch))

        if fmap_channel != filter_channel * groups:
            args_dict = {
                'errCode': 'E60010',
                'channel_of_x': str(fmap_channel),
                'channel_of_filter': str(filter_channel)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        if filter_w_dilation > fmap_w_padding:
            args_dict = {
                'errCode': 'E60015',
                'w_of_x': str(fmap_w_padding),
                'w_of_filter': str(filter_w_dilation)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        if filter_h_dilation > fmap_h_padding:
            args_dict = {
                'errCode': 'E60014',
                'h_of_x': str(fmap_h_padding),
                'h_of_filter': str(filter_h_dilation)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        # Third : value check, Mainly required by the convolution rule
        if ((fmap_w - filter_w_dilation + pad_left + pad_right) // stride_w +
                1) != dedy_w:
            args_dict = {'errCode': 'E60025'}
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        if ((fmap_h - filter_h_dilation + pad_up + pad_down) // stride_h +
                1) != dedy_h:
            args_dict = {'errCode': 'E60024'}
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

    _check_axis_hw()

    def _min_l1_byte():
        # Forth : L1 limitation, Mainly required by chip
        al1_min_byte = _C0 * _C0 * 2

        if dedy_w % _C0 == 0:
            bl1_min_byte = filter_h_dilation * fmap_w * _C0 * 2
        else:
            bl1_min_byte = (filter_h_dilation + stride_h) * fmap_w * _C0 * 2

        l1_size = tbe_platform.get_soc_spec("L1_SIZE")  # L1 size
        if (al1_min_byte + bl1_min_byte) > l1_size:
            args_dict = {'errCode': 'E60022'}
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

    _min_l1_byte()
    # Fifth : check shape size, 64 bits limitation
    c0_size = tbe_platform.C0_SIZE
    fmap_size = fmap_batch * fmap_d * util_common.align(fmap_channel,
                                                        c0_size) * fmap_h * fmap_w
    dedy_size = dedy_batch * dedy_d * util_common.align(dedy_channel,
                                                        c0_size) * dedy_h * dedy_w
    filter_size = util_common.align(filter_batch, c0_size) * filter_d * util_common.align(
        filter_channel, c0_size) * filter_h * filter_w
    _check_64bits_limitation("fmap_size", fmap_size, dtype=x_dtype)
    _check_64bits_limitation("dedy_size", dedy_size, dtype=out_backprop_dtype)
    _check_64bits_limitation("filter_size", filter_size, dtype=res_dtype)

    result = (shape_x, shape_out_backprop, filter_sizes, strides, pads,
              dilations, x_dtype, out_backprop_dtype, res_dtype,
              kernel_name, group_dict)
    return result


@para_check.check_input_type((list, tuple), (list, tuple), (list, tuple),
                             (list, tuple), (str, list, tuple), int,
                             (list, tuple), str, str, str, str)
def _conv3d_backprop_filter_cce(shape_x,
                                shape_out_backprop,
                                filter_sizes,
                                strides,
                                pads,
                                groups=1,
                                dilations=(1, 1, 1, 1, 1),
                                x_dtype='float16',
                                out_backprop_dtype='float16',
                                res_dtype='float32',
                                kernel_name="_conv3d_backprop_filter_cce"):
    """
    Topi interface of conv3d backprop filter

    Parameters:
    ----------
    shape_x : The shape of feature map
        5-D with shape [batch, depth, channels, height, weight]

    shape_out_backprop : The shape of gradients
        5-D with shape [batch, depth, channels, height, weight]

    filter_sizes : The shape of filter
        5-D with shape [batch, depth, channels, height, weight]

    strides : A tuple/list of 5 integers
        Filter move stride

    pads : A list/tuple of 6 integers or str

    groups : Int
        Param for group covolution, default value is 1

    dilations : An optional list/tuple of 5 integers. Default value is [1, 1, 1, 1, 1]

    x_dtype : The dtype of feature map data. Default value is float16

    out_backprop_dtype : The dtype of gradients data
        Default value is float16.

    res_dtype : The dtype of result(De/Dw) data. Default value is float32

    kernel_name : Cce kernel name
        Default value is "_conv3d_backprop_filter_cce"

    Returns
    ----------
    None
    """
    if tbe_platform.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",
                                                    "Hi3796CV300CS",
                                                    "SD3403"):
        res_dtype = "float16"

    res = _check_conv3dbp_filter_params(shape_x, shape_out_backprop,
                                       filter_sizes, strides, pads,
                                       groups, dilations, x_dtype,
                                       out_backprop_dtype, res_dtype,
                                       kernel_name)
    (shape_x, shape_out_backprop, filter_sizes, strides, pads, dilations,
     x_dtype, out_backprop_dtype, res_dtype, kernel_name, group_dict) = res

    fmap_batch, fmap_depth, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_d, dedy_channel, dedy_h, dedy_w = shape_out_backprop

    c0_size = tbe_platform.C0_SIZE  # Channel axis should be align with 16
    shape_dedy = (dedy_batch, dedy_d,
                  util_common.ceil(dedy_channel, c0_size),
                  dedy_h, dedy_w, c0_size)
    shape_fmap = (fmap_batch, fmap_depth,
                  util_common.ceil(fmap_channel, c0_size),
                  fmap_h, fmap_w, c0_size)
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    fmap = tvm.placeholder(shape_fmap, name="fmap", dtype=x_dtype)

    para_dict = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "res_dtype": res_dtype,
        "kernel_name": kernel_name,
        "group_dict": group_dict
    }

    dedw = conv3d_bp_dw.conv3d_dw(
        x=fmap,
        out_backprop=dedy,
        filter_size=filter_sizes,
        para_dict=para_dict
    )

    tensor_list_input = [fmap, dedy]
    with tvm.target.cce():
        sch = tbe.auto_schedule(dedw)

    real_outs = sch.cce_special["real_out_tensor"]
    tensor_list = tensor_list_input + real_outs

    config = {"name": kernel_name, "tensor_list": tensor_list, "dummy_placeholder": True}

    tbe.cce_build_code(sch, config)
