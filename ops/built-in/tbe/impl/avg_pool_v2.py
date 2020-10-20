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
avg_pool_v2
"""
import te.lang.cce as tbe
from te import tvm
from te.utils import op_utils
from te.utils import para_check


def _get_fusion_params(input_data, output_data, is_fused_compute=True):
    """
    function to get fusion params

    Parameters
    ----------
    input_data: tensor of input_data

    output_data: dict of output_data

    is_fused_compute: fused or not

    Returns
    -------
    fusion_params: dict fusion_params
    """
    # l1 fusion params assign
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in input_data.op.attrs else -1
    in_l1_flag = input_data.op.attrs["addr_type"].value == 1 \
        if "addr_type" in input_data.op.attrs else False
    in_valid_shape = input_data.op.attrs["valid_shape"] \
        if "valid_shape" in input_data.op.attrs else []
    in_slice_offset = input_data.op.attrs["slice_offset"] \
        if "slice_offset" in input_data.op.attrs else []
    in_select_read_flag = bool(in_valid_shape)
    in_split_index = input_data.op.attrs["split_index"].value \
        if "split_index" in input_data.op.attrs else 0
    out_l1_flag = output_data.get("addr_type") == 1
    fusion_params = {"is_fused_compute": is_fused_compute,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "out_l1_flag": out_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "in_split_index": in_split_index,
                     "in_slice_offset": in_slice_offset}

    return fusion_params


# pylint: disable=locally-disabled,too-many-arguments,too-many-statements,redefined-builtin
# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _check_window_rule(ksize, strides, pads, data_format):
    """
    check ksize and strides of window in pooling
    """
    if len(pads) != 4:
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool_v2'
        error_info['param_name'] = 'pads'
        error_info['min_value'] = '4'
        error_info['max_value'] = '4'
        error_info['real_value'] = len(pads)
        raise RuntimeError(error_info,
                           "In op[%s], the num of dimensions of input[%s] should"
                           " be in the range of [%s, %s], but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))
    if data_format in ("NHWC",):
        if len(ksize) != 4:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_012
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = 'ksize'
            error_info['min_value'] = '4'
            error_info['max_value'] = '4'
            error_info['real_value'] = len(ksize)
            raise RuntimeError(error_info,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s],"
                               "but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['min_value'], error_info['max_value'],
                                error_info['real_value']))

        elif ksize[0] != 1 or ksize[3] != 1:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_000
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = ",".join(("ksize[1]", "ksize[3]"))
            error_info['expected_value'] = '1'
            error_info['real_value'] = ",".join((str(ksize[1]), str(ksize[3])))
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be [%s], "
                               "but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['expected_value'],
                                error_info['real_value']))
        if len(strides) != 4:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_012
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = 'strides'
            error_info['min_value'] = '4'
            error_info['max_value'] = '4'
            error_info['real_value'] = len(strides)
            raise RuntimeError(error_info,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s],"
                               "but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['min_value'], error_info['max_value'],
                                error_info['real_value']))
        elif strides[0] != 1 or strides[3] != 1:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_000
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = ",".join(("strides[1]", "strodes[3]"))
            error_info['expected_value'] = '1'
            error_info['real_value'] = ",".join((str(strides[1]), str(strides[3])))
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['expected_value'],
                                error_info['real_value']))
    elif data_format in ("NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_012
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = 'ksize'
            error_info['min_value'] = '4'
            error_info['max_value'] = '4'
            error_info['real_value'] = len(ksize)
            raise RuntimeError(error_info,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s],"
                               "but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['min_value'], error_info['max_value'],
                                error_info['real_value']))
        elif ksize[0] != 1 or ksize[1] != 1:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_000
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = ",".join(("ksize[0]", "ksize[1]"))
            error_info['expected_value'] = '1'
            error_info['real_value'] = ",".join((str(ksize[0]), str(ksize[1])))
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['expected_value'],
                                error_info['real_value']))
        if len(strides) != 4:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_012
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = 'strides'
            error_info['min_value'] = '4'
            error_info['max_value'] = '4'
            error_info['real_value'] = len(strides)
            raise RuntimeError(error_info,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s], but"
                               "actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['min_value'], error_info['max_value'],
                                error_info['real_value']))
        elif strides[0] != 1 or strides[1] != 1:
            error_info = {}
            error_info['errCode'] = op_utils.OP_ERROR_CODE_000
            error_info['op_name'] = 'avg_pool_v2'
            error_info['param_name'] = ",".join(("strides[0]", "strides[1]"))
            error_info['expected_value'] = '1'
            error_info['real_value'] = ",".join((str(strides[1]), str(strides[1])))
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['expected_value'],
                                error_info['real_value']))
    else:
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool_v2'
        error_info['param_name'] = 'x'
        error_info['excepted_format_list'] = ",".join(("NC1HWC0", "NCHW", "NHWC"))
        error_info['format'] = data_format
        raise RuntimeError(error_info,
                           "In op[%s], the format[%s] of input"
                           "should be one of [%s],"
                           "but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'],
                              error_info['excepted_format_list'],
                              error_info['format']))


def _check_pads(pads, ksize_h, ksize_w):
    """
    check pads
    """
    if pads[0] >= ksize_h or pads[1] >= ksize_h:
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_002
        error_info['op_name'] = 'avg_pool_v2'
        error_info['param_name'] = ",".join(("pads[0]", "pads[1]"))
        error_info['min_value'] = '0'
        error_info['max_value'] = str(ksize_h-1)
        error_info['value'] = ",".join((str(pads[0]), str(pads[1])))
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s] should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['value']))
    if pads[2] >= ksize_w or pads[3] >= ksize_w:
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_002
        error_info['op_name'] = 'avg_pool_v2'
        error_info['param_name'] = ",".join(("pads[2]", "pads[3]"))
        error_info['min_value'] = '0'
        error_info['max_value'] = str(ksize_w-1)
        error_info['value'] = ",".join((str(pads[2]), str(pads[3])))
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s] should be in the range of [%s, %s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['value']))


def _get_corrected_pad(input_pad):
    """
    algorithm:
    get corrected pad value

    Parameters
    ----------
    input_pad: the value of pad

    Returns
    -------
    output_pad: the value of pad
    """
    if input_pad < 0:
        output_pad = 0
    else:
        output_pad = input_pad
    return output_pad


def _avg_pool_v2_check_rule(input_shape, input_dtype, output_dtype, input_format, ksize, strides,
                            pads, data_format, kernel_name):
    """
    function to check params

    Parameters
    ----------
    input_shape: shape of input_data

    input_dtype: dtype of input_data

    output_dtype: dtype of output_data

    input_format: format of input

    ksize: the window of avg_pool_v2

    strides: the stride of avg_pool_v2 window

    pads: padding value when padding_mode is CALCULATED

    data_format: NCHW default

    kernel_name: cce kernel name

    Returns
    -------
    None

    """
    # check input and output
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ["float16"])
    para_check.check_dtype(output_dtype, ["float16"])

    _check_window_rule(ksize, strides, pads, data_format)


def _calculate_pads(padding, input_h, input_w, stride_h, stride_w, ksize_h, ksize_w, dilations, pads, ceil_mode):
    """
    function to calculate pad value
    """
    if padding == "SAME":
        output_h = (input_h + stride_h - 1) // stride_h
        output_w = (input_w + stride_w - 1) // stride_w
        pad_row = (output_h - 1) * stride_h + ((ksize_h - 1) * dilations[0] + 1) - input_h
        pad_col = (output_w - 1) * stride_w + ((ksize_w - 1) * dilations[1] + 1) - input_w

        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left

        pad_top = _get_corrected_pad(int(pad_top))
        pad_bottom = _get_corrected_pad(int(pad_bottom))
        pad_left = _get_corrected_pad(int(pad_left))
        pad_right = _get_corrected_pad(int(pad_right))

        pad = (pad_top, pad_bottom, pad_left, pad_right)

    elif padding == "CALCULATED":
        pad_top, pad_bottom, pad_left, pad_right = pads

        if ceil_mode:
            ho = (input_h - ksize_h + pad_top + pad_bottom + stride_h - 1) // stride_h + 1
            wo = (input_w - ksize_w + pad_left + pad_right + stride_w - 1) // stride_w + 1
            pad_bottom = (ho - 1) * stride_h + ksize_h - input_h - pad_top
            pad_right = (wo - 1) * stride_w + ksize_w - input_w - pad_left
        else:
            ho = (input_h - ksize_h + pad_top + pad_bottom) // stride_h + 1
            wo = (input_w - ksize_w + pad_left + pad_right) // stride_w + 1
            pad_bottom = _get_corrected_pad((ho - 1) * stride_h + ksize_h - input_h - pad_top)
            pad_right = _get_corrected_pad((wo - 1) * stride_w + ksize_w - input_w - pad_left)

        pad = (pad_top, pad_bottom, pad_left, pad_right)

    else:
        pad = (0, 0, 0, 0)

    return pad


def avg_pool_v2_compute1(x, y, ksize, strides, padding="VALID", data_format="NHWC",
                         is_fused_compute=True, kernel_name="avg_pool_v2"):
    """
    function of avg_pool_v2 compute

    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support VALID, SAME and CALCULATED

    data_format : str, default = "NCHW"

    is_fused_compute : fuse or not

    kernel_name : cce kernel name, default value is "avg_pool_v2"

    Returns
    -------
    res : output of avg_pool_v2
    """
    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    window = list(window)
    stride = list(stride)

    # l1 fusion and l2 fusion
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in x.op.attrs else -1
    fusion_params = _get_fusion_params(x, y, is_fused_compute)
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    in_valid_shape = fusion_params.get("in_valid_shape")
    in_slice_offset = fusion_params.get("in_slice_offset")

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       x(n, c1, h + in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=x.op.attrs)
        res = tbe.pooling2d(select_tensor_in, window, stride, "AVG", padding, fusion_params=fusion_params)
    elif l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0:
                                         x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = tbe.pooling2d(l1_width_fusion_in, window, stride, "AVG", padding, fusion_params=fusion_params)
    else:
        res = tbe.pooling2d(x, window, stride, "AVG", padding, fusion_params=fusion_params)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def avg_pool_v2(x, filter, y, ksize, strides, padding="CALCULATED", pads=(0, 0, 0, 0), data_format="NCHW",
                global_pooling=False, ceil_mode=False, exclusive=True, kernel_name="avg_pool_v2"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    filter : assist matrix

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support VALID, SAME and CALCULATED

    pads : padding value when padding_mode is CALCULATED

    data_format : str, default = "NCHW"

    global_pooling : global pooling or not

    ceil_mode : use ceil or floor to calculate ho and wo when padding_mode is CALCULATED

    exclusive : ignore padding area or not when calculating the average

    kernel_name : cce kernel name, default value is "avg_pool_v2"

    Returns
    -------
    None
    """
    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    input_format = x.get("format")

    # check others parameter
    _avg_pool_v2_check_rule(input_shape, input_dtype, output_dtype, input_format, ksize, strides,
                            pads, data_format, kernel_name)

    # set tensor attrs, during L1 fusion these attrs will assign by te_fusion
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    split_index = x.get("split_index", 0)
    l1_fusion_type = x.get("L1_fusion_type", -1)
    attr = {"addr_type": addr_type,
            "valid_shape": valid_shape,
            "slice_offset": slice_offset,
            "split_index": split_index,
            "L1_fusion_type": l1_fusion_type}
    is_l1fusion = l1_fusion_type in (0, 1)

    input_h = input_shape[2]
    input_w = input_shape[3]
    if data_format in ("NHWC",):
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        stride_h = strides[1]
        stride_w = strides[2]
    else:
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        stride_h = strides[2]
        stride_w = strides[3]
    stride = [stride_h, stride_w]

    # compute
    # create tensor_in
    tensor_in = tvm.placeholder(input_shape, name="tensor_in", dtype=input_dtype, attrs=attr)

    if global_pooling:
        ksize = list(ksize)
        if data_format in ("NHWC",):
            ksize[1] = input_h
            ksize[2] = input_w
        else:
            ksize[2] = input_h
            ksize[3] = input_w
        padding = 'VALID'

    if list(pads) == [0, 0, 0, 0]:
        if ksize_h == input_h and ksize_w == input_w:
            padding = 'VALID'

    if filter is not None:
        filter_shape = filter.get("shape")
        filter_dtype = filter.get("dtype").lower()

        filter_shape_5d = filter_shape[0], ksize_h, ksize_w, 16, 16

        filter_in = tvm.placeholder(filter_shape_5d, name="filter_in", dtype=filter_dtype, attrs=attr)

        dilations = (1, 1)
        dsl_flag = False

        pad = _calculate_pads(padding, input_h, input_w, stride_h, stride_w, ksize_h, ksize_w,
                              dilations, pads, ceil_mode)

        _check_pads(pad, ksize_h, ksize_w)

        res = tbe.te_compute.depthwise_conv2d_compute(
            tensor_in, filter_in, output_dtype.lower(), stride, pad, dilations,
            {"bias_tensor": None, "dsl_flag": dsl_flag, "offset_x": 0}, None, kernel_name)

        tensor_list = [tensor_in, filter_in, res]
    else:
        res = avg_pool_v2_compute1(tensor_in, y, ksize, strides, padding, data_format, False, kernel_name)

        tensor_list = [tensor_in, res]

    # schedule
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    # build
    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              "l1_fusion_option": is_l1fusion}

    tbe.cce_build_code(sch, config)

