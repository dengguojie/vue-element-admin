#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
avg_pool
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import op_utils
from te.utils import para_check
from te.utils import shape_util
from te.platform.cce_policy import get_L1_info
from te.utils.error_manager import error_manager_util as err_man


def _get_fusion_params(input_data, output_data, is_fused_compute=True):
    """
    :param input_data: tensor of input_data
    :param output_data: dict of output_data
    :param is_fused_compute: the default value is true.
    :return: dict fusion_params
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


def _avgpool_conv2d_fusion_para(inputs, outputs):
    """
    get L1 fusion para for depthwise_conv2d
    :param inputs: input data
    :param outputs: output data
    :return: l1 convergence parameter
    """

    input_memory_type = inputs.op.attrs["addr_type"] \
        if "addr_type" in inputs.op.attrs else 0
    output_memory_type = outputs["addr_type"] \
        if "addr_type" in outputs else 0
    valid_shape = inputs.op.attrs["valid_shape"] \
        if "valid_shape" in inputs.op.attrs else ()
    slice_offset = inputs.op.attrs["slice_offset"] \
        if "slice_offset" in inputs.op.attrs else ()
    l1_fusion_type = inputs.op.attrs["L1_fusion_type"] \
        if "L1_fusion_type" in inputs.op.attrs else -1

    fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"] \
        if "L1_addr_flag" in inputs.op.attrs else -1
    fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"] \
        if "L1_valid_size" in inputs.op.attrs else -1

    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = shape_util.shape_to_list(valid_shape)
    slice_offset = shape_util.shape_to_list(slice_offset)

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        output_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    # 0 is ddr 1 is l1 2 is l2
    if int(input_memory_type) not in (0, 1, 2):
        err_man.raise_err_input_mem_type("depthwise_conv2d",
                                         input_memory_type)
    if int(output_memory_type) not in (0, 1, 2):
        err_man.raise_err_output_mem_type("depthwise_conv2d",
                                          output_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user(
            "depthwise_conv2d",
            "if valid_shape exists slice_offset can not be []")

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": output_memory_type,
                   "valid_shape": valid_shape,
                   "slice_offset": slice_offset,
                   "l1_fusion_type": l1_fusion_type,
                   "fmap_l1_addr_flag": fmap_l1_addr_flag,
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


## pylint: disable=locally-disabled,too-many-arguments
def _pad_compute(padding, input_h, input_w, stride, window, dilations):
    """
    Calculate the pad value.
    :param padding: str, SAME or VALID
    :param input_h: int, input h
    :param output_w: int, output w
    :param stride: list, stride attr
    :param window: list, window attr
    :param dilations: list, dilations attr
    :return: pad
    """

    if padding == "SAME":
        output_h = (input_h + stride[0] - 1) // stride[0]
        output_w = (input_w + stride[1] - 1) // stride[1]
        pad_row = (output_h - 1) * stride[0] + \
                  ((window[0] - 1) * dilations[0] + 1) - input_h
        pad_col = (output_w - 1) * stride[1] + \
                  ((window[1] - 1) * dilations[1] + 1) - input_w
        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left
        pad_top = _get_pad(int(pad_top))
        pad_bottom = _get_pad(int(pad_bottom))
        pad_left = _get_pad(int(pad_left))
        pad_right = _get_pad(int(pad_right))
        pad = (pad_top, pad_bottom, pad_left, pad_right)
    else:
        pad = (0, 0, 0, 0)
    return pad


# pylint: disable=locally-disabled,too-many-arguments,too-many-statements
# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def _check_window_rule(ksize, strides, data_format):
    """
    check ksize and strides of window in pooling
    :param ksize: list or tuple, the length must be 4
    :param strides: list or tuple, the length must be 4
    :param data_format: input format
    :return: None
    """

    ksize_c = ksize[3] if data_format in ("NHWC",) else ksize[1]
    strides_c = strides[3] if data_format in ("NHWC",) else strides[1]
    if len(ksize) != 4:
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
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

    if len(strides) != 4:
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_012
        error_info['op_name'] = 'avg_pool'
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

    if ksize[0] != 1 or (ksize_c != 1):
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("ksize[1]", "ksize[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(ksize[1]), str(ksize[3])))
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s], "
                           "but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['expected_value'],
                            error_info['real_value']))


    if strides[0] != 1 or strides_c != 1:
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_000
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = ",".join(("strides[1]", "strodes[3]"))
        error_info['expected_value'] = '1'
        error_info['real_value'] = ",".join((str(strides[1]), str(strides[3])))
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                           "but actually is [%s]." % (error_info['op_name'],
                                                      error_info['param_name'],
                                                      error_info['expected_value'],
                                                      error_info['real_value']))

    if data_format not in("NCHW", "NHWC", "NC1HWC0"):
        error_info = {}
        error_info['errCode'] = op_utils.OP_ERROR_CODE_015
        error_info['op_name'] = 'avg_pool'
        error_info['param_name'] = 'x'
        error_info['excepted_format_list'] = ",".join(("NC1HWC0",
                                                       "NCHW", "NHWC"))
        error_info['format'] = data_format
        raise RuntimeError(error_info, "In op[%s], the format[%s] of input"
                                       "should be one of [%s],"
                                       "but actually is [%s]."
                           % (error_info['op_name'], error_info['param_name'],
                              error_info['excepted_format_list'],
                              error_info['format']))


def _get_pad(input_pad):
    """
    algorithm:
    obtains the updated pad value.

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


def _avg_pool_check_rule(input_shape, input_dtype,
                         output_dtype, ksize, strides,
                         data_format, kernel_name):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param data_format: NHWC default
    :param kernel_name: cce kernel name
    :return: None
    """

    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ["float16", "int8"])
    para_check.check_dtype(output_dtype, ["float16", "int8", "int32"])

    _check_window_rule(ksize, strides, data_format)


def _avg_pool_global_compute(x, y, ksize, strides,
                             padding="VALID", data_format="NHWC",
                             is_fused_compute=True,
                             kernel_name="avg_pool"):
    """
    algorithm: avg_pool
    calculating the average pooling

    Parameters
    ----------
    x : placeholders, shape and dtype of input_data, only support float16
    y : dict, shape and dtype of output data, only support float16
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W
    padding : str, the mode of padding, support padding and not padding
    data_format : str, default = "NHWC"
    is_fused_compute : bool, default true
    kernel_name : kernel name, default value is "avg_pool"

    Returns
    -------
    Calculation result
    """

    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

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
        res = tbe.pooling2d(select_tensor_in, window, stride, "AVG",
                            padding, fusion_params=fusion_params)
    elif l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0:
                                         x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = tbe.pooling2d(l1_width_fusion_in, window, stride,
                            "AVG", padding,
                            fusion_params=fusion_params)
    else:
        res = tbe.pooling2d(x, window, stride, "AVG", padding,
                            fusion_params=fusion_params)

    return res


def _tensor_list_compute(tensor_in, filter, bias, y, ksize, strides, padding, data_format,
                         offset_x, attr, input_shape, kernel_name):
    """
    Parameters
    ----------
    tensor_in : placeholder, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    attr : dict, l1 fusion parameter

    input_shape : list or tuple, input shape

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    res and placeholder of tensor list
    """

    # filter is not None, deepwise branch
    if filter is not None:
        out_dtype = y.get("dtype")
        shape_x = input_shape
        input_h = shape_x[2]
        input_w = shape_x[3]
        dilations = (1, 1)
        dsl_flag = False
        if data_format in ("NHWC",):
            ksize_h = ksize[1]
            ksize_w = ksize[2]
            hw = ksize_h * ksize_w
            window = [ksize[1], ksize[2]]
            stride = [strides[1], strides[2]]
        else:
            ksize_h = ksize[2]
            ksize_w = ksize[3]
            hw = ksize_h * ksize_w
            window = [ksize[2], ksize[3]]
            stride = [strides[2], strides[3]]
        pad = _pad_compute(padding, input_h, input_w, stride, window, dilations)
        filter_shape = filter.get("shape")
        filter_dtype = filter.get("dtype").lower()
        filter_c1 = filter_shape[0] / hw
        if filter_dtype in("float16", "float32"):
            filter_shape_5d = filter_c1, ksize_h, ksize_w, filter_shape[2], \
                              filter_shape[3]
        else:
            # filter shape: C1HWNCoC0,N=1,Co=32,c0=32
            filter_shape_5d = filter_shape[0], ksize_h, ksize_w, 32, \
                              32
        filter_in = tvm.placeholder(filter_shape_5d, name="filter_in",
                                    dtype=filter_dtype, attrs=attr)
        bias_tensor = None
        # bias is not None quantitative scenario
        if bias is not None:
            bias_shape = bias.get("shape")
            bias_tensor = tvm.placeholder(bias_shape,
                                          name='bias_tensor',
                                          dtype=out_dtype.lower())


        res = tbe.te_compute.depthwise_conv2d_compute(
            tensor_in, filter_in, out_dtype.lower(), stride, pad, dilations, {
                "bias_tensor": bias_tensor, "dsl_flag": dsl_flag,
                "offset_x": offset_x}, None, kernel_name)


        tensor_list = [tensor_in, filter_in, res]
        if bias_tensor is not None:
            tensor_list = [tensor_in, filter_in, bias_tensor, res]

    # filter is None, pooling branch
    else:
        res = _avg_pool_global_compute(tensor_in, y, ksize, strides, padding,
                                       data_format, False, kernel_name)
        tensor_list = [tensor_in, res]

    return res, tensor_list


# pylint: disable=unnecessary-lambda,redefined-builtin,too-many-locals
# pylint: disable=unnecessary-lambda,too-many-statements
@tbe_platform.fusion_manager.fusion_manager.register("avg_pool")
def avg_pool_compute(x, filter, bias, y, ksize, strides, padding="VALID",
                     data_format="NHWC", offset_x=0, kernel_name="avg_pool"):
    """
    algorithm: avg_pool
    calculating the average pooling

    Parameters
    ----------
    x : placeholder, shape and dtype of input_data, only support float16 or int8
    filter : optional input, only support float16 or int8
    bias : optional input, only support int32
    y : dict, shape and dtype of output_data, only support float16
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W
    padding : str, the mode of padding, support padding and not padding
    data_format : str, default = "NHWC"
    offset_x : quantization parameter
    kernel_name : kernel name, default value is "avg_pool"

    Returns
    -------
    Calculation result
    """
    out_dtype = y.get("dtype")
    # create window and stride for pooling2d
    # check  parameter
    _check_window_rule(ksize, strides, data_format)

    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    shape_x = x.shape
    input_h = shape_x[2]
    input_w = shape_x[3]
    dilations = (1, 1)
    dsl_flag = True
    pad = _pad_compute(padding, input_h, input_w, stride, window, dilations)

    if int(input_h) == int(window[0]) and int(input_h) == int(window[1]):
        res = avg_pool_global_compute(x, y, ksize, strides, padding, data_format,
                                      is_fused_compute=True, kernel_name=kernel_name)
    else:
        l1_fusion_para = _avgpool_conv2d_fusion_para(x, y)
        res = tbe.te_compute.depthwise_conv2d_compute(x, filter, out_dtype.lower(), stride, pad, dilations,
                                                      {"bias_tensor": bias, "dsl_flag": dsl_flag,
                                                       "offset_x": offset_x}, l1_fusion_para, kernel_name)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def avg_pool(x, filter, bias, y, ksize, strides,
             padding="VALID", data_format="NHWC", offset_x=0,
             kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16 or int8

    filter : dict, optional input, shape and dtype of input_data, only support float16 or int8

    bias : dict, optional input, shape and dtype of input_data, only support int32

    y : dict, shape and dtype of output_data, only support float16 or int32

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    offset_x : int, quantization parameter

    kernel_name : cce kernel name, default value is "avg_pool_cce"

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

    # check others parameter
    _avg_pool_check_rule(input_shape, input_dtype, output_dtype, ksize, strides,
                         data_format, kernel_name)

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
    # create tensor_in
    tensor_in = tvm.placeholder(input_shape, name="tensor_in",
                                dtype=input_dtype, attrs=attr)
    res, tensor_list = _tensor_list_compute(tensor_in, filter, bias, y, ksize, strides, padding, data_format,
                                            offset_x, attr, input_shape, kernel_name)

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
