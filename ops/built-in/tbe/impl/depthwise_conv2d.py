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
depthwise_conv2d
"""
import json
import te.platform as tbe_platform
from te.utils import para_check
from te.utils.error_manager import error_manager_cube
from te.utils.error_manager import error_manager_util
from te.utils import cce
from te.lang.cce.te_compute.depthwise_conv2d_compute import depthwise_conv2d_compute
from te import tvm

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 5

# shape's dim of filter must be 4
FILTER_DIM = 6

# shape's dim of strides must be 2
STRIDES_DIM = 4


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    if isinstance(shape, (list, tuple)):
        return shape
    tmp = []
    if shape == "":
        return ()
    for i in shape:
        tmp.append(i.value)
    return tmp


def _depthwise_conv2d_fusion_para(inputs, outputs):
    """
    get L1 fusion para for depthwise_conv2d
    """
    input_memory_type = inputs.op.attrs["addr_type"] if "addr_type" in inputs.op.attrs else 0
    output_memory_type = outputs["addr_type"] if "addr_type" in outputs else 0
    valid_shape = inputs.op.attrs["valid_shape"] if "valid_shape" in inputs.op.attrs else ()
    slice_offset = inputs.op.attrs["slice_offset"] if "slice_offset" in inputs.op.attrs else ()
    l1_fusion_type = inputs.op.attrs["L1_fusion_type"] if "L1_fusion_type" in inputs.op.attrs else -1

    fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"] if "L1_addr_flag" in inputs.op.attrs else -1
    fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"] if "L1_valid_size" in inputs.op.attrs else -1

    l1_fusion_enable_flag = tbe_platform.get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = shape_to_list(valid_shape)
    slice_offset = shape_to_list(slice_offset)

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        output_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    if int(input_memory_type) not in (0, 1, 2):
        error_manager_cube.raise_err_input_mem_type("depthwise_conv2d", input_memory_type)
    if int(output_memory_type) not in (0, 1, 2):
        error_manager_cube.raise_err_output_mem_type("depthwise_conv2d", output_memory_type)
    if valid_shape and not slice_offset:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d",
                                                   "if valid_shape exists slice_offset can not be []")

    fusion_para = {
        "input_memory_type": input_memory_type,
        "output_memory_type": output_memory_type,
        "valid_shape": valid_shape,
        "slice_offset": slice_offset,
        "l1_fusion_type": l1_fusion_type,
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size
    }

    return fusion_para


# pylint: disable=locally-disabled, too-many-locals, too-many-arguments,
# pylint: disable=unused-argument
# pylint: disable=redefined-builtin
@tbe_platform.fusion_manager.fusion_manager.register("depthwise_conv2d")
def depthwise_compute(fmap,
                      filter,
                      bias,
                      offset_w,
                      out,
                      strides,
                      dilations,
                      pads,
                      data_format='NHWC',
                      offset_x=0,
                      kernel_name="depthwise_conv2d"):
    """
    algorithm: depthwise conv2d compute
    calculating  depthwise compute
    Parameters
    ----------
    fmap : a tensor of featureMap
    filter : a tensor of filter
    bias : a tensor of bias
    offset_w : a tensor of filter offset
    out : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.
    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]
    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]
    pads : padding added to each dimension of the input
    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]
    offset_x : offset of the input
    Returns
    -------
    None
    """
    out_dtype = out.get("dtype")
    l1_fusion_para = _depthwise_conv2d_fusion_para(fmap, out)
    dim_h, dim_w = 2, 3
    if data_format == 'NHWC':
        dim_h, dim_w = 1, 2

    strides_2d = strides[dim_h], strides[dim_w]
    dilations_2d = dilations[dim_h], dilations[dim_w]

    out = depthwise_conv2d_compute(fmap, filter, out_dtype.lower(), strides_2d, pads, dilations_2d, {
        "bias_tensor": bias,
        "dsl_flag": True,
        "offset_x": offset_x
    }, l1_fusion_para, kernel_name)
    return out


def _check_shape(fmap_shape, filter_shape, fmap_data_format):
    """check input shape"""
    _, in_c1, _, _, _ = fmap_shape
    filter_c1, _, _, filter_k, _, _ = filter_shape

    # check feature map API feature map  shape is 5hd
    # The shape of feature map and filter must be 5HD
    if len(fmap_shape) != FEATURE_MAP_DIM:
        dict_args = {
            'errCode': 'E60008',
            'op_name': 'depthwise_conv2d',
            'param_name': 'featuremap',
            'expected_format_list': '[{}]'.format('NC1HWC0'),
            'format': fmap_data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    # check feature map shape of c, equal filter of c
    if in_c1 != filter_c1:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d',
            'attr_name': 'channel',
            'param1_name': 'fmap',
            'param2_name': 'filter',
            'param1_value': str(in_c1),
            'param2_value': str(filter_c1)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    # check multiplier equal 1
    if filter_k != 1:
        dict_args = {
            'errCode': 'E60000',
            'op_name': 'depthwise_conv2d',
            'param_name': 'filter_k',
            'expected_value': '1',
            'input_value': str(filter_k)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_data_format(data_format, expect_format_list):
    """
    check data format
    """
    if data_format not in expect_format_list:
        dict_args = {
            'errCode': 'E60031',
            'op_name': 'depthwise_conv2d',
            'param_name': 'featuremap',
            'expected_format_list': str(expect_format_list),
            'format': data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_stride(strides, dim_n, dim_c, dim_h, dim_w):
    """
    check stride type and dim
    """
    if not isinstance(strides, (list, tuple)) and len(strides) == 4:
        dict_args = {'errCode': 'E60107', 'op_name': 'depthwise_conv2d', 'param_name': 'strides'}
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if strides[dim_n] != 1 or strides[dim_c] != 1:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d", "stride only support 1 in N axis and C axis.")


def _check_dilations(dilations, dim_n, dim_c, dim_h, dim_w):
    """
    check dilations dimension
    """
    if dilations[dim_n] != 1 or dilations[dim_c] != 1:
        dict_args = {
            'errCode': 'E60023',
            'op_name': 'depthwise_conv2d',
            'dilation_n': str(dilations[dim_n]),
            'dilation_c': str(dilations[dim_c])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


# pylint: disable=locally-disabled, too-many-locals, too-many-arguments,
# pylint: disable=unused-argument
# pylint: disable=redefined-builtin, invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def depthwise_conv2d(
        x,
        filter,
        bias,
        offset_w,
        y,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        offset_x=0,
        kernel_name="depthwise_conv2d",
):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype", "format"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    filter : a dict of filter
        {"shape", "dtype"}
        shape of filter tensor [C1, H, W, K, Co, C0],
        K is depthwise_multiplier, support int.

    bias : a dict of bias
        {"shape", "dtype"}
        shape of bias tensor [C1*C0,]
        support int8.

    offset_w : a dict of filter offset
        {"shape", "dtype"}
        shape of offset tensor [C1, H, W, K, Co, C0]
        support float16.

    y : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]

    pads : padding added to each dimension of the input

    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    offset_x : offset of the input

    kernel_name : str
       cce kernel name

    Returns
    -------
    None

    """
    shape_w = filter.get("shape")
    shape_in = x.get("shape")
    output_dtype = y.get("dtype")
    in_dtype = x.get("dtype")
    w_dtype = filter.get("dtype")
    fmap_data_format = x.get("format")

    para_check.check_dtype(in_dtype, ('float16', 'int8'), param_name="x")
    para_check.check_dtype(w_dtype, ('float16', 'int8'), param_name="filter")
    para_check.check_dtype(output_dtype, ('float16', 'int32'), param_name="y")

    para_check.check_shape(shape_in, min_rank=FEATURE_MAP_DIM, max_rank=FEATURE_MAP_DIM, param_name="x")
    para_check.check_shape(shape_w, min_rank=FILTER_DIM, max_rank=FILTER_DIM, param_name="filter")
    para_check.check_shape(strides, min_rank=STRIDES_DIM, max_rank=STRIDES_DIM, param_name="filter")

    _check_data_format(fmap_data_format, ["NC1HWC0"])

    # fmap shape reshape, c ceil 16, 6d shape;
    # c must be 16x, if data not 16x, framework reshape c 16x
    in_n, in_c1, in_h, in_w, in_c0 = shape_in
    fmap_shape_5d = in_n, in_c1, in_h, in_w, in_c0
    shape_w_5d = shape_w[0], shape_w[1], shape_w[2], shape_w[4], shape_w[5]

    # filter shape: C1HWNCoC0
    filter_c1, _, _, _, _, _ = shape_w
    _check_data_format(data_format, ['NCHW', 'NHWC'])

    _check_shape(shape_in, shape_w, fmap_data_format)

    dim_n, dim_c, dim_h, dim_w = 0, 1, 2, 3  # NCHW
    if data_format == 'NHWC':
        dim_n, dim_h, dim_w, dim_c = 0, 1, 2, 3

    # check strides is list, strides[0] ==shape_in[1]
    # strides list, and h w value equal
    _check_stride(strides, dim_n, dim_c, dim_h, dim_w)
    _check_dilations(dilations, dim_n, dim_c, dim_h, dim_w)

    # check pad parameter
    if len(pads) != 4:
        dict_args = {
            'errCode': 'E60030',
            'param_name': 'pads',
            'op_name': 'depthwise_conv2d',
            'expected_length': "4",
            'length': str(len(pads))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    strides_2d = strides[dim_h], strides[dim_w]
    dilations_2d = dilations[dim_h], dilations[dim_w]

    c0_val = 16
    if in_dtype == "int8":
        c0_val = 32

    bias_tensor = None
    if bias:
        bias_tensor = tvm.placeholder((filter_c1 * c0_val, ), name='bias_tensor', dtype=output_dtype.lower())
    fmap_placeholder = tvm.placeholder(fmap_shape_5d, dtype=in_dtype.lower(), name='fmap')
    filter_placeholder = tvm.placeholder(shape_w_5d, dtype=w_dtype.lower(), name='filter')
    out = depthwise_conv2d_compute(fmap_placeholder, filter_placeholder, output_dtype.lower(), strides_2d, pads,
                                   dilations_2d, {
                                       "bias_tensor": bias_tensor,
                                       "dsl_flag": False,
                                       "offset_x": offset_x
                                   }, None, kernel_name)

    tensor_list = [fmap_placeholder, filter_placeholder, out]
    if bias_tensor is not None:
        tensor_list = [fmap_placeholder, filter_placeholder, bias_tensor, out]

    with tvm.target.cce():
        sch = cce.auto_schedule(out)

    with tbe_platform.build_config:
        tvm.build_module.build(sch, tensor_list, "cce", name=kernel_name)


def get_op_support_info(x,
                        weights,
                        bias,
                        offset_w,
                        outputs,
                        strides,
                        pads,
                        dilations,
                        groups=1,
                        data_format='NCHW',
                        offset_x=0,
                        kernel_name="depthwiseconv2d"):
    """
    algorithm: get_op_support_info

    Notice
    ------
    get the depthwiseconv2d split

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype", "format"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    filter : a dict of filter
        {"shape", "dtype"}
        shape of filter tensor [C1, H, W, K, Co, C0],
        K is depthwise_multiplier, support int.

    bias : a dict of bias
        {"shape", "dtype"}
        shape of bias tensor [C1*C0,]
        support int8.

    offset_w : a dict of filter offset
        {"shape", "dtype"}
        shape of offset tensor [C1, H, W, K, Co, C0]
        support float16.

    y : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]
    kernel_name: str
        kernel name, default value is "depthwiseconv2d"

    Returns
    -------
    None
    """
    slice_info = {
        "_op_slice_info": {
            "splitMaps": [{
                "inputList": [{
                    "idx": 0,
                    "axis": [0],
                    "headOverLap": [0],
                    "tailOverLap": [0]
                }],
                "outputList": [{
                    "idx": 0,
                    "axis": [0]
                }]
            }, {
                "inputList": [{
                    "idx": 0,
                    "axis": [2],
                    "headOverLap": [0],
                    "tailOverLap": [0]
                }],
                "outputList": [{
                    "idx": 0,
                    "axis": [2]
                }]
            }, {
                "inputList": [{
                    "idx": 0,
                    "axis": [3],
                    "headOverLap": [0],
                    "tailOverLap": [0]
                }],
                "outputList": [{
                    "idx": 0,
                    "axis": [3]
                }]
            }, {
                "inputList": [{
                    "idx": 1,
                    "axis": [1],
                    "headOverLap": [0],
                    "tailOverLap": [0]
                }],
                "outputList": [{
                    "idx": 0,
                    "axis": [1]
                }]
            }],
            "reduceMaps": [],
            "l1FusionEnable":
            2,
            "minTbeL1Space":
            0
        }
    }
    if bias:
        bias_input = [{"idx": 2, "axis": [0], "headOverLap": [0], "tailOverLap": [0]}]
        slice_info['_op_slice_info']["splitMaps"][3]["inputList"].extend(bias_input)

    return json.dumps(slice_info)
