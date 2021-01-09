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
avg_pool3d_grad_d
"""
from te import tvm
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils.cce import auto_schedule
from te.tvm.target import cce
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from te.utils.error_manager import error_manager_util
from impl.conv3d_backprop_input_d import conv3d_backprop_input_fusion_compute

_BLOCK_SIZE = 16
_C0_SIZE = tbe_platform.C0_SIZE
_UB_FUSED_OP_NUM = 2
_FMAP_TARGET_FORMAT = "NDHWC"
_GRADS_TARGET_FORMAT = "NDHWC"
_GRADS_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]
_STRIDE_SORCE_FORMAT = "NDHWC"
_DATA_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC"]

def _transform_shape_with_format(ori_format, shape):
    idx_d = ori_format.find('D')
    idx_h = ori_format.find('H')
    idx_w = ori_format.find('W')
    shape_all = [1, 1, 1, 1, 1]
    if len(shape) == 1:
        shape_dhw = (shape[0], shape[0], shape[0])
        shape_all[idx_d] = shape[0]
        shape_all[idx_h] = shape[0]
        shape_all[idx_w] = shape[0]
    elif len(shape) == 3:
        shape_dhw = [shape[2], shape[0], shape[1]]
        shape_all[idx_d] = shape[2]
        shape_all[idx_h] = shape[0]
        shape_all[idx_w] = shape[1]
    elif len(shape) == 5:
        shape_dhw = (shape[idx_d], shape[idx_h], shape[idx_w])
        shape_all = shape
    return tuple(shape_all), shape_dhw


def _check_window_rule(ksize, strides, pads):
    if len(ksize) != 3:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad_d',
                                                                 'ksize',
                                                                 3,
                                                                 3,
                                                                 len(ksize))
    if len(strides) != 3:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad_d',
                                                                 'strides',
                                                                 3,
                                                                 3,
                                                                 len(strides))
    if len(pads) != 6:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d_grad_d',
                                                                 'pads',
                                                                 6,
                                                                 6,
                                                                 len(pads))

def _check_ub_limitation(input_shape, strides):
    w_value = input_shape[3] * strides[1]

    aub_size_min = input_shape[3] * _BLOCK_SIZE * 2
    aub_filling_size_min = w_value * _BLOCK_SIZE * 2
    cub_size_min = _BLOCK_SIZE * _BLOCK_SIZE * 2
    ub_size = tbe_platform.get_soc_spec("UB_SIZE")

    if (aub_size_min * _UB_FUSED_OP_NUM + aub_filling_size_min + cub_size_min) > ub_size:
        dict_args = {
            'errCode': 'E60119'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))

#pylint: disable=too-many-arguments,unused-argument,invalid-name
def _avg_pool3d_grad_check_rule(input_shape, input_dtype, ksize, strides, pads, kernel_name):
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ("float16", ))
    para_check.check_kernel_name(kernel_name)
    _check_window_rule(ksize, strides, pads)


def _correct_pads(input_shape, fmap_shape, ksize, strides, pads):
    _, input_d, input_h, input_w, _ = input_shape
    _, fmap_d, fmap_h, fmap_w, _ = fmap_shape
    ksize_h, ksize_w, ksize_d = ksize
    stride_h, stride_w, stride_d = strides
    pad_before, pad_after, pad_top, pad_bottom, pad_left, pad_right = pads

    pad_after = (input_d - 1) * stride_d + ksize_d - fmap_d - pad_before
    pad_bottom = (input_h - 1) * stride_h + ksize_h - fmap_h - pad_top
    pad_right = (input_w - 1) * stride_w + ksize_w - fmap_w - pad_left

    return [pad_before, pad_after, pad_top, pad_bottom, pad_left, pad_right]


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

#pylint: disable=too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME
                            )
def avg_pool3d_grad_d(grads,
                      filter,
                      multiplier,
                      output,
                      orig_input_shape,
                      ksize,
                      strides,
                      pads,
                      ceil_mode=False,
                      count_include_pad=True,
                      divisor_override=0,
                      data_format="NDHWC",
                      kernel_name="avg_pool3d_grad_d"):
    """
    computes average pooling3d backwards gradients.

    Parameters:
    -----------

    grads : dict, shape and dtype of input_data,
            only support float16, shape is 5dims, format is NDC1HWC0

    filter : dict, fractal_z_3d layout, float16 dtype

    multiplier : dict, NDC1HWC0 layout, float16 dtype

    output : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avg_pool3d,
            only support avg_pool3d in D or H or W

    strides:list or tuple, the window of avg_pool3d,
            only support avg_pool3d in D or H or W

    pads : list or tuple, count of padding zero or d, h, w axis

    ceil_mode : when True, will use ceil mode instead of floor
                in the formula to compute the output shape

    count_include_pad : when True, will include the zero-padding
                        in the averaging calculation

    divisor_override : if specified, it will be used as divisor,
                       otherwise size of the pooling region will be used

    data_format : str, default value is "NDHWC"

    kernel_name : cce kernel name, default value is "avg_pool3d_grad_d"

    Returns
    -------
    None
    """

    grads_ori_format = grads.get("ori_format")
    grads_ori_shape = grads.get("ori_shape")
    grads_shape = grads.get("shape")
    grads_dtype = grads.get("dtype").lower()

    _avg_pool3d_grad_check_rule(grads_shape, grads_dtype, ksize, strides, pads, kernel_name)

    strides_formated = _transform_shape_with_format(_STRIDE_SORCE_FORMAT,
                                                    grads_ori_format,
                                                    [1, strides[2], strides[0], strides[1], 1],
                                                    _DATA_FORMAT_WHITE_LIST)

    if strides_formated is None:
        dict_args = {
            'errCode': 'E62002',
            'param_name': 'data_format',
            'expected_format_list': ",".join(_DATA_FORMAT_WHITE_LIST),
            'format': data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    orig_input_shape_formated = _transform_shape_with_format(data_format,
                                                             _FMAP_TARGET_FORMAT,
                                                             orig_input_shape,
                                                             _DATA_FORMAT_WHITE_LIST)
    if orig_input_shape_formated is None:
        dict_args = {
            'errCode': 'E62002',
            'param_name': 'data_format',
            'expected_format_list': ",".join(_DATA_FORMAT_WHITE_LIST),
            'format': data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    grads_ori_shape_formated = _transform_shape_with_format(grads_ori_format,
                                                            _GRADS_TARGET_FORMAT,
                                                            grads_ori_shape,
                                                            _GRADS_FORMAT_WHITE_LIST)
    if grads_ori_shape_formated is None:
        dict_args = {
            'errCode': 'E62002',
            'param_name': 'grads',
            'expected_format_list': ",".join(_GRADS_FORMAT_WHITE_LIST),
            'format': grads_ori_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    _check_ub_limitation(grads_ori_shape_formated, strides)

    if ceil_mode:
        pads = _correct_pads(grads_ori_shape_formated, orig_input_shape_formated,
                             ksize, strides, pads)

    kh, kw, kd = ksize
    on, od, oh, ow, oc = orig_input_shape_formated

    grads = tvm.placeholder(grads_shape,
                            name="grads",
                            dtype=grads_dtype,
                            attrs={"ori_shape": grads_ori_shape,
                                   "ori_format": grads_ori_format,
                                   "data_type": "float16"})

    # global mode
    if (kd >= od + pads[0] + pads[1] and
            kh >= oh + pads[2] + pads[3] and
            kw >= ow + pads[4] + pads[5]):
        if (grads_ori_shape[grads_ori_format.find('D')] != 1 or
                grads_ori_shape[grads_ori_format.find('H')] != 1 or
                grads_ori_shape[grads_ori_format.find('W')] != 1):
            error_detail = "when global mode, " \
                           "the d-axis, h-axis and w-axis of input_grad must be 1."
            error_manager_vector.raise_err_input_shape_invalid(kernel_name,
                                                               "grads_ori_shape",
                                                               error_detail)

        kd = min(kd, od + pads[0] + pads[1])
        kh = min(kh, oh + pads[2] + pads[3])
        kw = min(kw, ow + pads[4] + pads[5])

        if divisor_override:
            kernel_size_reciprocal = 1.0 / divisor_override
        elif count_include_pad:
            kernel_size_reciprocal = 1.0 / (kh * kw * kd)
        else:
            kernel_size_reciprocal = 1.0 / (oh * ow * od)

        with cce():
            grad_tmp = tbe.vmuls(tbe.cast_to(grads, "float32"),
                                 kernel_size_reciprocal)
            if grads_dtype == "float16":
                grad_tmp = tbe.cast_to(grad_tmp, "float16")

            output_shape = (on, od, (oc + _C0_SIZE - 1) // _C0_SIZE, oh, ow, _C0_SIZE)

            res = tbe.broadcast(grad_tmp, output_shape)

            tensor_list = [grads, res]

            sch = tbe.auto_schedule(tensor_list[-1])

        config = {"name": kernel_name,
                  "tensor_list": tensor_list}
        tbe.cce_build_code(sch, config)
        return

    # cube mode
    dilations = (1, 1, 1, 1, 1)
    offset_w = None
    bias = None
    fmap_c = oc
    c1 = grads_shape[2]
    w_ori_shape = (kd, kh, kw, 1, fmap_c)
    filter_frac_z = (c1 * kd * kh * kw, 1, _C0_SIZE, _C0_SIZE)
    filter = tvm.placeholder(filter_frac_z,
                             name="filter",
                             dtype="float16",
                             attrs={"ori_shape": w_ori_shape,
                                    "ori_format": "DHWCN",
                                    "data_type": "float16"})
    if multiplier:
        mul_shape = multiplier.get("shape")
        multiplier = tvm.placeholder(mul_shape,
                                     name="multiplier",
                                     dtype="float16")
        mul_res = tbe.vmul(grads, multiplier)
        mul_res.op.attrs['ori_format'] = grads_ori_format
        mul_res.op.attrs['shape'] = grads_shape
        mul_res.op.attrs['ori_shape'] = grads_ori_shape

        res = conv3d_backprop_input_fusion_compute(filter,
                                                   mul_res,
                                                   output,
                                                   orig_input_shape,
                                                   strides_formated,
                                                   pads,
                                                   dilations,
                                                   groups=fmap_c,
                                                   data_format=data_format,
                                                   kernel_name=kernel_name)
        tensor_list = [grads, filter, multiplier, res]
    else:
        res = conv3d_backprop_input_fusion_compute(filter,
                                                   grads,
                                                   output,
                                                   orig_input_shape,
                                                   strides_formated,
                                                   pads,
                                                   dilations,
                                                   groups=fmap_c,
                                                   data_format=data_format,
                                                   kernel_name=kernel_name)
        tensor_list = [grads, filter, res]

    with cce():
        sch = auto_schedule(tensor_list[-1])

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)

