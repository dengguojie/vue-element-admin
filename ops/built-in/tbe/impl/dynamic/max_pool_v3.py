#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
max_pool_v3
"""

from impl.dynamic.max_pool import MaxPool
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MAX_KERNEL_SIZE_H_MUL_W = 255
    MAX_KERNEL_SIZE = 20


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,too-many-statements
# 'pylint: disable=self-assigning-variable,too-many-branches
def check_window_rule(global_pooling, ksize, strides, padding_mode, data_format, kernel_name):
    """
    check ksize and strides of window in pooling
    """
    if data_format not in ("NC1HWC0", "NCHW", "NHWC"):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "data_format", ["NHWC", "NC1HWC0", "NCHW"],
                                                            data_format)
    dim_n = 0
    dim_c = 1
    dim_h = 2
    dim_w = 3
    if data_format in ("NHWC",):
        dim_c = 3
        dim_h = 1
        dim_w = 2

    if len(ksize) != 4:
        expected_value = "equal to 4"
        real_value = "not equal to 4"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "length of ksize", expected_value, real_value)
    if ksize[dim_n] != 1 or ksize[dim_c] != 1:
        expected_value = "equal to 1"
        real_value = "not equal to 1"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize[0] and ksize[3]", expected_value,
                                                           real_value)

    if ksize[dim_h] < 1:
        expected_value = "greater than zero"
        real_value = ksize[dim_h]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize_h", expected_value, real_value)

    if ksize[dim_w] < 1:
        expected_value = "greater than zero"
        real_value = ksize[dim_w]
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize_w", expected_value, real_value)

    if len(strides) != 4:
        expected_value = "equal to 4"
        real_value = "not equal to 4"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "length of strides", expected_value, real_value)
    if strides[dim_n] != 1 or strides[dim_c] != 1:
        expected_value = "equal to 1"
        real_value = "not equal to 1"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides[0] and strides[3]", expected_value,
                                                           real_value)
    if padding_mode not in ("SAME", "VALID", "CALCULATED"):
        error_manager_vector.raise_err_pad_mode_invalid("max_pool_v3", "SAME, VALID or CALCULATED", str(padding_mode))

    if not global_pooling:
        is_support_kernel = (ksize[dim_h] * ksize[dim_w] <= Constant.MAX_KERNEL_SIZE_H_MUL_W) or \
            (ksize[dim_h] <= Constant.MAX_KERNEL_SIZE and ksize[dim_w] <= Constant.MAX_KERNEL_SIZE)

        if not is_support_kernel:
            expected_value = "(ksize[dim_h]*ksize[dim_w]<=MAX_KERNEL_SIZE_H_MUL_W) or \
                (ksize[dim_h]<=Constant.MAX_KERNEL_SIZE and ksize[dim_w]<=Constant.MAX_KERNEL_SIZE)"

            real_value = "Does not meet the restrictions"
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "ksize_h and ksize_w", expected_value,
                                                               real_value)
        if strides[dim_h] < 1:
            expected_value = "greater than zero"
            real_value = strides[dim_h]
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_h", expected_value, real_value)
        if strides[dim_w] < 1:
            expected_value = "greater than zero"
            real_value = strides[dim_w]
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "strides_w", expected_value, real_value)


def pretreat_to_maxpool(padding_mode, pads, global_pooling, ceil_mode):
    """
    change input and attr to adapt maxpool
    """
    if global_pooling:
        pads = (0, 0, 0, 0)
        global_pooling = 1
        padding_mode = 1
    else:
        global_pooling = 0

    if padding_mode == "CALCULATED":
        padding_mode = 2
        pads = pads
    elif padding_mode == "SAME":
        pads = (0, 0, 0, 0)
        padding_mode = 0
    else:
        pads = (0, 0, 0, 0)
        padding_mode = 1

    if ceil_mode:
        ceil_mode = 1
    else:
        ceil_mode = 0
    return [padding_mode, pads, global_pooling, ceil_mode]


@register_operator("MaxPoolV3")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_LIST_INT, para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
# 'pylint: disable=too-many-locals
def max_pool_v3(input_data,
                output_data,
                ksize,
                strides,
                padding_mode="CALCULATED",
                pads=(0, 0, 0, 0),
                data_format="NHWC",
                global_pooling=False,
                ceil_mode=False,
                kernel_name="max_pool_v3"):
    """
    Performs max pooling on the input.

    Parameters
    ----------
    input_data: dict
        dict of input_data, include keys(shape and dtype).
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    ksize: list or tuple
        A list of `ints` that has length 4.
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding_mode: str
        A `string` from: `"SAME", "VALID", "CALCULATED"`.The type of padding algorithm to use.
    pads: list
        A list of 'ints' that has length 4.
    data_format: str
        A `string` from: `"NC1HWC0", "NHWC", "NCHW"`.
    global_pooling: bool

    kernel_name: str
        kernel name, default value is 'max_pool'

    Returns:
    -------
    None
    """
    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype").lower()

    para_check.check_shape(shape_input, param_name="input_data")
    check_list = ("float16", "int8", "uint8")
    para_check.check_dtype(dtype_input, check_list, param_name="input_data")
    check_window_rule(global_pooling, ksize, strides, padding_mode, data_format, kernel_name)
    padding_mode, pads, global_pooling, ceil_mode = pretreat_to_maxpool(padding_mode, pads, global_pooling, ceil_mode)
    if data_format in ("NHWC",):
        ksize_h, ksize_w = ksize[1], ksize[2]
        strides_h, strides_w = strides[1], strides[2]
    elif data_format in ("NC1HWC0", "NCHW"):
        ksize_h, ksize_w = ksize[2], ksize[3]
        strides_h, strides_w = strides[2], strides[3]

    pad_top = pads[0]
    pad_bottom = pads[1]
    pad_left = pads[2]
    pad_right = pads[3]

    obj = MaxPool(dtype_input, [ksize_h, ksize_w], [strides_h, strides_w], "SAME", kernel_name)
    obj.max_pool_compute_tiling()
    opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}

    tbe_context.get_context().add_compile_info(
        "vars", {
            "ub_ele": obj.ub_ele,
            "core_num": obj.core_num,
            "ksize_h": obj.ksize_h,
            "ksize_w": obj.ksize_w,
            "strides_h": obj.strides_h,
            "strides_w": obj.strides_w,
            "padding": padding_mode,
            "ceil_mode": ceil_mode,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "global": global_pooling
        })

    obj.tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                              inputs=[obj.input_gm],
                              outputs=[obj.output_gm],
                              flowtable=[obj.tiling_gm],
                              config=opt_config)
    return obj.tik_instance
