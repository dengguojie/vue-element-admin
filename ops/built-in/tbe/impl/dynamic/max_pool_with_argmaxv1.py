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
max_pool_with_argmaxv1
"""
from impl.util.platform_adapter import register_operator
from impl.dynamic.max_pool_with_argmaxv2 import MaxPoolWithargmaxPytorch
from impl.dynamic.max_pool_with_argmaxv2 import _check_param


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """

    DT_INT32 = 3
    # dimension n
    DIM_N = 0
    # dimension h
    DIM_H = 1
    # dimension w
    DIM_W = 2
    # dimension c
    DIM_C = 3

    def __init__(self):
        pass


# 'pylint: disable=unused-argument
@register_operator("MaxPoolWithArgmaxV1")
def max_pool_with_argmax_v1(x, y, argmax, ksize, strides, pads, dtype=Constant.DT_INT32, dilation=(1, 1, 1, 1),
                            ceil_mode=False, kernel_name="max_pool_with_argmax_v1"):
    """
    implementation of max_pool_with_argmax for pytorch and return the \
    tik instance
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param dtype: input data type, only support int32 or int64
    :param kernel_name: the kernel's name
    :return: tik_instance
    """
    dim_n = Constant.DIM_N
    dim_h = Constant.DIM_H
    dim_w = Constant.DIM_W
    dim_c = Constant.DIM_C

    ksize = [ksize[dim_n], ksize[dim_c], ksize[dim_h], ksize[dim_w]]
    strides = [strides[dim_n], strides[dim_c], strides[dim_h], strides[dim_w]]
    pads = [pads[dim_n], pads[dim_c], pads[dim_h], pads[dim_w]]
    dilation = [dilation[dim_n], dilation[dim_c], dilation[dim_h], dilation[dim_w]]

    [_, _, dim_h, dim_w] = _check_param(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    obj = MaxPoolWithargmaxPytorch(x.get("shape"), [ksize[dim_h], ksize[dim_w]], [strides[dim_h], strides[dim_w]],
                                   [pads[dim_h], pads[dim_w]], x.get("dtype").lower(),
                                   dilation, ceil_mode, kernel_name)

    return obj.max_pool_operator()
