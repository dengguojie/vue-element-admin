#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
Tool functions of conv2d schedule.
"""
import tbe
from tbe import tvm
from tbe.common.utils.errormgr import error_manager_cube as err_man


INTRINSIC_FIXPIPE_UNIT_LIST = "Intrinsic_fix_pipe_unit_list"
UNIT_POST_ELTWISE = "post_eltwise"


def ceil_div(num_a, num_b):
    """
    Do upper division.
    """
    if num_b == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (num_a + num_b - 1) // num_b


def ceil(num_a, num_b):
    """
    Do upper align.
    """
    if num_b == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (num_a + num_b - 1) // num_b*num_b


def get_src_tensor(tensor):
    """
    Get the source tensor of input tensor.
    """
    src_tensor = tensor.op.input_tensors[0]
    return src_tensor


def is_placeholder(tensor):
    """
    Check whether the input tensor is a placeholder.
    """
    if tensor.op.input_tensors:
        return False
    return True


def is_elewise(tensor):
    """
    Check whether the input tensor is a eltwise op.
    """
    if tensor.op.tag.startswith("elewise_"):
        return True
    return False


def is_shape_equal(tensor_a, tensor_b):
    """
    Compare the shape of two input tensors.
    """
    shape_a = tuple(i.value for i in tensor_a.shape)
    shape_b = tuple(i.value for i in tensor_b.shape)

    return shape_a == shape_b


def is_support_fixpipe_op():
    """
    Check v300 intrinsic support.
    """
    if tbe.common.platform.platform_info.intrinsic_check_support(INTRINSIC_FIXPIPE_UNIT_LIST):
        return tbe.common.platform.platform_info.intrinsic_check_support(
            INTRINSIC_FIXPIPE_UNIT_LIST, UNIT_POST_ELTWISE)

    return False
