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
dilation compute
"""

from tbe import tvm
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils import para_check


def _calc_dilated_shape(shape, dilations, pads):
    """
    calculate dilated shape
    :param shape: list or tuple
    :param dilations: list or tuple
    :param pads: list or tuple
    return : list, dilated shape
    """
    return list(map(lambda a, b, c: (a - 1)*b + 1 + c[0] + c[1], shape, dilations, pads))


def _param_check(tensor_x, dilations):
    """
    check param
    :param tensor_x: tvm.tensor
    :param dilations: list or tuple
    """
    shape_x = [i.value for i in tensor_x.shape]
    para_check.check_shape(shape_x, param_name="x")
    check_list = ("int8", "float16", "float32")
    para_check.check_dtype(tensor_x.dtype, check_list, param_name="x")
    if not all([value > 0 and isinstance(value, int) for value in dilations]):
        args_dict = {
            "errCode": "E60038",
            "desc": "Elements in dilations should be positive integer"
        }
        raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))


def dilation_compute(tensor_x, dilations, pads=None, padding_value=0.0):
    """
    dilation_compute
    :param tensor_x: tensor
    :param dilations: list or tuple
    :param pads: list or tuple or None
    :param padding_value: float
    """
    _param_check(tensor_x, dilations)
    shape_x = [i.value for i in tensor_x.shape]
    if pads is None:
        pads = [(0, 0)] * len(dilations)
    else:
        pads = [(0, 0), (pads[0], pads[1]), (pads[2], pads[3]), (0, 0)]
    shape_dilated = _calc_dilated_shape(shape_x, dilations, pads)
    dtype_x = tensor_x.dtype
    zero_tensor = tvm.compute(
        shape_dilated,
        lambda *indices: tvm.convert(padding_value).astype(dtype_x),
        name="init_pad",
        tag="init_pad"
    )
    dilate_res = tvm.compute(
        shape_dilated,
        lambda *indices: tvm.select(
            tvm.all(*[((indices[i] - pads[i][0]) % dilations[i] == 0) for i in range(len(indices))]),
            tensor_x(*[(indices[i] - pads[i][0]) // dilations[i] for i in range(len(indices))]),
            zero_tensor(*indices)
        ),
        name="dilation",
        tag="dilation",
        attrs={
            "dilations_para": dilations,
            "pads_para": pads
        }
    )

    return dilate_res
