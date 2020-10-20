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
CCE configuration constants
"""

from __future__ import absolute_import as _abs

from te.tvm.intrin import call_pure_intrin


def cc_device_exp(input_data, p_scale, p_shift, p_base, p_shape):
    """Take exp of input_data by cc_device_api.(DeviceExp)

    Parameters
    ----------
    input_data: Tensor
        Input argument.

    p_scale: ptr of Tensor
        default [1]

    p_shift: ptr of Tensor
        default [0]

    p_base: ptr of Tensor
        default [-1]

    p_shape: ptr of Tensor
        default [1]

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(input_data.dtype, "DeviceExp", input_data,
                            p_scale, p_shift, p_base, p_shape)


def cc_device_log(input_data, p_scale, p_shift, p_base, p_shape):
    """Take log of input_data by cc_device_api(DeviceLog).

    Parameters
    ----------
    input_data: Tensor
        Input argument.

    p_scale: ptr of Tensor
        default [1]

    p_shift: ptr of Tensor
        default [0]

    p_base: ptr of Tensor
        default [-1]

    p_shape: ptr of Tensor
        default [1]

    Returns
    -------
    y : Expr
        The result.
    """
    return call_pure_intrin(input_data.dtype, "DeviceLog", input_data,
                            p_scale, p_shift, p_base, p_shape)
