# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
layer_norm_beta_gamma_backprop_v2 tiling case
"""

from tbe.dsl.base.operation import register_tiling_case
from .constants import Pattern

@register_tiling_case(pattern=Pattern.LAYER_NORM_BETA_GAMMA_BACKPROP_V2)
def calc_layernorm_beta_gamma_v2(outs, option=None):
    """tiling_case func for dynamic shape layernormbetagammabackpropv2

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    return [("no_split_reduce", 0), ("split_reduce", 1), ("split_reduce_i", 2)]

