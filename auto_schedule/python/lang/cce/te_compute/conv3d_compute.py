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
conv3d compute
"""
import warnings


class Conv3DParam(object):
    """
    ConvParam
    """
    def __init__(self):
        pass

    def _get_tensor_map(self):
        """
         get the tensor_map in convparam
        """
        return self._TENSOR_MAP

    _TENSOR_MAP = {}
    dim_map = {}
    tiling = None
    tiling_query_param = {}
    var_map = {}
    dynamic_mode = None
    tiling_info_dict = {}


def conv3d(x, filter, filter_size, para_dict):
    """
    conv

    Parameters
    ----------
    x: feature map

    weight: filter

    filter_size : filter_size

    para_dict: dict of params

    Returns
    -------
    tensor : res
    """
    warnings.warn("te.lang.cce.te_compute.conv3d_compute is expired, " \
        "please replace it with the func tbe.dsl.compute.conv3d_compute",
        DeprecationWarning)
    from tbe.dsl.compute.conv3d_compute import conv3d
    return conv3d(x, filter, filter_size, para_dict)