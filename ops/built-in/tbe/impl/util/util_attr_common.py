# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
util_attr_constant
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm


# 'pylint: disable=too-few-public-methods
class AttrBase:
    """
    AttrBase, define the attr base info
    """
    def __init__(self, attr_idx, attr_name, attr_type, attr_default_value=None):
        self.attr_idx = attr_idx
        self.attr_name = attr_name
        self.attr_type = attr_type
        self.attr_value = attr_default_value


def get_attr_by_cls(attr_value, attr_cls, target_dtype):
    """
    get the attr

    Parameters
    ----------
    attr_value: value of attr or None
    attr_cls: AttrBase
    target_dtype: the dtype used for calculation in tvm

    Returns
    -------
    attr_var
    """
    if attr_value is None:
        attr_sting_lower = attr_cls.attr_type.lower()
        attr_dtype = {"src_dtype": attr_sting_lower}
        attr_var = tbe.var_attr(attr_cls.attr_name, dtype=target_dtype, addition=attr_dtype)
    else:
        attr_var = tvm.const(attr_value, target_dtype)
    return attr_var


# begin to define the attr for op
class LayerNormAttrInfo:
    """
    define LayerNorm attr info
    """
    ATTR_NORM_AXIS = AttrBase(0, "begin_norm_axis", "Int", 0)
    ATTR_PARAMS_AXIS = AttrBase(1, "begin_params_axis", "Int", 0)
    ATTR_EPSILON = AttrBase(2, "epsilon", "Float", 0.0000001)


class SoftmaxV2AttrInfo:
    """
    define SoftmaxV2 attr info
    """
    ATTR_AXES = AttrBase(0, "axes", "ListInt", [-1])
