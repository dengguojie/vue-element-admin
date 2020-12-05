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
concat_offset.py
"""
from impl.util import util_common
from te.utils import para_check


# pylint: disable=unused-argument,invalid-name
def check_supported(concat_dim, x, y, kernel_name="concat_offset"):
    """
    verify the types of cast supported by tbe
    """
    if util_common.is_dynamic_input(x) or util_common.is_dynamic_input([concat_dim]):
        return True

    return False


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_INPUT,
                            para_check.DYNAMIC_OUTPUT, para_check.KERNEL_NAME)
def concat_offset(concat_dim, x, y, kernel_name="concat_offset"):
    """
    Compute the concat offset of the input tensor along `concat_dim`.

    Parameters
    ----------
    concat_dim: dict
                a number of int32, The dimension along which to concatenate,
                must be in the range [-rank(shape), rank(shape))
    x: list of dict, dict include shape and dtype, dtype must be in ('int32')
    y: list of dict, dict include shape and dtype, dtype must be in ('int32')
    kernel_name: kernel name

    Returns
    -------
    None
    """
    pass

