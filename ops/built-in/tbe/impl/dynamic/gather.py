# Copyright 2019 Huawei Technologies Co., Ltd
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
gather_d
"""
from . import gather_v2
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator


# pylint: disable=invalid-name,unused-argument
@register_operator("Gather")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def gather(x, indices, y, validate_indices=True, kernel_name="Gather"):
    """
    gather interface

    Parameters
    ----------
    x: input params shape, dtype and range
    indices: input indices shape, dtype and range
    y: output shape, dtype and range
    kernel_name: kernel name of gather op

    Returns
    -------
    compile info
    """
    axis_dict = {"dtype": "int32"}
    obj = gather_v2.GatherV2(x, indices, axis_dict, y, kernel_name)
    return obj.gather_compute()
