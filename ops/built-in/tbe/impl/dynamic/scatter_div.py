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
scatter_div
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from . import scatter_common


# 'pylint: disable=unused-argument,too-many-arguments
@register_operator("ScatterDiv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def scatter_div(var, indices, updates, var_out, use_locking=False, kernel_name="scatter_div"):
    """
    scatter_div interface

    Parameters
    ----------
    var_dict: input var shape, dtype and range
    indices_dict: input indices shape, dtype and range
    updates_dict: input updates shape, dtype and range
    var_out_dict: output shape, dtype and range
    kernel_name: kernel name of scatter_div op

    Returns
    -------
    compile info
    """
    obj = scatter_common.ScatterCommon(var, indices, updates, var_out, False, kernel_name, "vdiv")
    return obj.scatter_common_operator()
