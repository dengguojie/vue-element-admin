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
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-branches
# pylint: disable=superfluous-parens
def get_op_support_info(x, indices, y, validate_indices=True, batch_dims=0, kernel_name="Gather"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    format_indices = indices.get("format").upper()
    shape_indices_len = len(indices.get("shape"))
    if format_x == "ND" and format_indices == "ND":
        axis_split_matrix=[]
        for j in range(shape_indices_len):
            split_0 = [SplitInput([1, [j], [-1], [-1]]), SplitOutput([0, [j]])]
            axis_split_matrix.append(split_0)
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


@register_operator("Gather")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def gather(x, indices, y, validate_indices=True, batch_dims=0, kernel_name="Gather"):
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
    obj = gather_v2.GatherV2(x, indices, axis_dict, y, batch_dims, kernel_name)
    return obj.gather_compute()
