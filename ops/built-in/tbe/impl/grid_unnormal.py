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
grid_unnormal
"""

from te import tvm
import te.lang.cce as tbe
from te.platform.fusion_manager import fusion_manager
from tbe.common.utils.para_check import REQUIRED_INPUT
from tbe.common.utils.para_check import OPTION_INPUT
from tbe.common.utils.para_check import REQUIRED_OUTPUT
from tbe.common.utils.para_check import OPTION_ATTR_BOOL
from tbe.common.utils.para_check import KERNEL_NAME
from tbe.common.utils.para_check import check_op_params


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("grid_unnormal")
def grid_unnormal_compute(grid, assist, diff, position, align_corners=False, kernel_name="grid_unnormal"):
    """
    algorithm: unnormal grid data
    Parameters
    ----------
    grid : TVM tensor
        the placeholder of grid
    assist : TVM tensor
        the placeholder of assist
    diff: dict
        shape and dtype of output, only support float16, float32
    position: dict
        shape and dtype of output, only support int32
    align_corners : bool.
        An optional bool. If "true", the centers of the corner pixels of
        the input and output tensors are aligned. Defaults to "false" .
    kernel_name : str
        cce kernel name, default value is grid_unnormal

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    grid_tmp1 = tbe.vadds(grid, 1)
    grid_tmp2 = tbe.vmuls(grid_tmp1, 0.5)

    if align_corners:
        input_size = tbe.vadds(assist, -1)
        pos_base = tbe.vmul(grid_tmp2, input_size)
    else:
        tmp1 = tbe.vmul(grid_tmp2, assist)
        pos_base = tbe.vadds(tmp1, -0.5)

    res_pos = tbe.floor(pos_base)
    res_diff = tbe.vsub(pos_base, res_pos)
    return [res_diff, res_pos]


@check_op_params(REQUIRED_INPUT, OPTION_INPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT, OPTION_ATTR_BOOL, KERNEL_NAME)
def grid_unnormal(grid, assist, diff, position, align_corners=False, kernel_name="grid_unnormal"):
    """
    algorithm: unnormal grid data
    Parameters
    ----------
    grid : dict
        shape and dtype of first input, only support float16, float32
    assist : dict
        shape and dtype of second input, only support float16, float32
    diff: dict
        shape and dtype of output, only support float16, float32
    position: dict
        shape and dtype of output, only support int32
    align_corners : bool.
        An optional bool. If "true", the centers of the corner pixels of
        the input and output tensors are aligned. Defaults to "false" .
    kernel_name : str
        cce kernel name, default value is grid_unnormal

    Returns
    -------
    None
    """
    data_grid = tvm.placeholder(grid.get("shape"), dtype=grid.get("dtype"), name="data_grid")
    data_assist = tvm.placeholder(assist.get("shape"), dtype=assist.get("dtype"), name="data_assist")

    res_list = grid_unnormal_compute(data_grid, data_assist, diff, position, align_corners, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res_list)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_grid, data_assist, res_list[0], res_list[1]]}
    tbe.cce_build_code(schedule, config)
