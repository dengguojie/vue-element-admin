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
from te import tvm
import te.lang.dynamic
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import check_op_params
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import check_dtype


@te.op.register_operator("Gather")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_BOOL, KERNEL_NAME)
def gather(x, indices, y, validate_indices=True, kernel_name="Gather"):
# @check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
# def gather(x, indices, y, kernel_name="Gather"):
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
