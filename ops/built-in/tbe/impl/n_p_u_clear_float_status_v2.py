# Copyright 2021 Huawei Technologies Co., Ltd
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
n_p_u_clear_float_status_v2
"""
from te.utils import para_check
from te import tik


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    NUM_EIGHT = 8


# 'pylint:disable=invalid-name,too-many-locals,unused-argument,unused-variable
@para_check.check_op_params(para_check.KERNEL_NAME)
def n_p_u_clear_float_status_v2(kernel_name="n_p_u_clear_float_status_v2"):
    """
    the main function of npu_clear_float_status

    Parameters
    ----------
    kernel_name: cce kernel name, default value is "n_p_u_clear_float_status_v2"

    Returns
    -------
    tik_instance: tik_instance
    """
    tik_instance = tik.Tik()

    spec_workspace = tik_instance.Tensor("float32", (Constant.NUM_EIGHT,),
                                         name="spec_workspace", scope=tik.scope_gm, is_global_tensor=True)

    ub_tensor = tik_instance.Tensor("float32", (Constant.NUM_EIGHT,), name="ub_tensor", scope=tik.scope_ubuf)

    tik_instance.vec_dup(8, ub_tensor, 0, 1, 8)

    tik_instance.data_move(spec_workspace, ub_tensor, 0, 1, 1, 8, 8)

    tik_instance.BuildCCE(kernel_name, inputs=[], outputs=[])
    return tik_instance
