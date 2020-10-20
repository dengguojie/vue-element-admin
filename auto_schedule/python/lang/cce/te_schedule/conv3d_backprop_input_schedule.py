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
CceConv3dBackpropInputOp
"""
import te.platform as tbe_platform
from te.lang.cce.te_schedule import conv3d_backprop_input_general_schedule


class CceConv3dBackpropInputOp(object):   # pylint: disable=R0903
    """
    The class of conv3d backprop input

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing pragma when using calculate


    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        self._scope = scope
        self._need_tensorize = need_tensorize
        self._need_pragma = need_pragma
        self._res_tensor = None
        self._spec_node_list = None

    def schedule(self, res, spec_node_list, sch_list):
        """
        auto_schedule for cce AI-CORE.

        Parameters
        ----------
        res : tvm.tensor

        spec_node_list : same as other template in cce_schedule

        sch_list: use sch_list[0] to return conv schedule

        Returns
        -------
        True for success, False for no schedule
        """
        self._res_tensor = res
        self._spec_node_list = spec_node_list
        tbe_platform.cce_params.jump_expand_flag = True
        sch = conv3d_backprop_input_general_schedule.general_schedule(res, sch_list)
        return sch
