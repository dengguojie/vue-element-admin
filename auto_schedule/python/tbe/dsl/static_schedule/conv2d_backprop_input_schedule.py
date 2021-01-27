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
CceConv2dBackpropInputOp
"""
from te.lang.cce.te_compute.conv2d_backprop_input_compute import DeconvParam
from tbe.dsl.static_schedule.conv2d_backprop_input_general_schedule import general_schedule
from tbe.dsl.static_schedule.conv2d_backprop_input_opti_schedule import opti_schedule
from te.platform import cce_conf
from te.platform import cce_params
from te.utils.error_manager import error_manager_util


class CceConv2dBackpropInputOp:  # pylint: disable=R0903
    """
    The class of conv2d backprop input

    Parameters
    ----------
    scope : cal buffer like local.UB

    need_tensorize : if need to doing tensorize when using calculate

    need_pragma : if need to doing pragma when using calculate

    Returns
    -------
    CceConv2dBackpropInputOp_instance : instance of CceConv2dBackpropInputOp
    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        self._scope = scope
        self._need_tensorize = need_tensorize
        self._need_pragma = need_pragma
        self._res_tensor = None
        self._spec_node_list = None

    def schedule(  # pylint: disable=R0913
        self,
        res,
        spec_node_list,
        sch_list,
        tiling_case=None,
        var_range=None
    ):
        """
        auto_schedule for cce AI-CORE.

        Parameters
        ----------
        res : tvm.tensor

        spec_node_list : same as other template in cce_schedule

        sch_list: use sch_list[0] to return conv schedule

        tiling_case: fix tiling for dynamic shape

        var_range: var range for dynamic shape

        Returns
        -------
        True for sucess, False for no schedule
        """
        self._res_tensor = res
        self._spec_node_list = spec_node_list

        cce_params.jump_expand_flag = True

        def _check_l1_buffer():
            al1_size = DeconvParam.al1_size
            bl1_size = DeconvParam.bl1_size
            if res.dtype == "int8":
                bl1_size *= 2
                if al1_size + bl1_size > cce_conf.get_soc_spec("L1_SIZE"):
                    dict_args = dict()
                    dict_args["errCode"] = "E60026"
                    raise RuntimeError(
                        dict_args, error_manager_util.get_error_message(dict_args)
                    )

        _check_l1_buffer()
        schedule = sch_list[0]
        is_general = False
        for stage in schedule.stages:
            operation = stage.op
            # special case: when kernel_h and kernel_w are equal to 1,
            # and stride is greater than 1,
            # take the general scheme, add bias to l0c
            if operation.tag == "conv2d_backprop_input":
                is_general = True
                break
        if is_general:
            sch = general_schedule(res, sch_list, tiling_case, var_range)
        else:
            sch = opti_schedule(res, sch_list, tiling_case, var_range)
        return sch
