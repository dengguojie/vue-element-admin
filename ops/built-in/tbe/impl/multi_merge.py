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
multi_merge
"""
from te.utils import para_check
from te.utils.error_manager import error_manager_vector

from impl.ascend import AContainer
from impl.merge_sort import CommonMethod
from impl.merge_sort import MergeSort


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class MultiMerge:
    """
    MultiMerge
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, input_shape, k_num, data_type, kernel_name, cont):
        self.sorted_num = input_shape[1]
        self.data_type = data_type
        self.kernel_name = kernel_name

        self.cont = cont
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ub_size = self.cont.const_ub_max_byte
        self.pro_data_num = self.cont.const_proposal_data_num
        self.pro_repeat_num = self.cont.const_proposal_repeat_num

        self.method = CommonMethod(self.cont)
        self.data_size, self.block_data_num, self.repeat_data_num = self.method.get_type_const(self.data_type)
        self.merge_sort = MergeSort(self.cont, self.data_type, self.ub_size)

        self.tail_proposal_num = self.pro_repeat_num
        self.merge_channel = self.merge_sort.merge_channel_num
        self.fp16_ne_inf = -65504.0

        self.result_shape, self.ai_core_use, self.channel_num, self.k_num = self._get_result_info(input_shape, k_num)

        input_proposal_shape = (self.channel_num, input_shape[1] * self.merge_channel, self.pro_data_num)
        self.input_proposal = self.tik_inst.Tensor(self.data_type, input_proposal_shape,
                                                   self.tik.scope_gm, "input_proposal")
        self.output_proposal = self.tik_inst.Tensor(self.data_type, self.result_shape,
                                                    self.tik.scope_gm, "output_proposal")

    def _get_result_info(self, input_shape, k_num):
        sorted_num_align = self.method.get_align_num(k_num, self.pro_repeat_num)
        k_num_new = (input_shape[1] - self.tail_proposal_num) * self.merge_channel
        k_num_new = min(sorted_num_align, k_num_new)
        result_data_num = k_num_new + self.tail_proposal_num

        channel_num = input_shape[0] // self.merge_channel
        if channel_num <= self.merge_channel:
            ai_core_use_num = channel_num
        else:
            ai_core_use_num = self.method.get_align_num(channel_num, self.merge_channel)

        result_shape = (ai_core_use_num, result_data_num, self.pro_data_num)
        return result_shape, ai_core_use_num, channel_num, k_num_new

    def mode_compute(self):
        """
        compute function
        """
        with self.tik_inst.for_range(0, self.ai_core_use, block_num=self.ai_core_use) as core_index:
            with self.tik_inst.if_scope(core_index < self.channel_num):
                self._mode_compute_each_core(core_index)
            with self.tik_inst.else_scope():
                self._set_tail(core_index, 0)
        inputs_all = [self.input_proposal]
        outputs_all = [self.output_proposal]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, core_index):
        src_gm_rem_list = [self.sorted_num for _ in range(self.merge_channel)]
        self.merge_sort.merge_sort_gm_loop(self.output_proposal, self.input_proposal,
                                           core_index, 0, 0, src_gm_rem_list, self.k_num, self.sorted_num)
        self._set_tail(core_index, self.k_num)

    def _set_tail(self, core_index, data_num):
        tail_proposal_num = self.pro_repeat_num
        tail_num = tail_proposal_num * self.pro_data_num
        block_num = self.method.ceil_div(tail_num, self.block_data_num)
        ne_inf_ub = self.tik_inst.Tensor(self.data_type, (tail_num,), self.tik.scope_ubuf, "ne_inf_ub")
        self.method.vector_dup(ne_inf_ub, self.fp16_ne_inf)
        result_index = min(data_num, self.k_num)
        self.tik_inst.data_move(self.output_proposal[core_index, result_index, 0], ne_inf_ub, 0, 1, block_num, 0, 0)


def check_params(input_proposal, kernel_name):
    """
    check params
    """
    input_shape = input_proposal.get("shape")
    if len(input_shape) != 3 or input_shape[0] % 4 != 0 or input_shape[2] != 8:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of input_proposal", "support", "not support")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
# 'pylint: disable=unused-argument
def multi_merge(input_proposal, output_proposal, k_num, kernel_name="top_k_2"):
    """
    algorithm: merge and sort on single core
    Parameters
    ----------
    input_proposal:
        A Tensor. Proposal sorted for each channel. Support float16
    output_proposal :
        A Tensor. Datatype and format is same as input_data. Data sorted.
    k_num: int
        Number to be sorted.
    kernel_name : str
        cce kernel name, default value is top_k_3
    Returns
    -------
    None
    """
    input_shape = input_proposal.get("shape")
    input_dtype = input_proposal.get("dtype").lower()
    input_format = input_proposal.get("format")
    check_list = ("float16",)
    para_check.check_dtype(input_dtype, check_list, param_name="input_proposal")
    para_check.check_shape(input_shape, param_name="input_proposal")
    para_check.check_format(input_format)
    check_params(input_proposal, kernel_name)
    AContainer.reset_instance()
    cont = AContainer.get_instance()
    obj = MultiMerge(input_shape, k_num, input_dtype, kernel_name, cont)
    obj.mode_compute()
