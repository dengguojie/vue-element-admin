#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

from te.utils import para_check
from te.utils.error_manager import error_manager_vector

from impl.ascend import AContainer
from impl.merge_sort import CommonMethod
from impl.merge_sort import MergeSort


class SingleMerge():
    """method to merge and sort on single core"""
    # 'pylint: disable=too-many-arguments
    def __init__(self, input_shape, k_num, data_type, kernel_name, cont):
        self.channel_num = input_shape[0]
        self.sorted_num = input_shape[1]
        self.k_num = k_num
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
        self.ub_pro_num_max, self.ub_sort_num, self.each_loop_index_num = self.merge_sort.get_pro_num_info()

        self.k_proposal_num = self.method.get_align_num(self.k_num, self.pro_repeat_num)

        proposal_shape = (1, self.channel_num * self.sorted_num, self.pro_data_num)
        self.input_proposal = self.tik_inst.Tensor(self.data_type, proposal_shape, self.tik.scope_gm, "input_proposal")
        self.temp_proposal = self.tik_inst.Tensor(self.data_type, proposal_shape,
                                                  self.tik.scope_gm, "temp_proposal",
                                                  is_workspace=True)

        self.int_type = "int32"
        self.int_data_size, self.int_block_data_num, self.int_repeat_data_num = self.method.get_type_const(
            self.int_type)

        result_shape = (self.k_num, )
        self.output_data = self.tik_inst.Tensor(self.data_type, result_shape,
                                                self.tik.scope_gm, "output_data")
        self.output_index = self.tik_inst.Tensor(self.int_type, result_shape,
                                                 self.tik.scope_gm, "output_index")

    def mode_compute(self):
        """main compute mode"""
        self._mode_compute_each_core()
        inputs_all = [self.input_proposal]
        outputs_all = [self.output_data, self.output_index]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self):
        src_gm_rem_list = [self.sorted_num] * self.channel_num
        with self.tik_inst.new_stmt_scope():
            self.merge_sort.merge_sort_gm_loop(self.temp_proposal, self.input_proposal, 0, 0, 0,
                                               src_gm_rem_list, self.k_proposal_num, self.sorted_num)
        each_loop_data_num = self._get_loop_data_num()
        loop_time, last_loop_data_num = self.method.get_loop_info(self.k_num, each_loop_data_num)
        with self.tik_inst.for_range(0, loop_time) as loop_index:
            data_index_start = loop_index * each_loop_data_num
            with self.tik_inst.if_scope(loop_index != loop_time - 1):
                self._result_move_out_each_loop(data_index_start, each_loop_data_num)
            with self.tik_inst.else_scope():
                self._result_move_out_each_loop(data_index_start, last_loop_data_num)

    def _get_loop_data_num(self):
        each_data_size = self.data_size * (4 + self.pro_data_num) + self.int_data_size * 3
        data_num = self.ub_size // each_data_size
        data_num_align = self.method.get_align_num(data_num, self.int_repeat_data_num, False)
        return data_num_align

    # 'pylint: disable=too-many-locals
    def _result_move_out_each_loop(self, data_index_start, data_num):
        """algorithm: get result_data, result_index
            result_data = gm_tensor[0, :, 3]
            result_index = gm_tensor[0, :, 0] + gm_tensor[0, :, 0]*each_loop_index_num
                           + gm_tensor[0, :, 0]*each_loop_index_num*each_loop_index_num
        Parameters
        ----------
        data_index_start:
            int. gm tensor index
        data_num :
            int. num of data processed
        Returns
        -------
        None
        """
        proposal_num = self.method.get_align_num(data_num, self.pro_repeat_num)
        data_num_align = self.method.get_align_num(data_num, self.int_repeat_data_num)
        data_shape = (data_num_align, )
        index_ub_0 = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "index_ub_0")
        index_ub_1 = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "index_ub_1")
        index_ub_2 = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "index_ub_2")
        score_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "score_ub")

        index_int_ub_0 = self.tik_inst.Tensor(self.int_type, data_shape, self.tik.scope_ubuf, "index_int_ub_0")
        index_int_ub_1 = self.tik_inst.Tensor(self.int_type, data_shape, self.tik.scope_ubuf, "index_int_ub_1")
        index_int_ub_2 = self.tik_inst.Tensor(self.int_type, data_shape, self.tik.scope_ubuf, "index_int_ub_2")

        proposal_shape = (proposal_num, self.pro_data_num)
        proposal_ub = self.tik_inst.Tensor(self.data_type, proposal_shape, self.tik.scope_ubuf, "proposal_ub")
        block_num_move_in = self.method.ceil_div(data_num * self.pro_data_num, self.block_data_num)

        self.tik_inst.data_move(proposal_ub, self.temp_proposal[0, data_index_start, 0], 0, 1, block_num_move_in, 0, 0)
        self.method.vector_extract(index_ub_0, proposal_ub, 0, proposal_num)
        self.method.vector_extract(index_ub_1, proposal_ub, 1, proposal_num)
        self.method.vector_extract(index_ub_2, proposal_ub, 2, proposal_num)
        self.method.vector_extract(score_ub, proposal_ub, 3, proposal_num)

        mask = self.int_repeat_data_num
        repeat_num = data_num_align // self.int_repeat_data_num

        self.tik_inst.vconv(mask, "round", index_int_ub_0, index_ub_0, repeat_num, 1, 1, 8, 4)
        self._get_int_index(index_int_ub_0, index_int_ub_1, index_ub_1, index_int_ub_2, mask, repeat_num,
                            self.each_loop_index_num)
        mul_num = self.each_loop_index_num * self.each_loop_index_num
        self._get_int_index(index_int_ub_0, index_int_ub_1, index_ub_2, index_int_ub_2, mask, repeat_num, mul_num)

        data_block_num = self.method.ceil_div(data_num, self.block_data_num)
        index_block_num = self.method.ceil_div(data_num, self.int_block_data_num)
        self.tik_inst.data_move(self.output_data[data_index_start], score_ub, 0, 1, data_block_num, 0, 0)
        self.tik_inst.data_move(self.output_index[data_index_start], index_int_ub_0, 0, 1, index_block_num, 0, 0)

    # 'pylint: disable=too-many-arguments
    def _get_int_index(self, result_index, index_int_ub, index_ub, index_int_ub_temp, mask, repeat_num, mul_num):
        self.tik_inst.vconv(mask, "round", index_int_ub, index_ub, repeat_num, 1, 1, 8, 4)
        self.tik_inst.vector_dup(mask, index_int_ub_temp, mul_num, repeat_num, 1, 8)
        self.tik_inst.vmul(mask, index_int_ub, index_int_ub, index_int_ub_temp, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(mask, result_index, result_index, index_int_ub, repeat_num, 1, 1, 1, 8, 8, 8)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def single_merge(input_proposal, output_data, output_index, k_num, kernel_name="top_k_3"):
    """algorithm: merge and sort on single core
    Parameters
    ----------
    input_proposal:
        A Tensor. Proposal sorted for each channel. Support float16
    output_data :
        A Tensor. Datatype and format is same as input_data. Data sorted.
    output_index:
        A Tensor. int32. Data index.
    k_num: int
        Number to be sorted.
    kernel_name : str
        cce kernel name, default value is top_k_3
    Returns
    -------
    None
    """
    def _check_params(input_proposal, kernel_name):
        input_shape = input_proposal.get("shape")
        if len(input_shape) != 3 or input_shape[0] > 4 or input_shape[2] != 8:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "shape of input_proposal", "support", "not support")

    input_shape = input_proposal.get("shape")
    input_dtype = input_proposal.get("dtype").lower()
    input_format = input_proposal.get("format")
    check_list = ("float16",)
    para_check.check_dtype(input_dtype, check_list, param_name="input_proposal")
    para_check.check_shape(input_shape, param_name="input_proposal")
    para_check.check_format(input_format)
    _check_params(input_proposal, kernel_name)
    AContainer.reset_instance()
    cont = AContainer.get_instance()
    obj = SingleMerge(input_shape, k_num, input_dtype, kernel_name, cont)
    obj.mode_compute()
