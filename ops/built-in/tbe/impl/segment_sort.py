#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
from te.utils import para_check
from te.utils.error_manager import error_manager_vector

from impl.ascend import AContainer
from impl.merge_sort import CommonMethod
from impl.merge_sort import MergeSort


class SegmentSort:
    """define SegmentSort"""
    def __init__(self, data_num, index_num, data_type, k_num, kernel_name, cont):
        self.data_num = data_num
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
        self.ub_pro_num_max, self.ub_sort_num, self.each_loop_index_num = self.merge_sort.get_pro_num_info(index_num)
        self.fp16_index_num = index_num
        self.fp16_ne_inf = -65504.0
        self.tail_proposal_num = self.pro_repeat_num
        self.merge_channel = self.merge_sort.merge_channel_num

        self.result_shape, self.ai_core_use, self.channel_num = self.get_result_info()

        self.input_data = self.tik_inst.Tensor(self.data_type, (self.data_num, ), self.tik.scope_gm, "input_data")
        self.input_index = self.tik_inst.Tensor(self.data_type, (self.fp16_index_num, ),
                                                self.tik.scope_gm, "input_index")
        self.temp_proposal = self.tik_inst.Tensor(self.data_type, self.result_shape,
                                                  self.tik.scope_gm, "temp_proposal",
                                                  is_workspace=True)
        self.output_proposal = self.tik_inst.Tensor(self.data_type, self.result_shape,
                                                    self.tik.scope_gm, "output_proposal")

    def get_result_info(self):
        """get result shape, ai_core_use, channel_num"""
        ai_core_num = 32
        each_core_data_num = self.method.ceil_div(self.data_num, ai_core_num)
        each_core_data_num = self.method.get_align_num(each_core_data_num, self.each_loop_index_num)
        each_core_data_num = max(self.ub_pro_num_max, each_core_data_num)
        channel_num = self.method.ceil_div(self.data_num, each_core_data_num)
        if channel_num <= self.merge_channel:
            ai_core_use_num = channel_num
        else:
            ai_core_use_num = self.method.get_align_num(channel_num, self.merge_channel)

        each_core_data_num += self.tail_proposal_num
        result_shape = (ai_core_use_num, each_core_data_num, self.pro_data_num)
        return result_shape, ai_core_use_num, channel_num

    def mode_compute(self):
        """main compute"""
        each_core_data_num = self.result_shape[1] - self.tail_proposal_num
        last_core_data_num = self.data_num - each_core_data_num * (self.channel_num - 1)
        with self.tik_inst.for_range(0, self.ai_core_use, block_num=self.ai_core_use) as core_index:
            with self.tik_inst.if_scope(core_index < self.channel_num):
                data_index_start = each_core_data_num * core_index
                with self.tik_inst.if_scope(core_index != self.channel_num - 1):
                    self._mode_compute_each_core(core_index, data_index_start, each_core_data_num)
                with self.tik_inst.else_scope():
                    self._mode_compute_each_core(core_index, data_index_start, last_core_data_num)
            with self.tik_inst.else_scope():
                self._set_tail(core_index, 0)
        inputs_all = [self.input_data, self.input_index]
        outputs_all = [self.output_proposal]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, core_index, data_index_start, data_num):
        index_l1 = self._index_move_in()
        if data_num <= self.ub_pro_num_max or self.k_num <= self.ub_sort_num:
            self.merge_sort.get_top_proposal(self.output_proposal, core_index, data_num, self.k_num,
                                             self._get_proposal_ub, (index_l1, data_index_start))
        else:
            self.merge_sort.get_top_proposal_large(self.output_proposal, self.temp_proposal,
                                                   core_index, data_num, self.k_num,
                                                   self._get_proposal_ub, (index_l1, data_index_start))
        self._set_tail(core_index, data_num)

    def _index_move_in(self):
        index_l1 = self.tik_inst.Tensor(self.data_type, (self.fp16_index_num, ), self.tik.scope_cbuf, "index_l1")
        with self.tik_inst.new_stmt_scope():
            index_ub = self.tik_inst.Tensor(self.data_type, (self.fp16_index_num, ), self.tik.scope_ubuf, "index_ub")
            block_num = self.method.ceil_div(self.fp16_index_num, self.block_data_num)
            self.tik_inst.data_move(index_ub, self.input_index, 0, 1, block_num, 0, 0)
            self.tik_inst.data_move(index_l1, index_ub, 0, 1, block_num, 0, 0)
        return index_l1

    def _get_proposal_ub(self, boxes_num, start_index, index_l1, data_index_start):
        proposal_num = self.method.get_align_num(boxes_num, self.pro_repeat_num)
        ub_proposal_1 = self.tik_inst.Tensor(self.data_type, (proposal_num, self.pro_data_num),
                                             self.tik.scope_ubuf, "ub_proposal_1")
        with self.tik_inst.new_stmt_scope():
            boxes_index = start_index + data_index_start
            self._init_index_channel(ub_proposal_1, index_l1, boxes_index, proposal_num)
            self._init_score_channel(ub_proposal_1, boxes_index, boxes_num, proposal_num)
        ub_proposal_2 = self.tik_inst.Tensor(self.data_type, (proposal_num, self.pro_data_num),
                                             self.tik.scope_ubuf, "ub_proposal_2")
        return ub_proposal_1, ub_proposal_2

    def _init_index_channel(self, ub_proposal_1, index_l1, boxes_index, proposal_num):
        """
        algorithm: init index
            ub_proposal_1[:, 0] = index % each_loop_index_num
            ub_proposal_1[:, 1] = index // each_loop_index_num % each_loop_index_num
            ub_proposal_1[:, 2] = index // (each_loop_index_num * each_loop_index_num)
        Parameters
        ----------
        ub_proposal_1:
            ub tensor. (proposal_num , 8)
        index_l1:
            l1 Tensor. float16 range(0, 2048)
        boxes_index:
            data start index
        proposal_num:
            num of data processed
        Returns
        -------
        None
        """
        index_channel_0, index_channel_1, index_channel_2 = 0, 1, 2
        index_shape = (self.ub_pro_num_max, )
        index_ub_0 = self.tik_inst.Tensor(self.data_type, index_shape, self.tik.scope_ubuf, "index_ub_0")
        index_ub_1 = self.tik_inst.Tensor(self.data_type, index_shape, self.tik.scope_ubuf, "index_ub_1")
        index_ub_2 = self.tik_inst.Tensor(self.data_type, index_shape, self.tik.scope_ubuf, "index_ub_2")

        loop_time = self.ub_pro_num_max // self.each_loop_index_num
        index_block_num = self.each_loop_index_num // self.block_data_num
        self.tik_inst.data_move(index_ub_0, index_l1, 0, 1, index_block_num, 0, 0)

        with self.tik_inst.for_range(1, loop_time) as loop_index:
            index_ub_0_index = loop_index * self.each_loop_index_num
            self.tik_inst.data_move(index_ub_0[index_ub_0_index], index_ub_0, 0, 1, index_block_num, 0, 0)

        loop_time_align = self.method.get_align_num(loop_time, self.block_data_num)
        index_fp16_ub = self.tik_inst.Tensor(self.data_type, (2, loop_time_align), self.tik.scope_ubuf, "index_fp16_ub")
        pow_index = self.each_loop_index_num * self.each_loop_index_num
        int_type = "int32"
        index_1_s = self.tik_inst.Scalar(dtype=int_type)
        index_2_s = self.tik_inst.Scalar(dtype=int_type)
        index_1_s.set_as((boxes_index // self.each_loop_index_num) % self.each_loop_index_num)
        index_2_s.set_as(boxes_index // pow_index)
        index_1_block_num = loop_time_align // self.block_data_num
        index_2_block_num = 1

        self.tik_inst.data_move(index_fp16_ub[0, 0], self.input_index[index_1_s], 0, 1, index_1_block_num, 0, 0)
        self.tik_inst.data_move(index_fp16_ub[1, 0], self.input_index[index_2_s], 0, 1, index_2_block_num, 0, 0)

        index_fp16_scalar = self.tik_inst.Scalar(dtype=self.data_type)
        index_fp16_scalar.set_as(index_fp16_ub[1, 0])
        self.method.vector_dup(index_ub_2, index_fp16_scalar)

        with self.tik_inst.for_range(0, loop_time) as loop_index:
            index_fp16_scalar.set_as(index_fp16_ub[0, loop_index])
            self.method.vector_dup(index_ub_1, index_fp16_scalar,
                                   self.each_loop_index_num,
                                   self.each_loop_index_num * loop_index)

        self.method.vector_concat(ub_proposal_1, index_ub_0, index_channel_0, proposal_num)
        self.method.vector_concat(ub_proposal_1, index_ub_1, index_channel_1, proposal_num)
        self.method.vector_concat(ub_proposal_1, index_ub_2, index_channel_2, proposal_num)

    def _init_score_channel(self, ub_proposal_1, boxes_index, boxes_num, proposal_num):
        score_shape = (self.ub_pro_num_max,)
        score_ub = self.tik_inst.Tensor(self.data_type, score_shape, self.tik.scope_ubuf, "score_ub")
        score_label_0, score_label_1 = 3, 4
        score_block_num = self.method.ceil_div(boxes_num, self.block_data_num)
        self.tik_inst.data_move(score_ub, self.input_data[boxes_index, ], 0, 1, score_block_num, 0, 0)
        mask_h, mask_l, index_last = self.method.get_mask(boxes_num, self.repeat_data_num, self.pro_repeat_num)
        if mask_h != 0 or mask_l != 0:
            self.tik_inst.vector_dup([mask_h, mask_l], score_ub[index_last], self.fp16_ne_inf, 1, 1, 8)
        self.method.vector_concat(ub_proposal_1, score_ub, score_label_0, proposal_num)
        self.method.vector_concat(ub_proposal_1, score_ub, score_label_1, proposal_num)

    def _set_tail(self, core_index, data_num):
        tail_num = self.tail_proposal_num * self.pro_data_num
        block_num = self.method.ceil_div(tail_num, self.block_data_num)
        ne_inf_ub = self.tik_inst.Tensor(self.data_type, (tail_num,), self.tik.scope_ubuf, "ne_inf_ub")
        self.method.vector_dup(ne_inf_ub, self.fp16_ne_inf)
        result_index = min(data_num, self.k_num)
        self.tik_inst.data_move(self.output_proposal[core_index, result_index, 0], ne_inf_ub, 0, 1, block_num, 0, 0)


def check_params(input_data, input_index, kernel_name):
    """checking input params"""
    input_dtype = input_data.get("dtype").lower()
    if input_index.get("dtype") != input_dtype:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "dtype of input_data and {}".format("input_index"), "equal", "not equal")
    input_shape_1 = input_data.get("shape")
    input_shape_2 = input_index.get("shape")
    if len(input_shape_1) != 1:
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of input_data", "size equal 1", "size not equal 1")
    if input_shape_2 != (2048, ):
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of input_index", "equal (2048, )", "not equal (2048, )")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
# pylint: disable=unused-argument
def segment_sort(input_data, input_index, output_proposal, k_num, kernel_name="SegmentSort"):
    """algorithm: Segment merge sort on multiple core
    Parameters
    ----------
    input_data:
        A Tensor. Data to be sorted. Support float16
    input_index :
        A Tensor. Range(0, 2048). Datatype and format is same as input_data.
    output_proposal:
        A Tensor. Datatype and format is same as input_data. Proposal sorted for each channel.
    k_num: int
        Number to be sorted.
    kernel_name : str
        cce kernel name, default value is top_k_3
    Returns
    -------
    None
    """
    input_shape = input_data.get("shape")
    input_dtype = input_data.get("dtype").lower()
    input_format = input_data.get("format")
    check_list = ("float16", )
    para_check.check_dtype(input_dtype, check_list, param_name="input_data")
    para_check.check_shape(input_shape, param_name="input_data")
    para_check.check_format(input_format)
    check_params(input_data, input_index, kernel_name)

    AContainer.reset_instance()
    cont = AContainer.get_instance()
    data_num = input_shape[0]
    index_num = input_index.get("shape")[0]
    obj = SegmentSort(data_num, index_num, input_dtype, k_num, kernel_name, cont)
    obj.mode_compute()
