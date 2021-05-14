# -*- coding: utf-8 -*-
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
from te import tik
import numpy as np

from . import constant_util as constant

UB_SIZE = 240 * 1024  # size of 310 ai core ub buffer


def ceil_div(value, factor):
    return (value + factor - 1) // factor


class HardSigmoid():
    """
    function_description:hard_sigmoid
          /- 0 if x < -2.5
     y = --- 1 if x > 2.5
          \- 0.2 * x + 0.5 if -2.5 <= x <= 2.5.
    Parameters
    ----------
    input_tensor:
        input: dict shape and dtype of input
    """

    def __init__(self, input_x, kernel_name="HardSigmoid"):
        self.shape = input_x.get("shape")
        self.dtype = input_x.get("dtype")
        self.kernel_name = kernel_name

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.data_each_block = constant.MAX_BLOCK_NUMBER // constant.DATA_SIZE_FOUR
        self.ub_size_bytes = UB_SIZE
        self.ub_tensor_size = (self.ub_size_bytes // constant.DATA_SIZE_FOUR // 4 //
                               self.data_each_block * self.data_each_block)

        self.vec_mask_max = 64
        # compute num of input data
        dim = len(self.shape)
        self.input_num = 1
        for i in range(0, dim):
            self.input_num *= self.shape[i]
        self.is_singleop = True

    def sigmoid_compute(self):
        self.input_x_gm = self.tik_instance.Tensor(self.dtype, self.shape, name="input_x_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, self.shape, name="output_gm", scope=tik.scope_gm)

        with self.tik_instance.for_range(0, 1) as index:
            self.input_x_ub = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="input_x_ub",
                                                       scope=tik.scope_ubuf)
            move_offset = index * self.input_num
            self.sigmoid_compute_each_core(move_offset, self.input_num)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_x_gm],
                                   outputs=[self.output_gm])
        return self.tik_instance

    def sigmoid_compute_each_core(self, move_offset, move_num):
        loop_time = move_num // self.ub_tensor_size
        move_offset_init = move_offset
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                move_offset += loop_index * self.ub_tensor_size
                self._sigmoid_compute_each_loop(move_offset, self.ub_tensor_size)
            move_offset = move_offset_init + loop_time * self.ub_tensor_size

        last_num_data = move_num % self.ub_tensor_size
        if (last_num_data > 0):
            self._sigmoid_compute_each_loop(move_offset, last_num_data)

    def _sigmoid_compute_each_loop(self, move_offset, move_num):

        if self.is_singleop:
            burst_len = ceil_div(move_num, self.data_each_block)
            self.tik_instance.data_move(self.input_x_ub, self.input_x_gm[move_offset], 0, 1, burst_len, 0, 0)

        tmp_ub = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="tmp_ub", scope=tik.scope_ubuf)
        vzero_ub = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="vzero_ub", scope=tik.scope_ubuf)
        vone_ub = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="vone_ub", scope=tik.scope_ubuf)

        scalar_zero = self.tik_instance.Scalar(init_value=0, dtype=self.dtype)
        scalar_one = self.tik_instance.Scalar(init_value=1, dtype=self.dtype)

        sigmoid_loop = move_num // (self.vec_mask_max * 255)
        offset = 0
        if sigmoid_loop > 0:
            with self.tik_instance.for_range(0, sigmoid_loop) as sigmoid_index:
                offset = sigmoid_index * self.vec_mask_max * 255
                self.tik_instance.vmuls(self.vec_mask_max, tmp_ub[offset],
                                        self.input_x_ub[offset], 0.2, 255, 1, 1, 8, 8)
                self.tik_instance.vadds(self.vec_mask_max, self.input_x_ub[offset],
                                        tmp_ub[offset], 0.5, 255, 1, 1, 8, 8)
                self.tik_instance.vector_dup(self.vec_mask_max, vzero_ub[offset], scalar_zero, 255, 1, 8)
                self.tik_instance.vector_dup(self.vec_mask_max, vone_ub[offset], scalar_one, 255, 1, 8)
                self.tik_instance.vmin(self.vec_mask_max, tmp_ub[offset], self.input_x_ub[offset],
                                       vone_ub[offset], 255, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmax(self.vec_mask_max, self.input_x_ub[offset],
                                       tmp_ub[offset], vzero_ub[offset], 255, 1, 1, 1, 8, 8, 8)

        repeat_time = move_num % (self.vec_mask_max * 255) // self.vec_mask_max
        if repeat_time > 0:
            offset = self.vec_mask_max * 255 * sigmoid_loop
            self.tik_instance.vmuls(self.vec_mask_max, tmp_ub[offset],
                                    self.input_x_ub[offset], 0.2, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vadds(self.vec_mask_max, self.input_x_ub[offset],
                                    tmp_ub[offset], 0.5, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vector_dup(self.vec_mask_max, vzero_ub[offset], scalar_zero, repeat_time, 1, 8)
            self.tik_instance.vector_dup(self.vec_mask_max, vone_ub[offset], scalar_one, repeat_time, 1, 8)
            self.tik_instance.vmin(self.vec_mask_max, tmp_ub[offset], self.input_x_ub[offset],
                                   vone_ub[offset], repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmax(self.vec_mask_max, self.input_x_ub[offset], tmp_ub[offset],
                                   vzero_ub[offset], repeat_time, 1, 1, 1, 8, 8, 8)

        last_num = move_num % self.vec_mask_max
        if last_num > 0:
            offset = move_num // (self.vec_mask_max) * (self.vec_mask_max)
            self.tik_instance.vmuls(last_num, tmp_ub[offset], self.input_x_ub[offset], 0.2, 1, 1, 1, 8, 8)
            self.tik_instance.vadds(last_num, self.input_x_ub[offset], tmp_ub[offset], 0.5, 1, 1, 1, 8, 8)
            self.tik_instance.vector_dup(last_num, vzero_ub[offset], scalar_zero, 1, 1, 8)
            self.tik_instance.vector_dup(last_num, vone_ub[offset], scalar_one, 1, 1, 8)
            self.tik_instance.vmin(last_num, tmp_ub[offset], self.input_x_ub[offset],
                                   vone_ub[offset], 1, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmax(last_num, self.input_x_ub[offset], tmp_ub[offset],
                                   vzero_ub[offset], 1, 1, 1, 1, 8, 8, 8)
        if self.is_singleop:
            self.tik_instance.data_move(self.output_gm[move_offset], self.input_x_ub, 0, 1, burst_len, 0, 0)

    def call_as_fuc(self, tik_inst, input_x, ub_size_used):
        self.is_singleop = False
        self.tik_instance = tik_inst
        self.ub_size_bytes = ub_size_used
        self.input_x_ub = input_x
        self.ub_tensor_size = (self.ub_size_bytes // constant.DATA_SIZE_FOUR // 3 //
                               self.data_each_block * self.data_each_block)

    def tik_output_debug(self):
        input_x_data = np.random.uniform(0.5, 0.5, self.shape).astype(np.float32)
        feed_dict = {"input_x_gm": input_x_data}
        output_gm, = self.sigmoid_compute().tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        print("output_data:\n{}".format(output_gm))
