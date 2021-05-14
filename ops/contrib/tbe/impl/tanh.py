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
from functools import reduce as functools_reduce

TAYLOR_NEGATIVE_THRESHOLD = -8
TAYLOR_POSITIVE_THRESHOLD = 8
BLOCK_NUM = 8  # The number of blocks processed by the ai core at a time
UB_SIZE = 240 * 1024  # size of 310 ai core ub buffer
AI_COER_NUM = 2


class Tanh():
    """
    function_description:caculate tanh which is achieved by fp32 instruction
    Parameters
    ----------
    input_tensor:
        tan_input: dict shape and dtype of input , dtype should be fp32
    """

    def __init__(self, tan_input, kernel_name="Tanh"):
        self.shape = tan_input.get("shape")
        self.dtype = tan_input.get("dtype")
        self.kernel_name = kernel_name

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.aicore_num = AI_COER_NUM
        block_bite_size = 32
        ub_size_bytes = UB_SIZE
        self.dtype_bytes_size = 4
        self.vector_mask_max = block_bite_size // self.dtype_bytes_size * 8
        self.data_each_block = block_bite_size // self.dtype_bytes_size

        self.input_num = functools_reduce(lambda x, y: x * y, self.shape)
        if self.dtype != "float32":
            raise RuntimeError("The dtype of input should be float32")
        if self.input_num % 16 != 0:
            raise RuntimeError("The num of input data should be a multiple of 16")
        self.ub_tensor_size = (ub_size_bytes // self.dtype_bytes_size // 13 //
                               self.data_each_block * self.data_each_block)

        num_per_core = self._div_offline(self._div_offline(self.input_num, self.aicore_num), 16) * 16
        self.data_num_each_core = num_per_core
        self.data_num_last_core = self.input_num - (self.aicore_num - 1) * num_per_core
        self.is_singleop = True

    def tanh_compute(self):
        self.input_gm = self.tik_instance.Tensor(self.dtype, self.shape, name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, self.shape, name="output_gm", scope=tik.scope_gm)
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index:
            self.tanh_input = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="tanh_input",
                                                       scope=tik.scope_ubuf)
            if self.is_singleop:
                self._gen_tan_ub_tensor()

            move_offset = index * self.data_num_each_core

            with self.tik_instance.if_scope(index != self.aicore_num - 1):
                self.tanh_compute_each_core(move_offset, self.data_num_each_core)
            with self.tik_instance.else_scope():
                self.tanh_compute_each_core(move_offset, self.data_num_last_core)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_gm],
            outputs=[self.output_gm])
        return self.tik_instance

    def tanh_compute_each_core(self, move_offset, move_num):
        if not self.is_singleop:
            self._gen_tan_ub_tensor()

        loop_time = move_num // self.ub_tensor_size
        move_offset_init = move_offset
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                move_offset += loop_index * self.ub_tensor_size
                self._tanh_compute_each_loop(move_offset, self.ub_tensor_size)
            move_offset = move_offset_init + loop_time * self.ub_tensor_size

        last_num = move_num % self.ub_tensor_size

        if last_num > 0:
            self._tanh_compute_each_loop(move_offset, last_num)

    # Use the Newton iteration method (iterate twice) to calculate the reciprocal
    # Note that the output will cover the input
    def _rec_compute_each_vec_loop2(self, input_tensor, mask, offset, repeat_times):
        # Calculate the initial iteration value x0, which is obtained using inaccurate vrec
        self.tik_instance.vrec(mask, self.rec_tmp1[offset], input_tensor[offset], repeat_times, 1, 1, 8, 8)

        # Calculate a * x0 (a is the input, x0 is the initial value of iteration)
        self.tik_instance.vmul(mask, self.rec_tmp2[offset], self.rec_tmp1[offset], input_tensor[offset], repeat_times,
                               1, 1, 1, 8, 8, 8)

        # compute: a*x0 - 2
        self.tik_instance.vadds(mask, self.rec_tmp3[offset], self.rec_tmp2[offset], self.scalar_neg_2, repeat_times, 1,
                                1, 8, 8)

        # compute: x0*(ax0 - 2))
        self.tik_instance.vmul(mask, self.rec_tmp2[offset], self.rec_tmp3[offset], self.rec_tmp1[offset], repeat_times,
                               1, 1, 1, 8, 8, 8)

        # x1 is defined as: -x0*(ax0 - 2)
        self.tik_instance.vmuls(mask, self.rec_tmp1[offset], self.rec_tmp2[offset], self.scalar_neg_1, repeat_times, 1,
                                1, 8, 8)

        # Calculate a * x1 (a is the input, x1 is the iteration value)
        self.tik_instance.vmul(mask, self.rec_tmp2[offset], self.rec_tmp1[offset], input_tensor[offset], repeat_times,
                               1, 1, 1, 8, 8, 8)

        # compute: a*x1 - 2
        self.tik_instance.vadds(mask, self.rec_tmp3[offset], self.rec_tmp2[offset], self.scalar_neg_2, repeat_times, 1,
                                1, 8, 8)

        # compute: x1*(ax1 - 2)
        self.tik_instance.vmul(mask, self.rec_tmp2[offset], self.rec_tmp3[offset], self.rec_tmp1[offset], repeat_times,
                               1, 1, 1, 8, 8, 8)

        # x2 is defined as: -x1*(ax1 - 2)
        self.tik_instance.vmuls(mask, input_tensor[offset], self.rec_tmp2[offset], self.scalar_neg_1, repeat_times, 1,
                                1, 8, 8)

    # The input is self.tanh_taylor_tmp1 and the output is self.tanh_taylor_tmp1
    def _tanh_double_angle_vec_loop(self, mask, offset, repeat_times):
        # the input as w (=tanh (x))
        # it is to compute: 2*w
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp2[offset], self.tanh_taylor_tmp1[offset], self.scalar_2,
                                repeat_times, 1, 1, 8, 8)

        # it is to compute: w**2
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp3[offset], self.tanh_taylor_tmp1[offset],
                               self.tanh_taylor_tmp1[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: 1 + w**2
        self.tik_instance.vadds(mask, self.tanh_taylor_tmp4[offset], self.tanh_taylor_tmp3[offset], self.scalar_1,
                                repeat_times, 1, 1, 8, 8)

        # it is to compute: 1 / (1 + w**2)
        self._rec_compute_each_vec_loop2(self.tanh_taylor_tmp4, mask, offset, repeat_times)

        # it is to compute: 2*w / (1 + w**2)
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp1[offset], self.tanh_taylor_tmp4[offset],
                               self.tanh_taylor_tmp2[offset], repeat_times, 1, 1, 1, 8, 8, 8)

    def _tanh_taylor_vec_loop(self, input_tensor, mask, offset, repeat_times):
        # it is to compute: x**2
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp1[offset], input_tensor[offset], input_tensor[offset],
                               repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: x**3
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp2[offset], self.tanh_taylor_tmp1[offset], input_tensor[offset],
                               repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: - x**3 / 3
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp3[offset], self.tanh_taylor_tmp2[offset], self.taylor_coeff3,
                                repeat_times, 1, 1, 8, 8)

        # it is to compute: x - x**3/3
        self.tik_instance.vadd(mask, self.tanh_taylor_tmp4[offset], input_tensor[offset], self.tanh_taylor_tmp3[offset],
                               repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: x**5
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp3[offset], self.tanh_taylor_tmp1[offset],
                               self.tanh_taylor_tmp2[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: 2*x**5/15
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp2[offset], self.tanh_taylor_tmp3[offset], self.taylor_coeff5,
                                repeat_times, 1, 1, 8, 8)

        # it is to compute: x - x**3/3 + 2*x**5/15
        self.tik_instance.vadd(mask, self.tanh_taylor_tmp5[offset], self.tanh_taylor_tmp4[offset],
                               self.tanh_taylor_tmp2[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: x**7
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp2[offset], self.tanh_taylor_tmp1[offset],
                               self.tanh_taylor_tmp3[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: -17/315 * x**7
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp3[offset], self.tanh_taylor_tmp2[offset], self.taylor_coeff7,
                                repeat_times, 1, 1, 8, 8)

        # it is to compute: x - x**3/3 + 2*x**5/15 - 17/315*x**7
        self.tik_instance.vadd(mask, self.tanh_taylor_tmp4[offset], self.tanh_taylor_tmp5[offset],
                               self.tanh_taylor_tmp3[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: x**9
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp3[offset], self.tanh_taylor_tmp1[offset],
                               self.tanh_taylor_tmp2[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: 62/2835*x**9
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp2[offset], self.tanh_taylor_tmp3[offset], self.taylor_coeff9,
                                repeat_times, 1, 1, 8, 8)

        # it is to compute: x - x**3/3 + 2*x**5/15 - 17/315*x**7 + 62/2835*x**9
        self.tik_instance.vadd(mask, self.tanh_taylor_tmp5[offset], self.tanh_taylor_tmp4[offset],
                               self.tanh_taylor_tmp2[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: x**11
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp2[offset], self.tanh_taylor_tmp1[offset],
                               self.tanh_taylor_tmp3[offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: -1382/155925*x**11
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp3[offset], self.tanh_taylor_tmp2[offset], self.taylor_coeff11,
                                repeat_times, 1, 1, 8, 8)

        # it is to compute: x - x**3/3 + 2*x**5/15 - 17/315*x**7 + 62/2835*x**9 - 1382/155925*x**11
        self.tik_instance.vadd(mask, self.tanh_taylor_tmp1[offset], self.tanh_taylor_tmp5[offset],
                               self.tanh_taylor_tmp3[offset], repeat_times, 1, 1, 1, 8, 8, 8)

    def _tanh_compute_each_vec_loop(self, mask, tanh_offset, repeat_times):
        # it is to compute: x / 2
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp6[tanh_offset], self.tanh_input[tanh_offset],
                                self.scalar_reverse_16, repeat_times, 1, 1, 8, 8)

        # taylor tanh
        self._tanh_taylor_vec_loop(self.tanh_taylor_tmp6, mask, tanh_offset, repeat_times)

        # Double angle formula, output as self.tanh_taylor_tmp1
        for _ in range(4):
            self._tanh_double_angle_vec_loop(mask, tanh_offset, repeat_times)

        self.tik_instance.vconv(mask, '', self.input_fp16[tanh_offset], self.tanh_input[tanh_offset], repeat_times, 1,
                                1, 4, 8)

        # When input_fp16 > neg_thredhold, the bit corresponding to cmpmask is 1
        for i in range(0, repeat_times):
            cmpmask = self.tik_instance.vcmp_gt(mask, self.input_fp16[tanh_offset + i * mask], self.neg_thredhold, 1, 1)

            # When cmpmask is 1, vsel_tmp1_fp16 = ones_fp16, otherwise zeros_fp16
            self.tik_instance.vsel(mask, 0, self.vsel_tmp1_fp16[tanh_offset + i * mask], cmpmask, self.ones_fp16,
                                   self.zeros_fp16, 1, 1, 1, 1, 8, 8, 8)

        # turn fp16 to fp32
        self.tik_instance.vconv(mask, '', self.tanh_taylor_tmp2[tanh_offset], self.vsel_tmp1_fp16[tanh_offset],
                                repeat_times, 1, 1, 8, 4)

        # When x bigger than neg_thredhold, tanh_taylor_tmp2 is 1; otherwise 0
        # set tanh_taylor_tmp2 as a, and the double angle formula calculates y1
        # a * y1
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp3[tanh_offset], self.tanh_taylor_tmp1[tanh_offset],
                               self.tanh_taylor_tmp2[tanh_offset], repeat_times, 1, 1, 1, 8, 8, 8)
        # it is to compute: a - 1
        self.tik_instance.vadds(mask, self.tanh_taylor_tmp4[tanh_offset], self.tanh_taylor_tmp2[tanh_offset],
                                self.scalar_neg_1, repeat_times, 1, 1, 8, 8)

        # it is to compute: a*y1 + a - 1
        self.tik_instance.vadd(mask, self.tanh_taylor_tmp1[tanh_offset], self.tanh_taylor_tmp4[tanh_offset],
                               self.tanh_taylor_tmp3[tanh_offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # Calculate the divided interval of pos_thredhold
        # When input_fp16 <pos_thredhold, the bit corresponding to cmpmask is 1, otherwise it is 0
        for i in range(0, repeat_times):
            cmpmask = self.tik_instance.vcmp_lt(mask, self.input_fp16[tanh_offset + i * mask], self.pos_thredhold, 1, 1)

            # When cmpmask is 1, vsel_tmp1_fp16 = ones_fp16, otherwise it is zeros_fp16
            self.tik_instance.vsel(mask, 0, self.vsel_tmp1_fp16[tanh_offset + i * mask], cmpmask, self.ones_fp16,
                                   self.zeros_fp16, 1, 1, 1, 1, 8, 8, 8)

        # turn fp16 to fp32
        self.tik_instance.vconv(mask, '', self.tanh_taylor_tmp2[tanh_offset], self.vsel_tmp1_fp16[tanh_offset],
                                repeat_times, 1, 1, 8, 4)

        # When x smaller than neg_thredhold, tanh_taylor_tmp2 = 1; otherwise 0
        # set tanh_taylor_tmp2 as a, and the double angle formula calculates y1
        # it is to compute: a * y1
        self.tik_instance.vmul(mask, self.tanh_taylor_tmp3[tanh_offset], self.tanh_taylor_tmp1[tanh_offset],
                               self.tanh_taylor_tmp2[tanh_offset], repeat_times, 1, 1, 1, 8, 8, 8)

        # it is to compute: -a
        self.tik_instance.vmuls(mask, self.tanh_taylor_tmp4[tanh_offset], self.tanh_taylor_tmp2[tanh_offset],
                                self.scalar_neg_1, repeat_times, 1, 1, 8, 8)

        # it is to compute: -a + 1
        self.tik_instance.vadds(mask, self.tanh_taylor_tmp5[tanh_offset], self.tanh_taylor_tmp4[tanh_offset],
                                self.scalar_1, repeat_times, 1, 1, 8, 8)

        # it is to compute: -a + 1 + a*y1
        self.tik_instance.vadd(mask, self.tanh_input[tanh_offset], self.tanh_taylor_tmp5[tanh_offset],
                               self.tanh_taylor_tmp3[tanh_offset], repeat_times, 1, 1, 1, 8, 8, 8)

    def _tanh_compute_each_loop(self, move_offset, move_num):
        burst_len = self._div_offline(move_num, self.data_each_block)
        if self.is_singleop:
            self.tik_instance.data_move(self.tanh_input, self.input_gm[move_offset], 0, 1, burst_len, 0, 0)
        tanh_loop = move_num // (self.vector_mask_max * 255)

        tanh_offset = 0
        if tanh_loop > 0:
            with self.tik_instance.for_range(0, tanh_loop) as index:
                tanh_offset = index * self.vector_mask_max * 255
                self._tanh_compute_each_vec_loop(self.vector_mask_max, tanh_offset, 255)

        repeat_times = (move_num % (self.vector_mask_max * 255) // self.vector_mask_max)
        if repeat_times > 0:
            tanh_offset = self.vector_mask_max * 255 * tanh_loop
            self._tanh_compute_each_vec_loop(self.vector_mask_max, tanh_offset, repeat_times)
        left_num = move_num % (self.vector_mask_max)
        if left_num > 0:
            tanh_offset = move_num // (self.vector_mask_max) * (self.vector_mask_max)
            self._tanh_compute_each_vec_loop(left_num, tanh_offset, 1)

        if self.is_singleop:
            self.tik_instance.data_move(self.output_gm[move_offset], self.tanh_input, 0, 1, burst_len, 1, 0)

    def _gen_tan_ub_tensor(self):
        self.input_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size,), name="input_fp16",
                                                   scope=tik.scope_ubuf)
        self.tanh_taylor_tmp1 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,),
                                                         name="tanh_taylor_tmp1", scope=tik.scope_ubuf)
        self.tanh_taylor_tmp2 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,),
                                                         name="tanh_taylor_tmp2", scope=tik.scope_ubuf)
        self.tanh_taylor_tmp3 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,),
                                                         name="tanh_taylor_tmp3", scope=tik.scope_ubuf)
        self.tanh_taylor_tmp4 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,),
                                                         name="tanh_taylor_tmp4", scope=tik.scope_ubuf)
        self.tanh_taylor_tmp5 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,),
                                                         name="tanh_taylor_tmp5", scope=tik.scope_ubuf)
        self.tanh_taylor_tmp6 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,),
                                                         name="tanh_taylor_tmp6", scope=tik.scope_ubuf)
        self.ones_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size,), name="ones_fp16",
                                                  scope=tik.scope_ubuf)
        self.zeros_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size,), name="zeros_fp16",
                                                   scope=tik.scope_ubuf)
        self.vsel_tmp1_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size,), name="vsel_tmp1_fp16",
                                                       scope=tik.scope_ubuf)
        self.vsel_tmp2_fp16 = self.tik_instance.Tensor("float16", (self.ub_tensor_size,), name="vsel_tmp2_fp16",
                                                       scope=tik.scope_ubuf)
        self.pos_thredhold = self.tik_instance.Tensor("float16", (128,), name="pos_thredhold", scope=tik.scope_ubuf)

        self.neg_thredhold = self.tik_instance.Tensor("float16", (128,), name="neg_thredhold", scope=tik.scope_ubuf)

        self.rec_tmp1 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="rec_tmp1",
                                                 scope=tik.scope_ubuf)
        self.rec_tmp2 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="rec_tmp2",
                                                 scope=tik.scope_ubuf)
        self.rec_tmp3 = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size,), name="rec_tmp3",
                                                 scope=tik.scope_ubuf)

        self.scalar_neg_1 = self.tik_instance.Scalar(dtype=self.dtype)
        self.scalar_1 = self.tik_instance.Scalar(dtype=self.dtype)
        self.scalar_2 = self.tik_instance.Scalar(dtype=self.dtype)
        self.scalar_n = self.tik_instance.Scalar(dtype=self.dtype)
        self.scalar_neg_2 = self.tik_instance.Scalar(dtype=self.dtype)
        self.scalar_reverse_16 = self.tik_instance.Scalar(dtype=self.dtype)
        self.taylor_coeff3 = self.tik_instance.Scalar(dtype=self.dtype)
        self.taylor_coeff5 = self.tik_instance.Scalar(dtype=self.dtype)
        self.taylor_coeff7 = self.tik_instance.Scalar(dtype=self.dtype)
        self.taylor_coeff9 = self.tik_instance.Scalar(dtype=self.dtype)
        self.taylor_coeff11 = self.tik_instance.Scalar(dtype=self.dtype)
        self._init_scalar_and_vector()

    def _init_scalar_and_vector(self):
        self.tik_instance.vector_dup(128, self.ones_fp16, 1.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, self.zeros_fp16, 0.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, self.pos_thredhold, TAYLOR_POSITIVE_THRESHOLD, 1, 1, 8)
        self.tik_instance.vector_dup(128, self.neg_thredhold, TAYLOR_NEGATIVE_THRESHOLD, 1, 1, 8)
        self.scalar_neg_1.set_as(-1.0)
        self.scalar_1.set_as(1.0)
        self.scalar_2.set_as(2.0)
        self.scalar_n.set_as(1.0 / 4096.0)
        self.scalar_neg_2.set_as(-2.0)
        self.scalar_reverse_16.set_as(1.0 / 16.0)
        self.taylor_coeff3.set_as(-1.0 / 3.0)
        self.taylor_coeff5.set_as(2.0 / 15.0)
        self.taylor_coeff7.set_as(-17.0 / 315.0)
        self.taylor_coeff9.set_as(62.0 / 2835.0)
        self.taylor_coeff11.set_as(-1382 / 155925.0)

    def _div_offline(self, value, factor):
        if value % factor == 0:
            return value // factor
        else:
            return value // factor + 1

    def call_as_fuc(self, tik_inst, input_x, ub_size_used):
        self.is_singleop = False
        self.tik_instance = tik_inst
        self.ub_size_bytes = ub_size_used
        self.tanh_input = input_x
        self.ub_tensor_size = (self.ub_size_bytes // self.dtype_bytes_size // 12 //
                               self.data_each_block * self.data_each_block)
        self.aicore_use = 1

    def tik_output_debug(self):
        input_data = np.linspace(-0.1, 0.1, num=self.shape[0]).astype(np.float32)
        move_num = 1
        for dim in self.shape:
            move_num *= dim

        print("input_data:", input_data)
        feed_dict = {"input_gm": input_data}
        output_gm, = self.tanh_compute().tikdb.start_debug(feed_dict=feed_dict, interactive=False)
        print("\noutput_data:\n{}".format(output_gm))
        expect_data = np.tanh(input_data)
        print("\nexpect_data:\n", expect_data)
        real_error_rate = np.abs(expect_data - output_gm) / (np.abs(expect_data) + 1e-8)
        print("real_error_rate:\n", real_error_rate)
        print("max_error_rate:\n", np.max(np.abs(real_error_rate)))
        print("avg_error_rate:\n", np.mean(np.abs(real_error_rate)))
        print("max_error_input_val:", input_data[np.argmax(np.abs(real_error_rate))])
        input_data.tofile("input.data")
        real_error_rate.tofile("real_error_rate.data")
