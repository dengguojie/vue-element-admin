# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""

from te import tik
from functools import reduce as functools_reduce

BLOCK_NUM = 8  # block number proceed by each ai core
UB_SIZE = 240 * 1024  # size of 310 ai core ub buffer
AI_COER_NUM = 2


def ceil_div_offline(value, factor):
    return (value + factor - 1) // factor


def check_args(input0, output):
    if input0.get('shape') != output.get('shape'):
        raise RuntimeError('shape of input0 and output must be equal')
    if input0.get('dtype') != 'float16':
        raise RuntimeError('input0 data type must be float16')
    if input0.get('format') != 'NC1HWC0':
        raise RuntimeError('input0 format must be NC1HWC0')


class DtypeConfig():
    fp16 = {'bytes_size': 2, 'vector_mask_max': 128}
    int32 = {'bytes_size': 4, 'vector_mask_max': 64}


class Ceil():
    """
    Parameters
    ----------
    function_description : returns element-wise smallest integer not less than x
    input_0 : dict shape and dtype of input0
    output_0 : dict shape and dtype of output, should be same shape and type as input0
    kernel_name : kernel name, default value is "Ceil"
    -------
    Returns： None
    """

    def __init__(self, input0, output0, kernel_name="Ceil"):
        self.shape = input0.get("shape")
        self.dtype = input0.get("dtype")
        self.kernel_name = kernel_name
        check_args(input0, output0)
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.aicore_num = AI_COER_NUM
        block_bite_size = 32
        ub_size_bytes = UB_SIZE

        self.dtype_bytes_size_float16 = DtypeConfig.fp16["bytes_size"]
        self.vector_mask_max_float16 = DtypeConfig.fp16['vector_mask_max']
        self.data_each_block_float16 = block_bite_size // self.dtype_bytes_size_float16
        self.dtype_bytes_size_int32 = DtypeConfig.int32["bytes_size"]
        self.vector_mask_max_int32 = DtypeConfig.int32['vector_mask_max']
        self.data_each_block_int32 = block_bite_size // self.dtype_bytes_size_int32

        # tensor size
        self.ub_tensor_size = (ub_size_bytes // self.dtype_bytes_size_float16 // 3
                               // self.data_each_block_float16 * self.data_each_block_float16)
        # data proceed by each core
        self.input_num = functools_reduce(lambda x, y: x * y, self.shape)
        num_per_core = ceil_div_offline(ceil_div_offline(self.input_num, self.aicore_num), 16) * 16
        if self.input_num % self.aicore_num == 0:
            is_same_core = 0
        else:
            is_same_core = 1

        self.aicore_use = self.input_num // num_per_core + (
            0 if self.input_num // self.aicore_num == 0 else is_same_core)
        self.data_num_each_core = num_per_core
        self.data_num_last_core = self.input_num - (self.aicore_use - 1) * num_per_core
        self.input_gm = self.tik_instance.Tensor(self.dtype, (self.shape), name="input_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (self.shape), name="output_gm", scope=tik.scope_gm)

    def tilling_mode_select(self):
        self.mode = 1

    def ceil_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as index:
            move_offset = index * self.data_num_each_core
            last_core_index = self.aicore_use - 1
            with self.tik_instance.if_scope(index != last_core_index):
                self._ceil_compute_each_core(move_offset, self.data_num_each_core)
            with self.tik_instance.else_scope():
                self._ceil_compute_each_core(move_offset, self.data_num_last_core)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm],
                                   outputs=[self.output_gm], enable_l2=False)
        return self.tik_instance

    def _ceil_compute_each_core(self, move_offset, move_num):  # move_num should align to 32
        loop_time = move_num // self.ub_tensor_size
        move_offset_init = move_offset
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                move_offset += loop_index * self.ub_tensor_size
                self._ceil_compute_each_loop(move_offset, self.ub_tensor_size)
            move_offset = move_offset_init + loop_time * self.ub_tensor_size

        last_num_data = move_num % self.ub_tensor_size
        if last_num_data > 0:
            self._ceil_compute_each_loop(move_offset, last_num_data)

    def _ceil_compute_each_loop(self, move_offset, move_num):
        move_num_tmp = move_num
        move_num = move_num - move_num % 32

        if move_num > 0:
            move_num = move_num // 2  # move_num align to 16

            with self.tik_instance.for_range(0, 2, thread_num=2) as index_thread:
                self.input_x_ub = self.tik_instance.Tensor(self.dtype, (self.ub_tensor_size // 2,),
                                                           name="input_ub", scope=tik.scope_ubuf)
                self.input_vconv_ub = self.tik_instance.Tensor('int32', (self.ub_tensor_size // 2,),
                                                               name="vconv_ub", scope=tik.scope_ubuf)
                burst_len_float16 = ceil_div_offline(move_num, self.data_each_block_float16)
                self.tik_instance.data_move(self.input_x_ub, self.input_gm[move_offset + index_thread * move_num],
                                            0, 1, burst_len_float16, 0, 0)

                self._ceil_compute_each_thread(move_num)

                self.tik_instance.data_move(self.output_gm[move_offset + index_thread * move_num], self.input_x_ub,
                                            0, 1, burst_len_float16, 1, 0)

        if move_num_tmp % 32 > 0:
            end_data = ceil_div_offline(move_num_tmp % 32, 16) * 16
            self.input_x_ub = self.tik_instance.Tensor(self.dtype, (end_data,), name="input_ub", scope=tik.scope_ubuf)
            self.input_vconv_ub = self.tik_instance.Tensor('int32', (end_data,), name="vconv_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.input_x_ub, self.input_gm[move_offset + move_num_tmp - end_data],
                                        0, 1, 1, 0, 0)
            self.tik_instance.vconv(end_data, 'ceil', self.input_vconv_ub,
                                    self.input_x_ub, 1, 1, 1, 8, 4)
            self.tik_instance.vconv(end_data, '', self.input_x_ub,
                                    self.input_vconv_ub, 1, 1, 1, 4, 8, 1.0)
            self.tik_instance.data_move(self.output_gm[move_offset + move_num_tmp - end_data], self.input_x_ub,
                                        0, 1, 1, 0, 0)

    def _ceil_compute_each_thread(self, move_num):
        ceil_loop = move_num // (self.vector_mask_max_int32 * 255)
        ceil_offset = 0

        if ceil_loop > 0:
            with self.tik_instance.for_range(0, ceil_loop) as index:
                ceil_offset = index * self.vector_mask_max_int32 * 255
                self.tik_instance.vconv(self.vector_mask_max_int32, 'ceil', self.input_vconv_ub[ceil_offset],
                                        self.input_x_ub[ceil_offset], 255, 1, 1, 8, 4)
                self.tik_instance.vconv(self.vector_mask_max_int32, '', self.input_x_ub[ceil_offset],
                                        self.input_vconv_ub[ceil_offset], 255, 1, 1, 4, 8, 1.0)

        repeat_time = (move_num % (self.vector_mask_max_int32 * 255) // self.vector_mask_max_int32)

        if repeat_time > 0:
            ceil_offset = self.vector_mask_max_int32 * 255 * ceil_loop
            self.tik_instance.vconv(self.vector_mask_max_int32, 'ceil', self.input_vconv_ub[ceil_offset],
                                    self.input_x_ub[ceil_offset], repeat_time, 1, 1, 8, 4)
            self.tik_instance.vconv(self.vector_mask_max_int32, '', self.input_x_ub[ceil_offset],
                                    self.input_vconv_ub[ceil_offset], repeat_time, 1, 1, 4, 8, 1.0)
        left_num = move_num % self.vector_mask_max_int32

        if left_num > 0:
            ceil_offset = move_num // (self.vector_mask_max_int32) * (self.vector_mask_max_int32)
            self.tik_instance.vconv(left_num, 'ceil', self.input_vconv_ub[ceil_offset],
                                    self.input_x_ub[ceil_offset], 1, 1, 1, 8, 4)
            self.tik_instance.vconv(left_num, '', self.input_x_ub[ceil_offset],
                                    self.input_vconv_ub[ceil_offset], 1, 1, 1, 4, 8, 1.0)


def tik_ceil(input0, output0, kernel_name="Ceil"):
    """
    :param input0: dict, shape and dtype of input0
    :param output0: dict shape and dtype of output
    :param kernel_name: kernel name, default value is "Ceil"
    """

    obj = Ceil(input0, output0, kernel_name)
    obj.ceil_compute()
