#!/usr/bin/python
# -*- coding: utf-8 -*-
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
rnn_gen_mask
"""

from te import tik
from te.utils import para_check
from te.platform.fusion_manager import fusion_manager


@fusion_manager.register("rnn_gen_mask")
class RnnGenMask(object):
    """class for rnn_gen_mask"""

    def __init__(self, seq_length, seq_mask, num_step, hidden_size, kernel_name):
        """__init__"""
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.available_aicore_num = tik.Dprofile().get_aicore_num()

        self.kernel_name = kernel_name

        self.dtype = seq_length.get("dtype").lower()
        self.shape = seq_length.get("shape")

        self.dtype_out = seq_mask.get("dtype").lower()
        self.shape_out = seq_mask.get("shape")

        self.num_step = num_step
        self.batch_size = int(self.shape[0])
        self.hidden_size = hidden_size

        self.block = 8
        self.hidden_size_block = (self.hidden_size + self.block - 1) // self.block * self.block
        self.repeat = self.hidden_size_block // self.block

        self.rounds = self.num_step * self.batch_size
        self.thread_num = 2 if self.rounds > 1 else 1

        self.used_aicore_num = self.available_aicore_num if self.rounds > self.available_aicore_num else self.rounds
        self.batch_num_per_aicore = self.rounds // self.used_aicore_num
        self.batch_tail = self.rounds % self.used_aicore_num

        self.seq_length_gm = self.tik_instance.Tensor(self.dtype, self.shape, name="seq_length_gm", scope=tik.scope_gm)
        self.seq_mask_gm = self.tik_instance.Tensor(self.dtype_out, self.shape_out, name="seq_mask_gm",
                                                    scope=tik.scope_gm)
        self.temp_gm = self.tik_instance.Tensor(self.dtype_out,
                                                [self.num_step, self.batch_size, self.hidden_size_block], 
                                                name="temp_gm", scope=tik.scope_gm, is_workspace=True)

    def para_rule(self):
        """para_rule"""
        para_check.check_dtype(self.dtype, ("int32"), param_name="seq_length")
        para_check.check_dtype(self.dtype_out, ("int32"), param_name="seq_mask")

        para_check.check_shape(self.shape, param_name="seq_length")
        para_check.check_shape(self.shape_out, param_name="seq_mask")

        para_check.check_kernel_name(self.kernel_name)
        
        if self.repeat > 255:
            raise RuntimeError("Hidden size is too big to compute.")
        
        if self.shape_out[0] != self.num_step:
            raise RuntimeError("shape_out[0] != num_step.")
        if self.shape_out[1] != self.batch_size:
            raise RuntimeError("shape_out[1] != batch_size.")
        if self.shape_out[2] != self.hidden_size:
            raise RuntimeError("shape_out[2] != hidden_size.")

    def compute_core(self, task_id):
        """compute_core"""
        temp_ub = self.tik_instance.Tensor("int32", [self.block], name="temp_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(temp_ub, self.seq_length_gm[task_id % self.batch_size], 0, 1, 1, 0, 0)
        valid_len = self.tik_instance.Scalar("int32", init_value=temp_ub[0])
        curre_len = self.tik_instance.Scalar("int32", init_value=task_id // self.batch_size)
        
        mask_ub = self.tik_instance.Tensor(self.dtype_out, [self.hidden_size_block], name="mask_ub",
                                           scope=tik.scope_ubuf)
        with self.tik_instance.if_scope(valid_len > curre_len):
            self.tik_instance.vector_dup(self.block, mask_ub, 1, self.repeat, 1, 1)
        with self.tik_instance.else_scope():
            self.tik_instance.vector_dup(self.block, mask_ub, 0, self.repeat, 1, 1)
            
        if self.hidden_size != self.hidden_size_block:
            self.tik_instance.data_move(self.temp_gm[task_id * self.hidden_size_block], mask_ub, 0, 1, self.repeat, 0,
                                        0)
        else:
            self.tik_instance.data_move(self.seq_mask_gm[task_id * self.hidden_size], mask_ub, 0, 1, self.repeat, 0, 0)
    
    def data_tune(self):  
        """data_tune"""
        with self.tik_instance.for_range(0, self.rounds, thread_num=self.thread_num) as i:
            temp_ub = self.tik_instance.Tensor(self.dtype_out, [self.hidden_size_block], name="temp_ub",
                                               scope=tik.scope_ubuf)
            self.tik_instance.data_move(temp_ub, self.temp_gm[i * self.hidden_size_block], 0, 1, self.repeat, 0, 0)
            self.tik_instance.data_move(self.seq_mask_gm[i * self.hidden_size], temp_ub, 0, 1, self.repeat, 0, 0)                

    def rnn_gen_mask_compute(self):
        """rnn_gen_mask_compute"""
        self.para_rule()
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                self.compute_core(i + k * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.compute_core(i + self.batch_num_per_aicore * self.used_aicore_num)
        
        if self.hidden_size != self.hidden_size_block:
            self.data_tune()
        
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.seq_length_gm],
                                   outputs=[self.seq_mask_gm])
        return self.tik_instance


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def rnn_gen_mask(seq_length, seq_mask, num_step, hidden_size, kernel_name="rnn_gen_mask"):
    """
    Function: rnn_gen_mask.
    Modify : 2021-04-01

    Init base parameters
    Parameters
    ----------
    input(seq_length): dict
        data of input
    output(seq_mask): dict
        data of output
    num_step): int
    hidden_size: int
    kernel_name: str
        the name of the operator
    ----------
    """
    op_obj = RnnGenMask(seq_length, seq_mask, num_step, hidden_size, kernel_name)
    
    return op_obj.rnn_gen_mask_compute()
   