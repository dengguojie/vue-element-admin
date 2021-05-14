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
from . import constant_util as constant


class VecDupFp32():
    def __init__(self, tik_instance):
        self.tik_instance = tik_instance
        self.dst_blk_stride = 1
        self.dst_rep_stride = 8

    def set_config(self, config):
        if isinstance(config, list) and len(config) == 2:
            self.dst_blk_stride = config[0]
            self.dst_rep_stride = config[1]
        else:
            raise RuntimeError("config should be list or should configure 2 parameters")

    def vec_dup_compute(self, compute_data, dst_ub, scalar):
        ht_loop = compute_data // (constant.MASK64 * 255)
        if ht_loop > 0:
            with self.tik_instance.for_range(0, ht_loop) as index:
                ht_offset = index * constant.MASK64 * 255
                self.tik_instance.vector_dup(constant.MASK64, dst_ub[ht_offset], scalar, 255,
                                             self.dst_blk_stride, self.dst_rep_stride)
        repeat_times = (compute_data % (constant.MASK64 * 255) // constant.MASK64)
        if repeat_times > 0:
            ht_offset = constant.MASK64 * 255 * ht_loop
            self.tik_instance.vector_dup(constant.MASK64, dst_ub[ht_offset], scalar, repeat_times,
                                         self.dst_blk_stride, self.dst_rep_stride)
        left_num = compute_data % (constant.MASK64)
        if left_num > 0:
            ht_offset = compute_data // (constant.MASK64) * (constant.MASK64)
            self.tik_instance.vector_dup(left_num, dst_ub[ht_offset], scalar, 1,
                                         self.dst_blk_stride, self.dst_rep_stride)


class VecMulFp32():
    def __init__(self, tik_instance):
        self.tik_instance = tik_instance
        self.dst_blk_stride = 1
        self.src0_blk_stride = 1
        self.src1_blk_stride = 1
        self.dst_rep_stride = 8
        self.src0_rep_stride = 8
        self.src1_rep_stride = 8

    def set_config(self, config):
        if isinstance(config, list) and len(config) == 6:
            self.dst_blk_stride = config[0]
            self.src0_blk_stride = config[1]
            self.src1_blk_stride = config[2]
            self.dst_rep_stride = config[3]
            self.src1_rep_stride = config[4]
            self.src1_rep_stride = config[5]
        else:
            raise RuntimeError("config should be list or should configure 6 parameters")

    def vec_mul_compute(self, compute_data, dst_ub, src0, scr1):

        ht_loop = compute_data // (constant.MASK64 * 255)  # 每个vector的指令repeat次数最大为255
        if ht_loop > 0:
            with self.tik_instance.for_range(0, ht_loop) as index:
                ht_offset = index * constant.MASK64 * 255
                self.tik_instance.vmul(constant.MASK64, dst_ub[ht_offset], src0[ht_offset], scr1[ht_offset], 255,
                                       self.dst_blk_stride, self.src0_blk_stride, self.src1_blk_stride,
                                       self.dst_rep_stride, self.src1_rep_stride, self.src1_rep_stride)

        repeat_times = (compute_data % (constant.MASK64 * 255) // constant.MASK64)
        if repeat_times > 0:
            ht_offset = constant.MASK64 * 255 * ht_loop
            self.tik_instance.vmul(constant.MASK64, dst_ub[ht_offset], src0[ht_offset], scr1[ht_offset], repeat_times,
                                   self.dst_blk_stride, self.src0_blk_stride, self.src1_blk_stride,
                                   self.dst_rep_stride, self.src1_rep_stride, self.src1_rep_stride)

        left_num = compute_data % (constant.MASK64)
        if left_num > 0:
            ht_offset = compute_data // (constant.MASK64) * (constant.MASK64)
            self.tik_instance.vmul(left_num, dst_ub[ht_offset], src0[ht_offset], scr1[ht_offset], 1,
                                   self.dst_blk_stride, self.src0_blk_stride, self.src1_blk_stride,
                                   self.dst_rep_stride, self.src1_rep_stride, self.src1_rep_stride)
