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
from . import constant_util as constant
from . import tanh as tanh
from . import hard_sigmoid as sigm
from . import gate_compute as gate
from . import instr

BLOCK_NUM = 8  # The number of blocks processed by the ai core at a time
UB_SIZE = 240 * 1024  # size of 310 ai core ub buffer
AI_COER_NUM = 2


def ceil_div(value, factor):
    if value % factor == 0:
        return value // factor
    else:
        return value // factor + 1


def ceil_align(value, factor):
    if value % factor == 0:
        return value // factor * factor
    else:
        return (value // factor + 1) * factor


class FuseGru():
    """
    activation='tanh',
    recurrent_activation='hard_sigmoid',
    Parameters
    ----------
    input_tensor:
        gru_input: dict shape and dtype of input, shape is (batch, timesteps, feature_num)
        gru_weight: dict shape and dtype of gru_weight, shape is (feature_num + units + 1, units)
    attr:
        units: unit num
        fusegru: compute forword and backword gru, then do concat (axis=2)
        gobackwords: bool, switch forword gru or backword gru
    kernel_name : kernel name, default value is "FuseGru"
    output_tensor:
        output_z : dict shape and dtype of output, shape is (batch, timesteps, units)
    Returns
    -------
    None
    """

    def __init__(self, gru_input, gru_weight, units, fusegru, gobackwords,
                 kernel_name="FuseGru"):
        self.input_shape = gru_input.get("shape")
        self.dtype = gru_input.get("dtype")
        self.gru_weight_shape = gru_weight.get("shape")
        self.kernel_name = kernel_name
        self.rununits = units
        self.gobackwords = gobackwords
        self.fuse = fusegru
        self.batch = self.input_shape[0]
        self.timesteps = self.input_shape[1]
        self.feature_num = self.input_shape[2]
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.aicore_num = AI_COER_NUM
        ub_size_bytes = UB_SIZE

        if ((self.batch * self.rununits * 45 + self.batch * self.feature_num) * 2 > ub_size_bytes):
            raise RuntimeError("the shape of input is too big, this version can not support")
        if (self.batch != 1):
            raise RuntimeError("not support multibatch")
        if (self.feature_num != 256):
            raise RuntimeError("only support feature_num = 256")
        if (self.rununits != 128):
            raise RuntimeError("only support rununits = 128")
        if (self.timesteps != 46 and self.timesteps != 59):
            raise RuntimeError("only support timesteps = 46 or 59")

        # Each block contains the number of data of type float32
        self.one_blk_fp32 = constant.BLOCK_SIZE // constant.DATA_SIZE_FOUR
        # Each block contains the number of data of type float16
        self.one_blk_fp16 = constant.BLOCK_SIZE // constant.DATA_SIZE_TWO

    def gru(self):
        input_gm = self.tik_instance.Tensor(self.dtype, self.input_shape, name="input_gm", scope=tik.scope_gm)
        gru_weight_gm = self.tik_instance.Tensor("float16", self.gru_weight_shape, name="gru_weight_gm",
                                                 scope=tik.scope_gm)
        if self.fuse:
            output_gm = self.tik_instance.Tensor(self.dtype, (self.batch * self.timesteps * self.rununits * 2,),
                                                 name="output_gm", scope=tik.scope_gm)
            with self.tik_instance.for_range(0, AI_COER_NUM, block_num=AI_COER_NUM) as index:
                with self.tik_instance.if_scope(index % 2 == 0):
                    self._forword_gru(input_gm, gru_weight_gm, output_gm, index=index)
                with self.tik_instance.else_scope():
                    self._backword_gru(input_gm, gru_weight_gm, output_gm, index=index)
        else:
            output_gm = self.tik_instance.Tensor(self.dtype, (self.batch * self.timesteps * self.rununits,),
                                                 name="output_gm", scope=tik.scope_gm)
            if self.gobackwords:
                self._backword_gru(input_gm, gru_weight_gm, output_gm)
            else:
                self._forword_gru(input_gm, gru_weight_gm, output_gm)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[input_gm, gru_weight_gm],
                                   outputs=[output_gm], enable_l2=True)
        return self.tik_instance

    def _forword_gru(self, input_gm, gru_weight_gm, output_gm, index=0):

        feature_num_block_float16 = ceil_div(self.feature_num, self.one_blk_fp16)
        units_block_float16 = ceil_div(self.rununits, self.one_blk_fp16)

        # state: ht-1
        ht_1_ub_shape = (ceil_align(self.batch * self.rununits, self.one_blk_fp32),)
        ht_1_ub = self.tik_instance.Tensor('float32', ht_1_ub_shape,
                                           name="ht_1b_ub", scope=tik.scope_ubuf)
        # Input data is 3D:（batch, timesteps, feature_num）
        with self.tik_instance.for_range(0, self.timesteps) as t_steps:
            ht_1_ub_fp16 = self.tik_instance.Tensor('float16',
                                                    (ceil_align(self.batch * self.rununits, 16),),
                                                    name="ht_1_ub_fp16", scope=tik.scope_ubuf)

            xt_ub_fp16 = self.tik_instance.Tensor('float16', (ceil_align(self.batch * self.feature_num, 16),),
                                                  name="xt_ub_fp16", scope=tik.scope_ubuf)  # shape(batch, feature_num)

            xt_ub = self.tik_instance.Tensor('float32',
                                             (ceil_align(self.batch * self.feature_num, self.one_blk_fp32),),
                                             name="xt_ub", scope=tik.scope_ubuf)  # shape(batch, feature_num)
            with self.tik_instance.if_scope(t_steps == 0):
                instr.VecDupFp32(self.tik_instance).vec_dup_compute(self.batch * self.rununits, ht_1_ub, 0.0)

            self.tik_instance.data_move(xt_ub_fp16, input_gm[t_steps * self.feature_num], 0, self.batch,
                                        feature_num_block_float16, self.timesteps * self.feature_num // 16 - 1, 0)

            self.tik_instance.vconv(64, '', xt_ub, xt_ub_fp16, 4, 1, 1, 8, 4)

            self._gru_cell(xt_ub, ht_1_ub, gru_weight_gm, index)

            self.tik_instance.vconv(64, '', ht_1_ub_fp16, ht_1_ub, 2, 1, 1, 4, 8)

            if self.fuse:
                self.tik_instance.data_move(output_gm[t_steps * self.rununits * 2], ht_1_ub_fp16, 0, self.batch,
                                            units_block_float16, self.timesteps * self.rununits * 2 // 16 - 1, 0)
            else:
                self.tik_instance.data_move(output_gm[t_steps * self.rununits], ht_1_ub_fp16, 0, self.batch,
                                            units_block_float16, self.timesteps * self.rununits // 16 - 1, 0)

    def _backword_gru(self, input_gm, gru_weight_gm, output_gm, index=0):

        feature_num_block_float16 = ceil_div(self.feature_num, self.one_blk_fp16)
        units_block_float16 = ceil_div(self.rununits, self.one_blk_fp16)
        # state: ht-1
        ht_1_ub_shape = (ceil_align(self.batch * self.rununits, self.one_blk_fp32),)
        ht_1_ub = self.tik_instance.Tensor('float32', ht_1_ub_shape,
                                           name="ht_1b_ub", scope=tik.scope_ubuf)
        # Input data is 3D:（batch, timesteps, feature_num）
        with self.tik_instance.for_range(0, self.timesteps) as t_steps:
            ht_1_ub_fp16 = self.tik_instance.Tensor('float16',
                                                    (ceil_align(self.batch * self.rununits, 16),),
                                                    name="ht_1_ub_fp16", scope=tik.scope_ubuf)
            xt_ub_fp16 = self.tik_instance.Tensor('float16',
                                                  (ceil_align(self.batch * self.feature_num, 16),),
                                                  name="xt_ub_fp16", scope=tik.scope_ubuf)  # shape(batch, feature_num)

            xt_ub_shape = (ceil_align(self.batch * self.feature_num, self.one_blk_fp32),)
            xt_ub = self.tik_instance.Tensor('float32', xt_ub_shape, name="xt_b_ub",
                                             scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(t_steps == 0):
                instr.VecDupFp32(self.tik_instance).vec_dup_compute(self.batch * self.rununits, ht_1_ub, 0.0)
            back_ts = self.timesteps - t_steps - 1  # reverse loop
            # support feature_num // 16 ==0
            self.tik_instance.data_move(xt_ub_fp16, input_gm[back_ts * self.feature_num], 0, self.batch,
                                        feature_num_block_float16, self.timesteps * self.feature_num // 16 - 1, 0)
            self.tik_instance.vconv(64, '', xt_ub, xt_ub_fp16, 4, 1, 1, 8, 4)

            self._gru_cell(xt_ub, ht_1_ub, gru_weight_gm, index)

            self.tik_instance.vconv(64, '', ht_1_ub_fp16, ht_1_ub, 2, 1, 1, 4, 8)

            if self.fuse:
                self.tik_instance.data_move(output_gm[self.rununits + t_steps * self.rununits * 2], ht_1_ub_fp16, 0,
                                            self.batch, units_block_float16,
                                            self.timesteps * self.rununits * 2 // 16 - 1, 0)
            else:
                self.tik_instance.data_move(output_gm[t_steps * self.rununits], ht_1_ub_fp16, 0,
                                            self.batch, units_block_float16, self.timesteps * self.rununits // 16 - 1,
                                            0)

    def _gru_cell(self, xt_ub, ht_1_ub, weight_gm, index):
        """
        forword_weight data layout: (3 ,(feature + rununits + 1) * rununits)
        backword_weight data layout: (3 , (feature + rununits + 1) * rununits)
        weight_gm if fusegru: [[forword_weight],
                             [backword_weight]]
        forword_weight or backword_weight:
        --------------------------------------
        zt weight:
        kernel [0, 0 : feature * rununits]
        recurrent_weight[0, feature * rununit : (feature + rununit)  * rununit]
        bias [0, (feature + rununit) * rununit : (feature + rununit + 1)  * rununit]
        --------------------------------------
        --------------------------------------
        rt weight:
        kernel [1, 0 : feature * rununits]
        recurrent_weight[1, feature * rununit : (feature + rununit)  * rununit]
        bias [1, (feature + rununit) * rununit : (feature + rununit + 1)  * rununit]
        --------------------------------------
        --------------------------------------
        ~ht weight:
            kernel [2, 0 : feature * rununits]
            recurrent_weight[2, feature * rununit : (feature + rununit)  * rununit]
            bias [2, (feature + rununit) * rununit : (feature + rununit + 1)  * rununit]
            --------------------------------------
        """
        move_offset = 0
        move_num = self.batch * self.rununits
        # reset gate rt is defined as: σ(XtWxr+Ht−1Whr+br)
        rt_ub = self.tik_instance.Tensor('float32', (ceil_align(self.batch * self.rununits, self.one_blk_fp32),),
                                         name="rt_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            weight_index_start = (self.rununits + self.feature_num + 1) * (1 + self.fuse * 3 * index) * self.rununits
            weight_index_end = (self.rununits + self.feature_num + 1) * (2 + self.fuse * 3 * index) * self.rununits
            gate.gate_compute(self.tik_instance, xt_ub, ht_1_ub, weight_gm[weight_index_start: weight_index_end],
                              rt_ub, self.batch, self.feature_num, self.rununits)

        with self.tik_instance.new_stmt_scope():
            sigmoid_call = sigm.HardSigmoid({'shape': (self.batch, self.rununits), "dtype": "float32"})
            # call sigmoid and set the available size of ub
            sigmoid_call.call_as_fuc(self.tik_instance, rt_ub,
                                     ceil_div(self.batch * self.rununits, self.one_blk_fp32) * 32 * 3)
            sigmoid_call.sigmoid_compute_each_core(move_offset, move_num)

        # update gate zt is defined as: σ(XtWxz+Ht−1Whz+bz)
        zt_ub = self.tik_instance.Tensor('float32', (ceil_align(self.batch * self.rununits, self.one_blk_fp32),),
                                         name="zt_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            weight_index2_start = (self.rununits + self.feature_num + 1) * (0 + self.fuse * 3 * index) * self.rununits
            weight_index2_end = (self.rununits + self.feature_num + 1) * (1 + self.fuse * 3 * index) * self.rununits
            gate.gate_compute(self.tik_instance, xt_ub, ht_1_ub, weight_gm[weight_index2_start: weight_index2_end],
                              zt_ub, self.batch, self.feature_num, self.rununits)
        with self.tik_instance.new_stmt_scope():
            sigmoid_call = sigm.HardSigmoid({'shape': (self.batch, self.rununits), "dtype": "float32"})
            # call sigmoid and set the available size of ub
            sigmoid_call.call_as_fuc(self.tik_instance, zt_ub,
                                     ceil_div(self.batch * self.rununits, self.one_blk_fp32) * 32 * 3)
            sigmoid_call.sigmoid_compute_each_core(move_offset, move_num)

        # hidden layer ~ht is defined as: tanh(XtWxh+(Rt⊙Ht−1)Whh+bh)
        hh_ub = self.tik_instance.Tensor('float32', (ceil_align(self.batch * self.rununits, self.one_blk_fp32),),
                                         name="hh_ub", scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            # Rt⊙Ht−1
            instr.VecMulFp32(self.tik_instance).vec_mul_compute(move_num, hh_ub, rt_ub, ht_1_ub)
            weight_index3_start = (self.rununits + self.feature_num + 1) * (2 + self.fuse * 3 * index) * self.rununits
            weight_index3_end = (self.rununits + self.feature_num + 1) * (3 + self.fuse * 3 * index) * self.rununits
            gate.gate_compute(self.tik_instance, xt_ub, hh_ub, weight_gm[weight_index3_start: weight_index3_end], hh_ub,
                              self.batch, self.feature_num, self.rununits)

        with self.tik_instance.new_stmt_scope():
            tanh_call = tanh.Tanh({'shape': (self.batch, self.rununits), "dtype": "float32"})
            tanh_call.call_as_fuc(self.tik_instance, hh_ub,
                                  ceil_div(self.batch * self.rununits, self.one_blk_fp32) * 32 * 12)
            tanh_call.tanh_compute_each_core(move_offset, move_num)

        # reuse the space of rt_ub
        # result Ht is defined as: Zt⊙Ht−1+(1−Zt)⊙~Ht -> Zt⊙Ht−1+ ~Ht−Zt⊙~Ht
        self._ht_compute(move_num, rt_ub, zt_ub, ht_1_ub, hh_ub)

    def _ht_compute(self, move_num, rt_ub, zt_ub, ht_1_ub, hh_ub):
        ht_loop = move_num // (constant.MASK64 * 255)
        if ht_loop > 0:
            with self.tik_instance.for_range(0, ht_loop) as index:
                ht_offset = index * constant.MASK64 * 255
                self._ht_compute_each_vec_loop(constant.MASK64, ht_offset, 255, rt_ub, zt_ub, ht_1_ub, hh_ub)

        repeat_times = (move_num % (constant.MASK64 * 255) // constant.MASK64)
        if repeat_times > 0:
            ht_offset = constant.MASK64 * 255 * ht_loop
            self._ht_compute_each_vec_loop(constant.MASK64, ht_offset, repeat_times, rt_ub, zt_ub, ht_1_ub, hh_ub)

        left_num = move_num % constant.MASK64
        if left_num > 0:
            ht_offset = move_num // constant.MASK64 * constant.MASK64
            self._ht_compute_each_vec_loop(left_num, ht_offset, 1, rt_ub, zt_ub, ht_1_ub, hh_ub)

    def _ht_compute_each_vec_loop(self, mask, ht_offset, repeat_times, rt_ub, zt_ub, ht_1_ub, hh_ub):
        # Zt⊙Ht−1
        self.tik_instance.vmul(mask, rt_ub[ht_offset], zt_ub[ht_offset], ht_1_ub[ht_offset],
                               repeat_times, 1, 1, 1, 8, 8, 8)
        # Zt⊙~Ht
        self.tik_instance.vmul(mask, zt_ub[ht_offset], zt_ub[ht_offset], hh_ub[ht_offset], repeat_times,
                               1, 1, 1, 8, 8, 8)

        # ~Ht − Zt⊙~Ht
        self.tik_instance.vmuls(mask, ht_1_ub[ht_offset], zt_ub[ht_offset], -1.0, repeat_times,
                                1, 1, 8, 8)

        self.tik_instance.vadd(mask, ht_1_ub[ht_offset], hh_ub[ht_offset], ht_1_ub[ht_offset],
                               repeat_times, 1, 1, 1, 8, 8, 8)

        # Zt⊙~Ht + ~Ht − Zt⊙~Ht
        self.tik_instance.vadd(mask, ht_1_ub[ht_offset], ht_1_ub[ht_offset], rt_ub[ht_offset],
                               repeat_times, 1, 1, 1, 8, 8, 8)


def tik_fusegru(gru_input, input_weight, output, units, fusegru=False, gobackwords=False, kernel_name="Gru"):
    gru_instance = FuseGru(gru_input, input_weight, units, fusegru, gobackwords, kernel_name)
    gru_instance.gru()
