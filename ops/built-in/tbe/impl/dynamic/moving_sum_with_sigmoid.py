#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
moving_sum_with_sigmoid
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    BLOCK_ALIGN = 8
    CONV_ALIGN = 16
    MAX_INT32 = 2 ** 31 - 1
    REDUCE_ALIGN = 64


# 'pylint: disable=too-many-instance-attributes
@register_operator("moving_sum_with_sigmoid")
class MovingSumWithSigmoid(object):
    """class for moving_sum_with_sigmoid"""

    # 'pylint: disable=too-many-arguments
    def __init__(self, alpha, energy, frame_size, y, window_size, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name

        self.dtype = alpha.get("dtype").lower()
        self.block = Constant.CONV_ALIGN if self.dtype == "float16" else Constant.BLOCK_ALIGN
        self.conv = True if self.dtype == "float16" else False

        self.alpha_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="alpha", scope=tik.scope_gm)
        self.energy_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="energy", scope=tik.scope_gm)
        self.frame_size_gm = self.tik_instance.Tensor("int32", [1], name="frame_size", scope=tik.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.dtype, [Constant.MAX_INT32], name="y", scope=tik.scope_gm)

        self.tiling_gm = self.tik_instance.Tensor('int32', [1], name="tiling_gm", scope=tik.scope_gm)

        self.window_size = window_size
        self.window_size_align = (self.window_size + self.block - 1) // self.block * self.block

        self.used_aicore_num = tik.Dprofile().get_aicore_num()

        self.frame_size = None
        self.task_num = None
        self.batch_num_per_aicore = None
        self.batch_tail = None

    def moving_sum_with_sigmoid_compute(self):

        frame_size_ub = self.tik_instance.Tensor("int32", [self.block], name="frame_size_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(frame_size_ub, self.frame_size_gm, 0, 1, 1, 0, 0)
        self.frame_size = self.tik_instance.Scalar("int32", init_value=frame_size_ub[0])
        self.task_num = self.tik_instance.Scalar("int32", init_value=(self.frame_size + self.block - 1) // self.block)
        self.frame_size_align = self.tik_instance.Scalar("int32", init_value=self.task_num * self.block)

        self.batch_num_per_aicore = self.tik_instance.Scalar("int32",
                                                             init_value=self.task_num // self.used_aicore_num)
        self.batch_tail = self.tik_instance.Scalar("int32", init_value=self.task_num % self.used_aicore_num)

        version = tik.Dprofile().get_product_name()
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.moving_sum_with_sigmoid_compute_core(i + j * self.used_aicore_num, version)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.moving_sum_with_sigmoid_compute_core(self.batch_num_per_aicore * self.used_aicore_num + i,
                                                          version)

        self.data_tune()

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.used_aicore_num,
            })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.alpha_gm, self.energy_gm, self.frame_size_gm],
                                   outputs=[self.y_gm], flowtable=[self.tiling_gm], config=opt_config)

        return self.tik_instance

    def moving_sum_with_sigmoid_compute_core(self, task_idx, version):
        alpha_ub = self.tik_instance.Tensor("float32", [self.frame_size_align], name="alpha_ub",
                                            scope=tik.scope_ubuf)
        energy_ub = self.tik_instance.Tensor("float32", [self.block], name="energy_ub",
                                             scope=tik.scope_ubuf)
        y_ub = self.tik_instance.Tensor("float32", [self.block], name="y_ub", scope=tik.scope_ubuf)

        if self.conv:
            alpha_ub_fp16 = self.tik_instance.Tensor("float16", [self.frame_size_align], name="alpha_ub_fp16",
                                                     scope=tik.scope_ubuf)
            energy_ub_fp16 = self.tik_instance.Tensor("float16", [self.block], name="energy_ub_fp16",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(alpha_ub_fp16, self.alpha_gm[task_idx * self.block], 0, 1,
                                        self.task_num - task_idx, 0, 0)
            self.tik_instance.data_move(energy_ub_fp16, self.energy_gm[task_idx * self.block], 0, 1, 1, 0, 0)
            self.tik_instance.vec_conv(self.block, "none", alpha_ub, alpha_ub_fp16, self.task_num - task_idx, 2, 1)
            self.tik_instance.vec_conv(self.block, "none", energy_ub, energy_ub_fp16, 1, 2, 1)
        else:
            self.tik_instance.data_move(alpha_ub, self.alpha_gm[task_idx * self.block], 0, 1,
                                        self.task_num - task_idx, 0, 0)
            self.tik_instance.data_move(energy_ub, self.energy_gm[task_idx * self.block], 0, 1, 1, 0, 0)

        ones_ub = self.tik_instance.Tensor("float32", [self.block], name="ones_ub", scope=tik.scope_ubuf)
        zero_ub = self.tik_instance.Tensor("float32", [self.block], name="zero_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor("float32", [self.block], name="tmp_ub", scope=tik.scope_ubuf)
        sigmoid_ub = self.tik_instance.Tensor("float32", [self.block], name="sigmoid_ub", scope=tik.scope_ubuf)
        sum_ub = self.tik_instance.Tensor("float32", [self.block], name="sum_ub", scope=tik.scope_ubuf)
        work_tensor_ub = self.tik_instance.Tensor("float32", [self.window_size_align], name="work_tensor_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.block, ones_ub, 1, 1, 1, 1)
        self.tik_instance.vector_dup(self.block, zero_ub, 0, 1, 1, 1)

        tmp_val = self.tik_instance.Scalar("float32")
        loop = self.tik_instance.Scalar("int32", init_value=self.frame_size - task_idx * self.block)
        # func '1 / (1 + np.exp(-x))'
        if version == "mini":
            exp_ub = self.tik_instance.Tensor("float16", [self.block], name="exp_ub", scope=tik.scope_ubuf)
            work_ub = self.tik_instance.Tensor("float16", [self.block], name="work_ub", scope=tik.scope_ubuf)
            tmp_ub_ = self.tik_instance.Tensor("float32", [self.block], name="tmp_ub_", scope=tik.scope_ubuf)

            self.tik_instance.vec_sub(self.block, tmp_ub, zero_ub, energy_ub, 1, 1, 1, 1)

            self.tik_instance.vec_conv(self.block, "none", work_ub, tmp_ub, 1, 0, 0)
            self.tik_instance.vec_exp(self.block, exp_ub, work_ub, 1, 1, 1)
            self.tik_instance.vec_conv(self.block, "none", tmp_ub, exp_ub, 1, 0, 0)

            self.tik_instance.vec_add(self.block, tmp_ub, tmp_ub, ones_ub, 1, 1, 1, 1)
            self.tik_instance.vec_rec_high_preci(self.block, sigmoid_ub, tmp_ub, work_tensor_ub, 1, 1, 1)
            block_len = self.tik_instance.Scalar("int32")
            with self.tik_instance.for_range(0, self.block) as idx:
                with self.tik_instance.if_scope(loop > self.window_size):
                    block_len.set_as(self.window_size + idx)
                with self.tik_instance.else_scope():
                    block_len.set_as(self.frame_size - task_idx * self.block)

                with self.tik_instance.if_scope(block_len > Constant.REDUCE_ALIGN):
                    with self.tik_instance.if_scope(block_len % Constant.REDUCE_ALIGN > 0):
                        self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, tmp_ub_, alpha_ub, work_tensor_ub,
                                                         block_len // Constant.REDUCE_ALIGN, 8)
                        tmp_val.set_as(tmp_ub_[0])
                        self.tik_instance.vec_reduce_add(block_len % Constant.REDUCE_ALIGN, tmp_ub_,
                                                         alpha_ub[block_len - (block_len % Constant.REDUCE_ALIGN)],
                                                         work_tensor_ub, 1, 0)
                        tmp_ub_[1].set_as(tmp_val)
                        self.tik_instance.vec_reduce_add(2, tmp_ub, tmp_ub_, work_tensor_ub, 1, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, tmp_ub, alpha_ub, work_tensor_ub,
                                                         block_len // Constant.REDUCE_ALIGN, 8)
                with self.tik_instance.else_scope():
                    self.tik_instance.vec_reduce_add(block_len, tmp_ub, alpha_ub, work_tensor_ub, 1, 1)

                sum_ub[idx].set_as(tmp_ub[0])
                alpha_ub[idx].set_as(0)
                loop.set_as(loop - 1)
        else:
            sum_val = self.tik_instance.Scalar("float32")
            exp_ub = self.tik_instance.Tensor("float32", [self.block], name="exp_ub", scope=tik.scope_ubuf)
            self.tik_instance.vec_sub(self.block, tmp_ub, zero_ub, energy_ub, 1, 1, 1, 1)
            self.tik_instance.vec_exp(self.block, exp_ub, tmp_ub, 1, 1, 1)
            self.tik_instance.vec_add(self.block, tmp_ub, exp_ub, ones_ub, 1, 1, 1, 1)
            self.tik_instance.vec_rec_high_preci(self.block, sigmoid_ub, tmp_ub, work_tensor_ub, 1, 1, 1)

            with self.tik_instance.if_scope(loop > self.window_size):
                with self.tik_instance.if_scope(self.window_size > Constant.REDUCE_ALIGN):
                    with self.tik_instance.if_scope(self.window_size % Constant.REDUCE_ALIGN > 0):
                        self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                         self.window_size // Constant.REDUCE_ALIGN, 8)
                        tmp_val.set_as(sum_ub[0])
                        self.tik_instance.vec_reduce_add(self.window_size % Constant.REDUCE_ALIGN, sum_ub,
                                                         alpha_ub[
                                                             self.window_size - (
                                                                 self.window_size % Constant.REDUCE_ALIGN)],
                                                         work_tensor_ub, 1, 0)
                        sum_val.set_as(sum_ub[0])
                        sum_ub[0].set_as(sum_val + tmp_val)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                         self.window_size // Constant.REDUCE_ALIGN, 8)
                with self.tik_instance.else_scope():
                    self.tik_instance.vec_reduce_add(self.window_size, sum_ub, alpha_ub, work_tensor_ub, 1, 1)

            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(loop > Constant.REDUCE_ALIGN):
                    with self.tik_instance.if_scope(loop % Constant.REDUCE_ALIGN > 0):
                        self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                         loop // Constant.REDUCE_ALIGN, 8)
                        tmp_val.set_as(sum_ub[0])
                        self.tik_instance.vec_reduce_add(loop % Constant.REDUCE_ALIGN, sum_ub,
                                                         alpha_ub[loop - loop % Constant.REDUCE_ALIGN],
                                                         work_tensor_ub, 1, 0)
                        sum_val.set_as(sum_ub[0])
                        sum_ub[0].set_as(sum_val + tmp_val)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_reduce_add(Constant.REDUCE_ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                         loop // Constant.REDUCE_ALIGN, 8)

                with self.tik_instance.else_scope():
                    self.tik_instance.vec_reduce_add(loop, sum_ub, alpha_ub, work_tensor_ub, 1, 1)

            sum_val.set_as(sum_ub[0])
            with self.tik_instance.for_range(1, self.block) as idx:

                tmp_val.set_as(alpha_ub[idx - 1])
                sum_val.set_as(sum_val - tmp_val)
                loop.set_as(loop - 1)
                with self.tik_instance.if_scope(loop >= self.window_size):
                    tmp_val.set_as(alpha_ub[idx + self.window_size - 1])
                    sum_val.set_as(sum_val + tmp_val)

                sum_ub[idx].set_as(sum_val)

        self.tik_instance.vec_mul(self.block, y_ub, sum_ub, sigmoid_ub, 1, 0, 0, 0)
        if self.conv:
            y_ub_fp16 = self.tik_instance.Tensor("float16", [self.block], name="y_ub_fp16", scope=tik.scope_ubuf)
            self.tik_instance.vec_conv(self.block, "none", y_ub_fp16, y_ub, 1, 1, 2)
            self.tik_instance.data_move(self.y_gm[task_idx * self.block], y_ub_fp16, 0, 1, 1, 0, 0)
        else:
            self.tik_instance.data_move(self.y_gm[task_idx * self.block], y_ub, 0, 1, 1, 0, 0)

    def data_tune(self):
        with self.tik_instance.if_scope(self.frame_size_align > self.frame_size):
            y_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="y_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(y_ub, self.y_gm[self.frame_size_align - self.block], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.frame_size_align - self.frame_size) as idx:
                y_ub[self.block - idx - 1].set_as(0)
            self.tik_instance.data_move(self.y_gm[self.frame_size_align - self.block], y_ub, 0, 1, 1, 0, 0)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def moving_sum_with_sigmoid(alpha, energy, frame_size, y, window_size,
                            kernel_name="moving_sum_with_sigmoid"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    op_obj = MovingSumWithSigmoid(alpha, energy, frame_size, y, window_size, kernel_name)

    return op_obj.moving_sum_with_sigmoid_compute()
