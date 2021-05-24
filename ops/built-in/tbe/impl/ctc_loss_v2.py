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
"""
ctc_loss_v2
"""

# pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
from te import tik
from topi.cce import util
from te.utils import para_check
import te.platform as tbe_platform

BLOCK = 8
MIN = -3.4e38


@tbe_platform.fusion_manager.fusion_manager.register("ctc_loss_v2")
class CTCLossV2():
    """CTCLossV2"""
    def __init__(self, log_probs, targets, blank, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        params = self.paras_check(log_probs, targets, kernel_name)
        self.blank = blank
        self.T = params[0]
        self.N = params[1]
        self.C = params[2]
        self.S = params[3]
        self.S_BLOCK = (self.S + BLOCK - 1) // BLOCK * BLOCK
        self.C_BLOCK = (self.C + BLOCK - 1) // BLOCK * BLOCK

        self.output_size = 2 * self.S + 1
        self.output_size_up = (self.output_size + BLOCK - 1) // BLOCK * BLOCK
        self.alpha_size = self.T * self.output_size

        if self.output_size < BLOCK:
            raise RuntimeError("Unexcepted case: 2 * S + 1 < 8.")

        self.log_probs = self.tik_instance.Tensor("float32", [self.T, self.N, self.C], name="log_probs",
                                                  scope=tik.scope_gm)
        self.targets = self.tik_instance.Tensor("int32", [self.N, self.S], name="targets", scope=tik.scope_gm)
        self.input_lengths = self.tik_instance.Tensor("int32", [self.N], name="input_lengths", scope=tik.scope_gm)
        self.target_lengths = self.tik_instance.Tensor("int32", [self.N], name="target_lengths", scope=tik.scope_gm)

        self.log_alpha = self.tik_instance.Tensor("float32", [self.N, self.T, self.output_size], name="log_alpha",
                                                  scope=tik.scope_gm)
        self.log_alpha_ = self.tik_instance.Tensor("float32", [self.N, BLOCK], name="log_alpha_", scope=tik.scope_gm,
                                                   is_workspace=True)

        self.neg_log_likelihood = self.tik_instance.Tensor("float32", [self.N], name="neg_log_likelihood",
                                                           scope=tik.scope_gm)
        self.neg_log_likelihood_ = self.tik_instance.Tensor("float32", [self.N, BLOCK], name="neg_log_likelihood_",
                                                            scope=tik.scope_gm, is_workspace=True)

        self.available_aicore_num = tik.Dprofile().get_aicore_num()
        self.used_aicore_num = self.available_aicore_num if self.N > self.available_aicore_num else self.N
        self.batch_num_per_aicore = self.N // self.used_aicore_num
        self.batch_tail = self.N % self.used_aicore_num

    def paras_check(self, log_probs, targets, kernel_name):
        """paras_check"""
        shape_log_probs = log_probs.get("shape")
        dtype_float = log_probs.get("dtype").lower()
        util.check_shape_rule(shape_log_probs)
        util.check_dtype_rule(dtype_float, ("float32"))

        shape_targets = targets.get("shape")
        dtype_int = targets.get("dtype").lower()
        util.check_shape_rule(shape_targets)
        util.check_dtype_rule(dtype_int, ("int32"))

        util.check_kernel_name(kernel_name)

        return [shape_log_probs[0], shape_log_probs[1], shape_log_probs[2], shape_targets[1]]

    def ctc_loss_compute(self):
        """ctc_loss_compute"""
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.ctc_loss_compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.ctc_loss_compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)

        self.move_out()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.log_probs, self.targets, self.input_lengths, self.target_lengths],
                                   outputs=[self.neg_log_likelihood, self.log_alpha])

        return self.tik_instance

    def ctc_loss_compute_core(self, task_idx):
        """ctc_loss_compute_core"""
        targets_ub = self.tik_instance.Tensor("int32", [self.S_BLOCK], name="targets_ub", scope=tik.scope_ubuf)
        input_length_ub = self.tik_instance.Tensor("int32", [BLOCK], name="input_length_ub", scope=tik.scope_ubuf)
        target_length_ub = self.tik_instance.Tensor("int32", [BLOCK], name="target_length_ub", scope=tik.scope_ubuf)

        self.tik_instance.data_move(targets_ub, self.targets[task_idx * self.S], 0, 1, self.S_BLOCK // BLOCK, 0, 0)
        self.tik_instance.data_move(input_length_ub, self.input_lengths[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(target_length_ub, self.target_lengths[task_idx], 0, 1, 1, 0, 0)

        T_i = self.tik_instance.Scalar("int32", init_value=input_length_ub[0])
        S_i = self.tik_instance.Scalar("int32", init_value=target_length_ub[0])

        repeats, s_inc, e_inc = self.count_trace(S_i, targets_ub)

        start = self.tik_instance.Scalar("int32")
        start_loop = self.tik_instance.Scalar("int32")
        end = self.tik_instance.Scalar("int32")
        remain = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        offset = self.tik_instance.Scalar("int32")
        current_target = self.tik_instance.Scalar("int32")
        next_target = self.tik_instance.Scalar("int32")
        tmp = self.tik_instance.Scalar("int32")
        a_tmp = self.tik_instance.Scalar("float32")
        b_tmp = self.tik_instance.Scalar("float32")
        min_float = self.tik_instance.Scalar("float32", init_value=MIN)

        log_ub = self.tik_instance.Tensor("float32", [self.T, BLOCK], name="log_ub", scope=tik.scope_ubuf)
        exp_ub = self.tik_instance.Tensor("float32", [self.T, BLOCK], name="exp_ub", scope=tik.scope_ubuf)
        add_ub = self.tik_instance.Tensor("float32", [self.T, BLOCK], name="add_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor("float32", [self.T, BLOCK], name="tmp_ub", scope=tik.scope_ubuf)

        work_tensor_ub = self.tik_instance.Tensor("float32", [BLOCK], name="work_tensor_ub", scope=tik.scope_ubuf)

        self.alpha_neg_log_likelihood_update(task_idx, T_i, S_i, exp_ub, log_ub, add_ub, tmp_ub, work_tensor_ub, a_tmp,
                                             b_tmp, start, start_loop, end, remain, repeat_times, offset,
                                             current_target, next_target, tmp, min_float, repeats, s_inc, e_inc,
                                             targets_ub)

    def alpha_neg_log_likelihood_update(self, task_idx, T_i, S_i, exp_ub, log_ub, add_ub, tmp_ub, work_tensor_ub, a_tmp,
                                        b_tmp, start, start_loop, end, remain, repeat_times, offset, current_target,
                                        next_target, tmp, min_float, repeats, s_inc, e_inc, targets_ub):
        """alpha_neg_log_likelihood_update"""
        log_probs_ub = self.tik_instance.Tensor("int32", [self.C_BLOCK], name="input_length_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(log_probs_ub[0], self.log_probs[self.C * task_idx], 0, 1, self.C_BLOCK // BLOCK, 0,
                                    0)

        output_dst = self.tik_instance.Scalar("int32", init_value=0)
        output_src = self.tik_instance.Scalar("int32", init_value=self.output_size_up)

        log_alpha_ub = self.tik_instance.Tensor("float32", [2, self.output_size_up], name="log_alpha_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(BLOCK, log_alpha_ub[output_dst], MIN, self.output_size_up // BLOCK, 1, 1)

        log_alpha_ub[output_dst].set_as(log_probs_ub[self.blank])
        current_target.set_as(targets_ub[0])
        log_alpha_ub[output_dst + 1].set_as(log_probs_ub[current_target])

        with self.tik_instance.if_scope(repeats < T_i - S_i):
            start.set_as(0)
        with self.tik_instance.else_scope():
            start.set_as(1)
        end.set_as(2)

        with self.tik_instance.for_range(1, T_i) as t:
            self.tik_instance.data_move(log_probs_ub[0], self.log_probs[self.C * task_idx + self.N * self.C * t], 0, 1,
                                        self.C_BLOCK // BLOCK, 0, 0)
            self.tik_instance.vector_dup(BLOCK, log_alpha_ub[output_src], MIN, self.output_size_up // BLOCK, 1, 1)

            remain.set_as(S_i + repeats - T_i + t)
            with self.tik_instance.if_scope(remain >= 0):
                tmp.set_as(s_inc[remain])
                start.set_as(start + tmp)
            start_loop.set_as(start)

            with self.tik_instance.if_scope(t <= S_i + repeats):
                tmp.set_as(e_inc[t - 1])
                end.set_as(end + tmp)

            with self.tik_instance.if_scope(start_loop == 0):
                a_tmp.set_as(log_alpha_ub[output_dst])
                b_tmp.set_as(log_probs_ub[self.blank])

                log_alpha_ub[output_src].set_as(a_tmp + b_tmp)
                start_loop.set_as(1)

            with self.tik_instance.for_range(start_loop, end) as s:
                with self.tik_instance.if_scope(s % 2 == 0):
                    current_target.set_as(self.blank)
                with self.tik_instance.else_scope():
                    current_target.set_as(targets_ub[s // 2])

                offset.set_as((s - start_loop) * BLOCK)

                tmp_ub[offset].set_as(log_probs_ub[current_target])
                log_ub[offset].set_as(log_alpha_ub[output_dst + s])
                log_ub[offset + 1].set_as(log_alpha_ub[output_dst + s - 1])

                with self.tik_instance.if_scope(tik.all((s % 2 != 0), (s != 1))):
                    next_target.set_as(targets_ub[s // 2 - 1])
                    with self.tik_instance.if_scope(current_target != next_target):
                        log_ub[offset + 2].set_as(log_alpha_ub[output_dst + s - 2])
                    with self.tik_instance.else_scope():
                        log_ub[offset + 2].set_as(min_float)
                with self.tik_instance.else_scope():
                    log_ub[offset + 2].set_as(min_float)

            repeat_times.set_as(end - start_loop)
            self.tik_instance.vec_exp(3, exp_ub, log_ub, repeat_times, 1, 1)
            with self.tik_instance.for_range(0, repeat_times) as s:
                self.tik_instance.vec_reduce_add(3, add_ub[s * BLOCK], exp_ub[s * BLOCK], work_tensor_ub, 1, 1)
            self.tik_instance.vln(1, log_ub, add_ub, repeat_times, 1, 1, 1, 1)
            self.tik_instance.vec_add(1, add_ub, tmp_ub, log_ub, repeat_times, 1, 1, 1)

            with self.tik_instance.for_range(start_loop, end) as s:
                offset.set_as((s - start_loop) * BLOCK)
                log_alpha_ub[output_src + s].set_as(add_ub[offset])

            self.tik_instance.data_move(self.log_alpha[task_idx * self.alpha_size + (t - 1) * self.output_size],
                                        log_alpha_ub[output_dst], 0, 1, self.output_size_up // BLOCK, 0, 0)

            output_src.set_as(output_dst)
            output_dst.set_as(self.output_size_up - output_src)

        with self.tik_instance.if_scope(self.T == T_i):
            self.tik_instance.data_move(self.log_alpha[task_idx * self.alpha_size + (self.T - 1) * self.output_size],
                                        log_alpha_ub[output_dst], 0, 1, self.output_size // BLOCK, 0, 0)

            with self.tik_instance.if_scope(self.output_size % BLOCK != 0):
                self.tik_instance.data_move(self.log_alpha_[task_idx * BLOCK],
                                            log_alpha_ub[output_dst + self.output_size_up - BLOCK], 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.log_alpha[task_idx * self.alpha_size + (T_i - 1) * self.output_size],
                                        log_alpha_ub[output_dst], 0, 1, self.output_size_up // BLOCK, 0, 0)

        self.neg_log_likelihood_update(log_alpha_ub, exp_ub, log_ub, add_ub, tmp_ub, work_tensor_ub, a_tmp, output_dst,
                                       S_i, task_idx)

    def neg_log_likelihood_update(self, log_alpha_ub, exp_ub, log_ub, add_ub, tmp_ub, work_tensor_ub, a_tmp, output_dst,
                                  S_i, task_idx):
        """neg_log_likelihood_update"""
        log_ub[0].set_as(log_alpha_ub[output_dst + 2 * S_i])
        log_ub[1].set_as(log_alpha_ub[output_dst + 2 * S_i - 1])

        self.tik_instance.vec_exp(2, exp_ub, log_ub, 1, 1, 1)
        self.tik_instance.vec_reduce_add(2, add_ub, exp_ub, work_tensor_ub, 1, 1)

        self.tik_instance.vln(1, log_ub, add_ub, 1, 1, 1, 1, 1)
        a_tmp.set_as(log_ub[0])
        tmp_ub[0].set_as(-a_tmp)
        self.tik_instance.data_move(self.neg_log_likelihood_[task_idx * BLOCK], tmp_ub[0], 0, 1, 1, 0, 0)

    def count_trace(self, S_i, targets_ub):
        """count_trace"""
        s_inc = self.tik_instance.Tensor("int32", [self.output_size], name="s_inc", scope=tik.scope_ubuf)
        e_inc = self.tik_instance.Tensor("int32", [self.output_size], name="e_inc", scope=tik.scope_ubuf)

        one_step = self.tik_instance.Scalar("int32", init_value=1)
        two_step = self.tik_instance.Scalar("int32", init_value=2)

        left = self.tik_instance.Scalar("int32")
        right = self.tik_instance.Scalar("int32")

        repeats = self.tik_instance.Scalar("int32", init_value=0)
        idx_counter = self.tik_instance.Scalar("int32", init_value=1)

        s_inc[0].set_as(one_step)
        with self.tik_instance.for_range(1, S_i) as idx:
            left.set_as(targets_ub[idx - 1])
            right.set_as(targets_ub[idx])
            with self.tik_instance.if_scope(left == right):
                s_inc[idx_counter].set_as(one_step)
                e_inc[idx_counter - 1].set_as(one_step)

                s_inc[idx_counter + 1].set_as(one_step)
                e_inc[idx_counter].set_as(one_step)

                idx_counter.set_as(idx_counter + 2)
                repeats.set_as(repeats + 1)
            with self.tik_instance.else_scope():
                s_inc[idx_counter].set_as(two_step)
                e_inc[idx_counter - 1].set_as(two_step)

                idx_counter.set_as(idx_counter + 1)

        e_inc[idx_counter - 1].set_as(one_step)

        return repeats, s_inc, e_inc

    def move_out(self):
        """move_out"""
        neg_log_likelihood_ub = self.tik_instance.Tensor("int32", [BLOCK], name="input_length_ub", scope=tik.scope_ubuf)
        log_alpha_ub = self.tik_instance.Tensor("int32", [BLOCK], name="input_length_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor("int32", [BLOCK], name="input_length_ub", scope=tik.scope_ubuf)

        mask = self.tik_instance.Scalar("int32", init_value=self.output_size % BLOCK)
        with self.tik_instance.for_range(0, self.N) as task_idx:
            with self.tik_instance.if_scope(mask != 0):
                self.tik_instance.data_move(log_alpha_ub, self.log_alpha[(task_idx + 1) * self.alpha_size - mask],
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(tmp_ub, self.log_alpha_[task_idx * BLOCK], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, mask) as idx:
                    log_alpha_ub[idx].set_as(tmp_ub[idx])
                self.tik_instance.data_move(self.log_alpha[(task_idx + 1) * self.alpha_size - mask], log_alpha_ub,
                                            0, 1, 1, 0, 0)

            self.tik_instance.data_move(neg_log_likelihood_ub, self.neg_log_likelihood_[task_idx * BLOCK], 0, 1, 1, 0,
                                        0)
            self.tik_instance.data_move(self.neg_log_likelihood[task_idx], neg_log_likelihood_ub[0], 0, 1, 1, 0, 0)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def ctc_loss_v2(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank=0,
                reduction="mean", zero_infinity=False, kernel_name="ctc_loss_v2"):
    """
    Function: The Connectionist Temporal Classification loss.
    Modify : 2021-05-23

    Init base parameters
    Parameters
    ----------
    Inputs:
    Log_probs: Tensor of size (T,N,C), where T =input length, N =batch size,
               and C = number of classes (including blank).
    Targets: Tensor of size (N, S), where S= max target length.
    It represent the target sequences.
    Input_lengths: Tuple or tensor of size (N).
    It represent the lengths of the inputs.
    Target_lengths: Tuple or tensor of size (N). It represent lengths of the targets.

    Attributes:
    blank: Blank label. Default 0.
    reduction: Specifies the reduction to apply to the output. Default: 'mean'.
    zero_infinity: Whether to zero infinite losses and the associated gradients.

    Outputs:
    neg_log_likelihood: A loss value which is differentiable with respect to each input node.
    log_alpha: The probability of possible trace of input to target.
    ----------
    """
    op_obj = CTCLossV2(log_probs, targets, blank, kernel_name)

    return op_obj.ctc_loss_compute()
