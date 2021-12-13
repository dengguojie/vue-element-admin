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

from impl.util.platform_adapter import tik

from impl.util.platform_adapter import para_check
from te.platform.fusion_manager import fusion_manager


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    Constant
    """
    BLOCK = 8
    MIN = -3.4e38
    REPEAT_OFFSET = 255
    LABEL_MAX = 1000


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
def check_supported(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank=0,
                    reduction="mean", zero_infinity=False, kernel_name="ctc_loss_v2"):
    """
    check the op support situation.
    Go to AICPU when the label's length is less than 4.
    """
    targets_shape = targets.get("shape")
    if targets_shape[-1] > Constant.LABEL_MAX:
        reason = "The label's length is over 1K."
        return False, reason

    return True, ""


@fusion_manager.register("ctc_loss_v2")
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
        self.S_BLOCK = (self.S + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.C_BLOCK = (self.C + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK

        self.output_size = 2 * self.S + 1
        self.output_size_up = (self.output_size + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.alpha_size = self.T * self.output_size
        self.alpha_size_up = self.T * self.output_size_up

        self.log_probs = self.tik_instance.Tensor("float32", [self.T, self.N, self.C], name="log_probs",
                                                  scope=tik.scope_gm)
        self.targets = self.tik_instance.Tensor("int32", [self.N, self.S], name="targets", scope=tik.scope_gm)
        self.input_lengths = self.tik_instance.Tensor("int32", [self.N], name="input_lengths", scope=tik.scope_gm)
        self.target_lengths = self.tik_instance.Tensor("int32", [self.N], name="target_lengths", scope=tik.scope_gm)

        self.log_alpha = self.tik_instance.Tensor("float32", [self.N, self.T, self.output_size], name="log_alpha",
                                                  scope=tik.scope_gm)
        self.log_alpha_ = self.tik_instance.Tensor("float32", [self.N, self.T, self.output_size_up], name="log_alpha_",
                                                   scope=tik.scope_gm, is_workspace=True)

        self.neg_log_likelihood = self.tik_instance.Tensor("float32", [self.N], name="neg_log_likelihood",
                                                           scope=tik.scope_gm)
        self.neg_log_likelihood_ = self.tik_instance.Tensor("float32", [self.N, Constant.BLOCK],
                                                            name="neg_log_likelihood_",
                                                            scope=tik.scope_gm, is_workspace=True)

        self.available_aicore_num = tik.Dprofile().get_aicore_num()
        self.used_aicore_num = self.available_aicore_num if self.N > self.available_aicore_num else self.N
        self.batch_num_per_aicore = self.N // self.used_aicore_num
        self.batch_tail = self.N % self.used_aicore_num

    @staticmethod
    def paras_check(log_probs, targets, kernel_name):
        """
        paras_check
        """
        shape_log_probs = log_probs.get("shape")
        dtype_float = log_probs.get("dtype").lower()
        para_check.check_shape_rule(shape_log_probs)
        para_check.check_dtype_rule(dtype_float, ("float32"))

        shape_targets = targets.get("shape")
        dtype_int = targets.get("dtype").lower()
        para_check.check_shape_rule(shape_targets)
        para_check.check_dtype_rule(dtype_int, ("int32"))

        para_check.check_kernel_name(kernel_name)

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

    # 'pylint: disable=too-many-statements
    def ctc_loss_compute_core(self, task_idx):
        """ctc_loss_compute_core"""
        targets_ub = self.tik_instance.Tensor("int32", [self.S_BLOCK], name="targets_ub", scope=tik.scope_ubuf)
        input_length_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="input_length_ub",
                                                   scope=tik.scope_ubuf)
        target_length_ub = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="target_length_ub",
                                                    scope=tik.scope_ubuf)

        self.tik_instance.data_move(targets_ub, self.targets[task_idx * self.S], 0, 1, self.S_BLOCK // Constant.BLOCK,
                                    0, 0)
        self.tik_instance.data_move(input_length_ub, self.input_lengths[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(target_length_ub, self.target_lengths[task_idx], 0, 1, 1, 0, 0)

        t_i = self.tik_instance.Scalar("int32", init_value=input_length_ub[0])
        s_i = self.tik_instance.Scalar("int32", init_value=target_length_ub[0])

        repeats, s_inc, e_inc = self.count_trace(s_i, targets_ub)

        start = self.tik_instance.Scalar("int32")
        start_loop = self.tik_instance.Scalar("int32")
        end = self.tik_instance.Scalar("int32")
        remain = self.tik_instance.Scalar("int32")
        current_target = self.tik_instance.Scalar("int32")
        next_target = self.tik_instance.Scalar("int32")
        tmp = self.tik_instance.Scalar("int32")
        min_float = self.tik_instance.Scalar("float32", init_value=Constant.MIN)

        # func: a_ub/b_ub/tmp_ub: used in exp/log/add/sub api
        a_ub = self.tik_instance.Tensor("float32", [self.output_size, Constant.BLOCK], name="a_ub",
                                        scope=tik.scope_ubuf)
        b_ub = self.tik_instance.Tensor("float32", [self.output_size, Constant.BLOCK], name="b_ub",
                                        scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor("float32", [self.output_size, Constant.BLOCK], name="tmp_ub",
                                          scope=tik.scope_ubuf)

        work_tensor_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="work_tensor_ub",
                                                  scope=tik.scope_ubuf)

        offset = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        a_tmp = self.tik_instance.Scalar("float32")
        b_tmp = self.tik_instance.Scalar("float32")
        c_tmp = self.tik_instance.Scalar("float32")
        max_tmp = self.tik_instance.Scalar("float32")
        log_probs_ub = self.tik_instance.Tensor("float32", [self.C_BLOCK], name="log_probs_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(log_probs_ub[0], self.log_probs[self.C * task_idx], 0, 1,
                                    self.C_BLOCK // Constant.BLOCK, 0, 0)

        output_dst = self.tik_instance.Scalar("int32", init_value=0)
        output_src = self.tik_instance.Scalar("int32", init_value=self.output_size_up)

        log_alpha_ub = self.tik_instance.Tensor("float32", [2, self.output_size_up], name="log_alpha_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(Constant.BLOCK, log_alpha_ub[output_dst], Constant.MIN,
                                     self.output_size_up // Constant.BLOCK, 1, 1)

        lamax_ub = self.tik_instance.Tensor("float32", [self.output_size], name="lamax_ub", scope=tik.scope_ubuf)

        log_alpha_ub[output_dst].set_as(log_probs_ub[self.blank])
        current_target.set_as(targets_ub[0])
        log_alpha_ub[output_dst + 1].set_as(log_probs_ub[current_target])

        with self.tik_instance.if_scope(repeats < t_i - s_i):
            start.set_as(0)
        with self.tik_instance.else_scope():
            start.set_as(1)
        end.set_as(2)

        with self.tik_instance.for_range(1, t_i) as t:
            self.tik_instance.data_move(log_probs_ub[0], self.log_probs[self.C * task_idx + self.N * self.C * t], 0, 1,
                                        self.C_BLOCK // Constant.BLOCK, 0, 0)
            self.tik_instance.vector_dup(Constant.BLOCK, log_alpha_ub[output_src], Constant.MIN,
                                         self.output_size_up // Constant.BLOCK, 1, 1)

            remain.set_as(s_i + repeats - t_i + t)
            with self.tik_instance.if_scope(remain >= 0):
                tmp.set_as(s_inc[remain])
                start.set_as(start + tmp)
            start_loop.set_as(start)

            with self.tik_instance.if_scope(t <= s_i + repeats):
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

                offset.set_as((s - start_loop) * Constant.BLOCK)

                tmp_ub[offset].set_as(log_probs_ub[current_target])
                a_ub[offset].set_as(log_alpha_ub[output_dst + s])
                a_ub[offset + 1].set_as(log_alpha_ub[output_dst + s - 1])

                with self.tik_instance.if_scope(tik.all((s % 2 != 0), (s != 1))):
                    next_target.set_as(targets_ub[s // 2 - 1])
                    with self.tik_instance.if_scope(current_target != next_target):
                        a_ub[offset + 2].set_as(log_alpha_ub[output_dst + s - 2])
                    with self.tik_instance.else_scope():
                        a_ub[offset + 2].set_as(min_float)
                with self.tik_instance.else_scope():
                    a_ub[offset + 2].set_as(min_float)

                a_tmp.set_as(a_ub[offset])
                b_tmp.set_as(a_ub[offset + 1])
                c_tmp.set_as(a_ub[offset + 2])

                # func: get max in a_tmp/b_tmp/c_tmp
                with self.tik_instance.if_scope(a_tmp > b_tmp):
                    with self.tik_instance.if_scope(a_tmp > c_tmp):
                        lamax_ub[s - start_loop].set_as(a_tmp)
                    with self.tik_instance.else_scope():
                        lamax_ub[s - start_loop].set_as(c_tmp)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(b_tmp > c_tmp):
                        lamax_ub[s - start_loop].set_as(b_tmp)
                    with self.tik_instance.else_scope():
                        lamax_ub[s - start_loop].set_as(c_tmp)

            repeat_times.set_as(end - start_loop)

            # func: get `-max`
            with self.tik_instance.for_range(0, repeat_times) as s:
                max_tmp.set_as(lamax_ub[s])
                max_tmp.set_as(-max_tmp)
                self.tik_instance.vec_adds(3, b_ub[s * Constant.BLOCK], a_ub[s * Constant.BLOCK], max_tmp, 1, 8, 8)

            # func: `exp(a_tmp- max_tmp)  exp(b_tmp- max_tmp)  exp(b_tmp- max_tmp)`
            with self.tik_instance.if_scope(repeat_times > Constant.REPEAT_OFFSET):
                with self.tik_instance.for_range(0, repeat_times // Constant.REPEAT_OFFSET) as b:
                    self.tik_instance.vec_exp(3, a_ub[b * Constant.REPEAT_OFFSET * Constant.BLOCK],
                                              b_ub[b * Constant.REPEAT_OFFSET * Constant.BLOCK],
                                              Constant.REPEAT_OFFSET, 1, 1)
                self.tik_instance.vec_exp(
                    3, a_ub[repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET * Constant.BLOCK],
                    b_ub[repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET * Constant.BLOCK],
                    repeat_times - repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.vec_exp(3, a_ub, b_ub, repeat_times, 1, 1)

            # func: `exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)`
            with self.tik_instance.for_range(0, repeat_times) as s:
                self.tik_instance.vec_reduce_add(3, b_ub[s * Constant.BLOCK], a_ub[s * Constant.BLOCK],
                                                 work_tensor_ub, 1, 1)

            # func: `log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp))`
            with self.tik_instance.if_scope(repeat_times > Constant.REPEAT_OFFSET):
                with self.tik_instance.for_range(0, repeat_times // Constant.REPEAT_OFFSET) as b:
                    self.tik_instance.vln(1, a_ub[b * Constant.REPEAT_OFFSET * Constant.BLOCK],
                                          b_ub[b * Constant.REPEAT_OFFSET * Constant.BLOCK],
                                          Constant.REPEAT_OFFSET, 1, 1, 1, 1)
                self.tik_instance.vln(
                    1, a_ub[repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET * Constant.BLOCK],
                    b_ub[repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET * Constant.BLOCK],
                    repeat_times - repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET, 1, 1, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.vln(1, a_ub, b_ub, repeat_times, 1, 1, 1, 1)

            # func: `log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)) + max_tmp`
            with self.tik_instance.for_range(0, repeat_times) as s:
                max_tmp.set_as(lamax_ub[s])
                self.tik_instance.vec_adds(1, b_ub[s * Constant.BLOCK], a_ub[s * Constant.BLOCK], max_tmp, 1, 8, 8)

            # func: `log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)) + max_tmp + log_probs`
            with self.tik_instance.if_scope(repeat_times > Constant.REPEAT_OFFSET):
                with self.tik_instance.for_range(0, repeat_times // Constant.REPEAT_OFFSET) as b:
                    self.tik_instance.vec_add(1, a_ub[b * Constant.REPEAT_OFFSET * Constant.BLOCK],
                                              tmp_ub[b * Constant.REPEAT_OFFSET * Constant.BLOCK],
                                              b_ub[b * Constant.REPEAT_OFFSET * Constant.BLOCK],
                                              Constant.REPEAT_OFFSET, 1, 1, 1)
                self.tik_instance.vec_add(
                    1, a_ub[repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET * Constant.BLOCK],
                    tmp_ub[repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET * Constant.BLOCK],
                    b_ub[repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET * Constant.BLOCK],
                    repeat_times - repeat_times // Constant.REPEAT_OFFSET * Constant.REPEAT_OFFSET, 1, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.vec_add(1, a_ub, tmp_ub, b_ub, repeat_times, 1, 1, 1)

            # func: update log_beta in current T
            with self.tik_instance.for_range(start_loop, end) as s:
                offset.set_as((s - start_loop) * Constant.BLOCK)
                log_alpha_ub[output_src + s].set_as(a_ub[offset])

            self.tik_instance.data_move(self.log_alpha_[task_idx * self.alpha_size_up + (t - 1) * self.output_size],
                                        log_alpha_ub[output_dst], 0, 1, self.output_size_up // Constant.BLOCK, 0, 0)

            output_src.set_as(output_dst)
            output_dst.set_as(self.output_size_up - output_src)

        self.tik_instance.data_move(self.log_alpha_[task_idx * self.alpha_size_up + (t_i - 1) * self.output_size],
                                    log_alpha_ub[output_dst], 0, 1, self.output_size_up // Constant.BLOCK, 0, 0)

        a_tmp.set_as(log_alpha_ub[output_dst + 2 * s_i])
        b_tmp.set_as(log_alpha_ub[output_dst + 2 * s_i - 1])
        c_tmp.set_as(0)

        with self.tik_instance.if_scope(a_tmp > b_tmp):
            max_tmp.set_as(a_tmp)
            a_ub[0].set_as(c_tmp)
            a_ub[1].set_as(b_tmp - a_tmp)
        with self.tik_instance.else_scope():
            max_tmp.set_as(b_tmp)
            a_ub[0].set_as(a_tmp - b_tmp)
            a_ub[1].set_as(c_tmp)

        self.tik_instance.vec_exp(2, b_ub, a_ub, 1, 1, 1)
        self.tik_instance.vec_reduce_add(2, a_ub, b_ub, work_tensor_ub, 1, 1)

        self.tik_instance.vln(1, b_ub, a_ub, 1, 1, 1, 1, 1)
        a_tmp.set_as(b_ub[0])
        b_ub[0].set_as(-a_tmp - max_tmp)
        self.tik_instance.data_move(self.neg_log_likelihood_[task_idx * Constant.BLOCK], b_ub, 0, 1, 1, 0, 0)

    def count_trace(self, s_i, targets_ub):
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
        with self.tik_instance.for_range(1, s_i) as idx:
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
        neg_log_likelihood_ub = self.tik_instance.Tensor("float32", [Constant.BLOCK], name="neg_log_likelihood_ub",
                                                         scope=tik.scope_ubuf)
        log_alpha_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="log_alpha_ub",
                                                scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.N) as task_idx:
            self.tik_instance.data_move(neg_log_likelihood_ub, self.neg_log_likelihood_[task_idx * Constant.BLOCK],
                                        0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.neg_log_likelihood[task_idx], neg_log_likelihood_ub[0], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.T) as task_jdx:
                self.tik_instance.data_move(log_alpha_ub,
                                            self.log_alpha_[task_idx * self.alpha_size_up \
                                                            + task_jdx * self.output_size],
                                            0, 1, self.output_size_up // Constant.BLOCK, 0, 0)
                self.tik_instance.data_move(self.log_alpha[task_idx * self.alpha_size + task_jdx * self.output_size],
                                            log_alpha_ub, 0, 1, self.output_size_up // Constant.BLOCK, 0, 0)


# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
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
