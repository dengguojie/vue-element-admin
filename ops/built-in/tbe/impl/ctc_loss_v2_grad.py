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
ctc_loss_v2_grad
"""

from impl.util.platform_adapter import tik
from topi.cce import util
from impl.util.platform_adapter import para_check
from te.platform.fusion_manager import fusion_manager

BLOCK = 8
MIN = -3.4e38
REPEAT_OFFSET = 255

@fusion_manager.register("ctc_loss_v2_grad")
class CTCLossV2Grad(object):
    """
    Function: Class CTCLossV2Grad.
    Modify : 2021-5-26
    """
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
        self.rounds = self.C_BLOCK // (BLOCK * REPEAT_OFFSET)

        self.output_size = 2 * self.S + 1
        self.output_size_up = (self.output_size + BLOCK - 1) // BLOCK * BLOCK
        self.alpha_size = self.T * self.output_size

        self.grad_out = self.tik_instance.Tensor("float32", [self.N], name="grad_out", scope=tik.scope_gm)
        self.log_probs = self.tik_instance.Tensor("float32", [self.T, self.N, self.C], name="log_probs",
                                                  scope=tik.scope_gm)
        self.targets = self.tik_instance.Tensor("int32", [self.N, self.S], name="targets", scope=tik.scope_gm)
        self.input_lengths = self.tik_instance.Tensor("int32", [self.N], name="input_lengths", scope=tik.scope_gm)
        self.target_lengths = self.tik_instance.Tensor("int32", [self.N], name="target_lengths", scope=tik.scope_gm)

        self.neg_log_likelihood = self.tik_instance.Tensor("float32", [self.N], name="neg_log_likelihood",
                                                           scope=tik.scope_gm)
        self.log_alpha = self.tik_instance.Tensor("float32", [self.N, self.T, self.output_size], name="log_alpha",
                                                  scope=tik.scope_gm)
        self.grad = self.tik_instance.Tensor("float32", [self.T, self.N, self.C], name="grad",
                                             scope=tik.scope_gm)
        self.grad_ = self.tik_instance.Tensor("float32", [self.T, self.N, self.C_BLOCK], name="grad_",
                                              scope=tik.scope_gm, is_workspace=True)

        self.available_aicore_num = tik.Dprofile().get_aicore_num()
        self.used_aicore_num = self.available_aicore_num if self.N > self.available_aicore_num else self.N
        self.batch_num_per_aicore = self.N // self.used_aicore_num
        self.batch_tail = self.N % self.used_aicore_num

    def paras_check(self, log_probs, targets, kernel_name):
        """
        Function: paras_check.
        Modify : 2021-5-26
        """
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

    def ctc_loss_grad_compute(self):
        """
        Function: ctc_loss_grad_compute.
        Modify : 2021-5-26
        """
        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.ctc_loss_grad_compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.ctc_loss_grad_compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)

        with self.tik_instance.if_scope(self.C % BLOCK != 0):
            self.move_out()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.grad_out, self.log_probs, self.targets, self.input_lengths,
                                           self.target_lengths, self.neg_log_likelihood, self.log_alpha],
                                   outputs=[self.grad])

        return self.tik_instance

    def ctc_loss_grad_compute_core(self, task_idx):
        """
        Function: ctc_loss_grad_compute_core.
        Modify : 2021-5-26
        """
        grad_out_ub = self.tik_instance.Tensor("float32", [BLOCK], name="grad_out_ub", scope=tik.scope_ubuf)
        targets_ub = self.tik_instance.Tensor("int32", [self.S_BLOCK], name="targets_ub", scope=tik.scope_ubuf)
        input_length_ub = self.tik_instance.Tensor("int32", [BLOCK], name="input_length_ub", scope=tik.scope_ubuf)
        target_length_ub = self.tik_instance.Tensor("int32", [BLOCK], name="target_length_ub", scope=tik.scope_ubuf)
        grad_ub = self.tik_instance.Tensor("float32", [self.C_BLOCK], name="grad_ub", scope=tik.scope_ubuf)
        neg_log_likelihood_ub = self.tik_instance.Tensor("float32", [BLOCK], name="neg_log_likelihood_ub",
                                                         scope=tik.scope_ubuf)
        # func: initial grad_ub
        with self.tik_instance.for_range(0, self.rounds) as j:
            self.tik_instance.vector_dup(BLOCK, grad_ub[(BLOCK * REPEAT_OFFSET) * j], 0, REPEAT_OFFSET, 1, 1)
        self.tik_instance.vector_dup(BLOCK, grad_ub[(BLOCK * REPEAT_OFFSET) * self.rounds], 0,
                                     (self.C_BLOCK % (BLOCK * REPEAT_OFFSET)) // BLOCK, 1, 1)

        self.tik_instance.data_move(grad_out_ub, self.grad_out[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(targets_ub, self.targets[task_idx * self.S], 0, 1, self.S_BLOCK // BLOCK, 0, 0)
        self.tik_instance.data_move(input_length_ub, self.input_lengths[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(target_length_ub, self.target_lengths[task_idx], 0, 1, 1, 0, 0)
        self.tik_instance.data_move(neg_log_likelihood_ub, self.neg_log_likelihood[task_idx], 0, 1, 1, 0, 0)
        # func: recored current T and S
        T_i = self.tik_instance.Scalar("int32", init_value=input_length_ub[0])
        S_i = self.tik_instance.Scalar("int32", init_value=target_length_ub[0])
        # func: get valid compute trace
        repeats, s_inc, e_inc = self.count_trace(S_i, targets_ub)

        start = self.tik_instance.Scalar("int32")
        end_loop = self.tik_instance.Scalar("int32")
        end = self.tik_instance.Scalar("int32")
        remain = self.tik_instance.Scalar("int32")
        current_target = self.tik_instance.Scalar("int32")
        next_target = self.tik_instance.Scalar("int32")
        tmp = self.tik_instance.Scalar("int32")
        a_tmp = self.tik_instance.Scalar("float32")
        b_tmp = self.tik_instance.Scalar("float32")
        lcab = self.tik_instance.Scalar("float32")
        res = self.tik_instance.Scalar("float32")
        lp = self.tik_instance.Scalar("float32")
        nll = self.tik_instance.Scalar("float32", init_value=neg_log_likelihood_ub[0])
        grad_out = self.tik_instance.Scalar("float32", init_value=grad_out_ub[0])
        min_float = self.tik_instance.Scalar("float32", init_value=MIN)
        
        # func: a_ub/b_ub/tmp_ub: used in exp/log/add/sub api
        a_ub = self.tik_instance.Tensor("float32", [self.output_size, BLOCK], name="a_ub", scope=tik.scope_ubuf)
        b_ub = self.tik_instance.Tensor("float32", [self.output_size, BLOCK], name="b_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor("float32", [self.output_size, BLOCK], name="tmp_ub", scope=tik.scope_ubuf)

        work_tensor_ub = self.tik_instance.Tensor("float32", [BLOCK], name="work_tensor_ub", scope=tik.scope_ubuf)
        log_probs_ub = self.tik_instance.Tensor("float32", [self.C_BLOCK], name="log_probs_ub", scope=tik.scope_ubuf)

        # beta_grad_update
        offset = self.tik_instance.Scalar("int32")
        repeat_times = self.tik_instance.Scalar("int32")
        c_tmp = self.tik_instance.Scalar("float32")
        zero_tmp = self.tik_instance.Scalar("float32", init_value=0)
        alpha_beta_tmp = self.tik_instance.Scalar("float32")
        max_tmp = self.tik_instance.Scalar("float32")

        copy_ub_a = self.tik_instance.Tensor("float32", [self.C_BLOCK], name="copy_ub_a", scope=tik.scope_ubuf)
        copy_ub_b = self.tik_instance.Tensor("float32", [self.C_BLOCK], name="copy_ub_b", scope=tik.scope_ubuf)
        copy_ub_zero = self.tik_instance.Tensor("float32", [self.C_BLOCK], name="copy_ub_zero", scope=tik.scope_ubuf)
        
        # func: init copy_ub_zero 
        with self.tik_instance.for_range(0, self.rounds) as j:
            self.tik_instance.vector_dup(BLOCK, copy_ub_zero[(BLOCK * REPEAT_OFFSET) * j], 0, REPEAT_OFFSET, 1, 1)
        self.tik_instance.vector_dup(BLOCK, copy_ub_zero[(BLOCK * REPEAT_OFFSET) * self.rounds], 0,
                                     (self.C_BLOCK % (BLOCK * REPEAT_OFFSET)) // BLOCK, 1, 1)

        with self.tik_instance.for_range(T_i, self.T) as t:
             with self.tik_instance.if_scope(self.C % BLOCK != 0):
                self.tik_instance.data_move(self.grad_[t * self.N * self.C_BLOCK + task_idx * self.C_BLOCK],
                                            copy_ub_zero[0], 0, 1, self.C_BLOCK // BLOCK, 1, 1)
             with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.grad[t * self.N * self.C + task_idx * self.C],
                                            copy_ub_zero[0], 0, 1, self.C_BLOCK // BLOCK, 1, 1)
        # func: get log_prob in current T
        self.tik_instance.data_move(log_probs_ub[0], self.log_probs[self.C * task_idx + self.N * self.C * (T_i - 1)],
                                    0, 1, self.C_BLOCK // BLOCK, 0, 0)

        output_dst = self.tik_instance.Scalar("int32", init_value=0)
        output_src = self.tik_instance.Scalar("int32", init_value=self.output_size_up)
        # func: calculate log_beta in current T
        log_beta_ub = self.tik_instance.Tensor("float32", [2, self.output_size_up], name="log_beta_ub",
                                               scope=tik.scope_ubuf)
        current_target_ub = self.tik_instance.Tensor("int32", [self.output_size_up], name="current_target_ub",
                                                     scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(BLOCK, log_beta_ub[output_dst], MIN, self.output_size_up // BLOCK, 1, 1)
        lamax_ub = self.tik_instance.Tensor("float32", [self.output_size], name="lamax_ub", scope=tik.scope_ubuf)
        log_beta_ub[output_dst + 2 * S_i].set_as(log_probs_ub[self.blank])
        current_target.set_as(targets_ub[S_i - 1])
        log_beta_ub[output_dst + 2 * S_i - 1].set_as(log_probs_ub[current_target])
        # func: get log_alpha in current T
        log_alpha_ub = self.tik_instance.Tensor("float32", [self.output_size_up], name="log_alpha_ub",
                                                scope=tik.scope_ubuf)
        self.tik_instance.data_move(log_alpha_ub,
                                    self.log_alpha[task_idx * self.alpha_size + (T_i - 1) * self.output_size],
                                    0, 1, self.output_size_up // BLOCK, 0, 0)

        a_ub[0].set_as(log_alpha_ub[output_dst + 2 * S_i])
        a_ub[1].set_as(log_alpha_ub[output_dst + 2 * S_i - 1])
        b_ub[0].set_as(log_beta_ub[output_dst + 2 * S_i])
        b_ub[1].set_as(log_beta_ub[output_dst + 2 * S_i - 1])
        self.tik_instance.vec_add(2, tmp_ub, a_ub, b_ub, 1, 1, 1, 1)
        # func: update grad_ub in current T with log_alpha and log_beta
        grad_ub[self.blank].set_as(tmp_ub[0])
        grad_ub[current_target].set_as(tmp_ub[1])

        start.set_as(2 * S_i - 1)
        with self.tik_instance.if_scope(repeats < T_i - S_i):
            end.set_as(2 * S_i + 1)
        with self.tik_instance.else_scope():
            end.set_as(2 * S_i)

        t = self.tik_instance.Scalar("int32", init_value=T_i - 1)

        # func: grad = exp(log_probs)
        with self.tik_instance.for_range(0, self.rounds) as j:
            self.tik_instance.vec_exp(BLOCK, copy_ub_a[(BLOCK * REPEAT_OFFSET) * j],
                                      log_probs_ub[(BLOCK * REPEAT_OFFSET) * j], REPEAT_OFFSET, 1, 1)
        self.tik_instance.vec_exp(BLOCK, copy_ub_a[(BLOCK * REPEAT_OFFSET) * self.rounds],
                                  log_probs_ub[(BLOCK * REPEAT_OFFSET) * self.rounds],
                                  (self.C_BLOCK % (BLOCK * REPEAT_OFFSET)) // BLOCK, 1, 1)

        with self.tik_instance.for_range(0, self.C) as c:
            res.set_as(grad_ub[c])
            # func: update certain grad
            with self.tik_instance.if_scope(res != 0):
                lp.set_as(log_probs_ub[c])
                a_ub[0].set_as(res + nll - lp)
                self.tik_instance.vec_exp(1, b_ub, a_ub, 1, 1, 1)

                a_tmp.set_as(copy_ub_a[c])
                b_tmp.set_as(b_ub[0])
                copy_ub_a[c].set_as(a_tmp - b_tmp)

        with self.tik_instance.for_range(0, self.rounds) as j:
            self.tik_instance.vec_muls(BLOCK, copy_ub_b[(BLOCK * REPEAT_OFFSET) * j],
                                       copy_ub_a[(BLOCK * REPEAT_OFFSET) * j], grad_out, REPEAT_OFFSET, 1, 1)
        self.tik_instance.vec_muls(BLOCK, copy_ub_b[(BLOCK * REPEAT_OFFSET) * self.rounds],
                                   copy_ub_a[(BLOCK * REPEAT_OFFSET) * self.rounds], grad_out,
                                   (self.C_BLOCK % (BLOCK * REPEAT_OFFSET)) // BLOCK, 1, 1)

        with self.tik_instance.if_scope(self.C % BLOCK != 0):
            self.tik_instance.data_move(self.grad_[t * self.N * self.C_BLOCK + task_idx * self.C_BLOCK], 
                                        copy_ub_b[0], 0, 1, self.C_BLOCK // BLOCK, 1, 1)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.grad[t * self.N * self.C + task_idx * self.C], 
                                        copy_ub_b[0], 0, 1, self.C_BLOCK // BLOCK, 1, 1)

        with self.tik_instance.for_range(1, T_i):
            t.set_as(t - 1)

            # func: initial grad_ub
            with self.tik_instance.for_range(0, self.rounds) as j:
                self.tik_instance.vector_dup(BLOCK, grad_ub[(BLOCK * REPEAT_OFFSET) * j], 0, REPEAT_OFFSET, 1, 1)
            self.tik_instance.vector_dup(BLOCK, grad_ub[(BLOCK * REPEAT_OFFSET) * self.rounds], 0,
                                         (self.C_BLOCK % (BLOCK * REPEAT_OFFSET)) // BLOCK, 1, 1)

            self.tik_instance.data_move(log_probs_ub[0], self.log_probs[self.C * task_idx + self.N * self.C * t], 0,
                                        1, self.C_BLOCK // BLOCK, 0, 0)
            self.tik_instance.data_move(log_alpha_ub[0],
                                        self.log_alpha[task_idx * self.alpha_size + t * self.output_size],
                                        0, 1, self.output_size_up // BLOCK, 0, 0)
            self.tik_instance.vector_dup(BLOCK, log_beta_ub[output_src], MIN, self.output_size_up // BLOCK, 1, 1)

            remain.set_as(S_i + repeats - T_i + t)
            with self.tik_instance.if_scope(remain >= -1):
                tmp.set_as(s_inc[remain + 1])
                start.set_as(start - tmp)
            with self.tik_instance.if_scope(t < S_i + repeats):
                tmp.set_as(e_inc[t])
                end.set_as(end - tmp)
            end_loop.set_as(end)

            with self.tik_instance.if_scope(end_loop == 2 * S_i + 1):
                current_target.set_as(self.blank)
                a_tmp.set_as(log_beta_ub[output_dst + 2 * S_i])
                b_tmp.set_as(log_probs_ub[self.blank])
                # func: calculate log_beta in current T
                log_beta_ub[output_src + 2 * S_i].set_as(a_tmp + b_tmp)
                end_loop.set_as(end_loop - 1)

                a_tmp.set_as(log_beta_ub[output_src + 2 * S_i])
                b_tmp.set_as(log_alpha_ub[2 * S_i])
                # func: update grad_ub in current T with log_alpha and log_beta
                grad_ub[current_target].set_as(a_tmp + b_tmp)

            with self.tik_instance.for_range(start, end_loop) as s:
                with self.tik_instance.if_scope(s % 2 == 0):
                    current_target.set_as(self.blank)
                with self.tik_instance.else_scope():
                    current_target.set_as(targets_ub[s // 2])
                current_target_ub[s].set_as(current_target)
                    
                offset.set_as((s - start) * BLOCK)
                
                tmp_ub[offset].set_as(log_probs_ub[current_target])
                a_ub[offset].set_as(log_beta_ub[output_dst + s])
                a_ub[offset + 1].set_as(log_beta_ub[output_dst + s + 1])

                with self.tik_instance.if_scope(tik.all((s % 2 != 0), (s < 2 * S_i - 1))):
                    next_target.set_as(targets_ub[s // 2 + 1])
                    with self.tik_instance.if_scope(current_target != next_target):
                        a_ub[offset + 2].set_as(log_beta_ub[output_dst + s + 2])
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
                        lamax_ub[s - start].set_as(a_tmp)
                    with self.tik_instance.else_scope():
                        lamax_ub[s - start].set_as(c_tmp)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(b_tmp > c_tmp):
                        lamax_ub[s - start].set_as(b_tmp)
                    with self.tik_instance.else_scope():
                        lamax_ub[s - start].set_as(c_tmp)

            repeat_times.set_as(end_loop - start)        
            # func: get -max
            with self.tik_instance.for_range(0, repeat_times) as s:
                max_tmp.set_as(lamax_ub[s])
                max_tmp.set_as(-max_tmp)
                self.tik_instance.vec_adds(3, b_ub[s * BLOCK], a_ub[s * BLOCK], max_tmp, 1, 8, 8)

            # func: exp(a_tmp- max_tmp)  exp(b_tmp- max_tmp)  exp(b_tmp- max_tmp)    
            with self.tik_instance.if_scope(repeat_times > REPEAT_OFFSET):  
                with self.tik_instance.for_range(0, repeat_times // REPEAT_OFFSET) as b:
                    self.tik_instance.vec_exp(3, a_ub[b * REPEAT_OFFSET * BLOCK], b_ub[b * REPEAT_OFFSET * BLOCK],
                                              REPEAT_OFFSET, 1, 1)
                self.tik_instance.vec_exp(3, a_ub[repeat_times // REPEAT_OFFSET * REPEAT_OFFSET * BLOCK],
                                          b_ub[repeat_times // REPEAT_OFFSET * REPEAT_OFFSET * BLOCK],
                                          repeat_times - repeat_times // REPEAT_OFFSET * REPEAT_OFFSET, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.vec_exp(3, a_ub, b_ub, repeat_times, 1, 1)

            # func: exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)
            with self.tik_instance.for_range(0, repeat_times) as s:    
                self.tik_instance.vec_reduce_add(3, b_ub[s * BLOCK], a_ub[s * BLOCK], work_tensor_ub, 1, 1)

            # func: log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp))
            with self.tik_instance.if_scope(repeat_times > REPEAT_OFFSET):  
                with self.tik_instance.for_range(0, repeat_times // REPEAT_OFFSET) as b:
                    self.tik_instance.vln(1, a_ub[b * REPEAT_OFFSET * BLOCK], b_ub[b * REPEAT_OFFSET * BLOCK],
                                          REPEAT_OFFSET, 1, 1, 1, 1)
                self.tik_instance.vln(1, a_ub[repeat_times // REPEAT_OFFSET * REPEAT_OFFSET * BLOCK],
                                      b_ub[repeat_times // REPEAT_OFFSET * REPEAT_OFFSET * BLOCK],
                                      repeat_times - repeat_times // REPEAT_OFFSET * REPEAT_OFFSET, 1, 1, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.vln(1, a_ub, b_ub, repeat_times, 1, 1, 1, 1)

            # func: log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)) + max_tmp
            with self.tik_instance.for_range(0, repeat_times) as s: 
                max_tmp.set_as(lamax_ub[s])
                self.tik_instance.vec_adds(1, b_ub[s * BLOCK], a_ub[s * BLOCK], max_tmp, 1, 8, 8)
     
            # func: log(exp(a_tmp- max_tmp) + exp(b_tmp- max_tmp) + exp(b_tmp- max_tmp)) + max_tmp + log_probs
            with self.tik_instance.if_scope(repeat_times > REPEAT_OFFSET):  
                with self.tik_instance.for_range(0, repeat_times // REPEAT_OFFSET) as b:
                    self.tik_instance.vec_add(1, a_ub[b * REPEAT_OFFSET * BLOCK], tmp_ub[b * REPEAT_OFFSET * BLOCK],
                                              b_ub[b * REPEAT_OFFSET * BLOCK], REPEAT_OFFSET, 1, 1, 1)
                self.tik_instance.vec_add(1, a_ub[repeat_times // REPEAT_OFFSET * REPEAT_OFFSET * BLOCK],
                                          tmp_ub[repeat_times // REPEAT_OFFSET * REPEAT_OFFSET * BLOCK],
                                          b_ub[repeat_times // REPEAT_OFFSET * REPEAT_OFFSET * BLOCK],
                                          repeat_times - repeat_times // REPEAT_OFFSET * REPEAT_OFFSET, 1, 1, 1)   
            with self.tik_instance.else_scope():
                self.tik_instance.vec_add(1, a_ub, tmp_ub, b_ub, repeat_times, 1, 1, 1)

            # func: update log_beta in current T
            with self.tik_instance.for_range(start, end_loop) as s:
                current_target.set_as(current_target_ub[s])
                offset.set_as((s - start) * BLOCK)    

                a_tmp.set_as(a_ub[offset])
                log_beta_ub[output_src + s].set_as(a_tmp)
                # func: get log_alpha in current T
                b_tmp.set_as(log_alpha_ub[s])
                alpha_beta_tmp.set_as(a_tmp + b_tmp)
                lcab.set_as(grad_ub[current_target])

                # func: update grad_ub in current T with log_alpha and log_beta
                with self.tik_instance.if_scope(lcab != 0):
                    with self.tik_instance.if_scope(lcab > alpha_beta_tmp):
                        max_tmp.set_as(lcab)
                        a_ub[0].set_as(zero_tmp)
                        a_ub[1].set_as(alpha_beta_tmp - lcab)
                    with self.tik_instance.else_scope():
                        max_tmp.set_as(alpha_beta_tmp)
                        a_ub[0].set_as(lcab - alpha_beta_tmp)
                        a_ub[1].set_as(zero_tmp)     

                    self.tik_instance.vec_exp(2, b_ub, a_ub, 1, 0, 0)
                    self.tik_instance.vec_reduce_add(2, a_ub, b_ub, work_tensor_ub, 1, 1)

                    self.tik_instance.vln(1, b_ub, a_ub, 1, 1, 1, 1, 1)
                    a_tmp.set_as(b_ub[0])
                    grad_ub[current_target].set_as(a_tmp + max_tmp)
                with self.tik_instance.else_scope():
                    grad_ub[current_target].set_as(alpha_beta_tmp)

            # func: grad = exp(log_probs)
            with self.tik_instance.for_range(0, self.rounds) as j:
                self.tik_instance.vec_exp(BLOCK, copy_ub_a[(BLOCK * REPEAT_OFFSET) * j],
                                          log_probs_ub[(BLOCK * REPEAT_OFFSET) * j], REPEAT_OFFSET, 1, 1)
            self.tik_instance.vec_exp(BLOCK, copy_ub_a[(BLOCK * REPEAT_OFFSET) * self.rounds],
                                      log_probs_ub[(BLOCK * REPEAT_OFFSET) * self.rounds],
                                      (self.C_BLOCK % (BLOCK * REPEAT_OFFSET)) // BLOCK, 1, 1)

            with self.tik_instance.for_range(0, self.C) as c:
                res.set_as(grad_ub[c])
                # func: update certain grad
                with self.tik_instance.if_scope(res != 0):
                    lp.set_as(log_probs_ub[c])
                    a_ub[0].set_as(res + nll - lp)
                    self.tik_instance.vec_exp(1, b_ub, a_ub, 1, 1, 1)

                    a_tmp.set_as(copy_ub_a[c])
                    b_tmp.set_as(b_ub[0])
                    copy_ub_a[c].set_as(a_tmp - b_tmp)

            with self.tik_instance.for_range(0, self.rounds) as j:
                self.tik_instance.vec_muls(BLOCK, copy_ub_b[(BLOCK * REPEAT_OFFSET) * j],
                                           copy_ub_a[(BLOCK * REPEAT_OFFSET) * j], grad_out, REPEAT_OFFSET, 1, 1)
            self.tik_instance.vec_muls(BLOCK, copy_ub_b[(BLOCK * REPEAT_OFFSET) * self.rounds],
                                       copy_ub_a[(BLOCK * REPEAT_OFFSET) * self.rounds], grad_out,
                                       (self.C_BLOCK % (BLOCK * REPEAT_OFFSET)) // BLOCK, 1, 1)

            with self.tik_instance.if_scope(self.C % BLOCK != 0):
                self.tik_instance.data_move(self.grad_[t * self.N * self.C_BLOCK + task_idx * self.C_BLOCK],
                                            copy_ub_b[0], 0, 1, self.C_BLOCK // BLOCK, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.grad[t * self.N * self.C + task_idx * self.C],
                                            copy_ub_b[0], 0, 1, self.C_BLOCK // BLOCK, 1, 1)

            output_src.set_as(output_dst)
            output_dst.set_as(self.output_size_up - output_src)

    def count_trace(self, S_i, targets_ub):
        """
        Function: mark the valid trace.
        Modify : 2021-5-26

        Init base parameters
        Parameters
        ----------
        Inputs:
        S_i: label length.
        targets_ub: label index.
        ----------
        """
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
        temp_ub = self.tik_instance.Tensor("float32", [self.C_BLOCK], name="temp_ub", scope=tik.scope_ubuf)
        
        with self.tik_instance.for_range(0, self.T) as t_idx:      
            with self.tik_instance.for_range(0, self.N) as n_idx:
                self.tik_instance.data_move(temp_ub,
                                            self.grad_[t_idx * self.N * self.C_BLOCK + n_idx * self.C_BLOCK], 0,
                                            1, self.C_BLOCK // BLOCK, 1, 1)
                self.tik_instance.data_move(self.grad[t_idx * self.N * self.C + n_idx * self.C], temp_ub, 0, 1,
                                            self.C_BLOCK // BLOCK, 1, 1)


# pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ctc_loss_v2_grad(grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, grad,
                     blank=0, reduction="mean", zero_infinity=False, kernel_name="ctc_loss_v2_grad"):
    """
    Function: The grad of Connectionist Temporal Classification loss.
    Modify : 2021-5-26

    Init base parameters
    Parameters
    ----------
    Inputs:
    grad_out: Gradient renewal coefficient. Tensor of size (N), where N = batch size.
    Log_probs: Tensor of size (T,N,C), where T =input length, N =batch size,
               and C = number of classes (including blank).
    Targets: Tensor of size (N, S), where S= max target length.
    It represent the target sequences.
    Input_lengths: Tuple or tensor of size (N).
    It represent the lengths of the inputs.
    Target_lengths: Tuple or tensor of size (N). It represent lengths of the targets.
    log_alpha: The probability of possible trace of input to target.
    neg_log_likelihood: A loss value which is differentiable with respect to each input node.

    Attributes:
    blank : Blank label. Default 0.
    reduction: Specifies the reduction to apply to the output. Default: 'mean'.
    zero_infinity : Whether to zero infinite losses and the associated gradients.

    Outputs:
    grad: The grad of Connectionist Temporal Classification loss.
    ----------
    """

    op_obj = CTCLossV2Grad(log_probs, targets, blank, kernel_name)

    return op_obj.ctc_loss_grad_compute()