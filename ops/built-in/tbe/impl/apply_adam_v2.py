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
apply_adam_v2.py
"""

from te.utils import para_check
from te.utils.error_manager import error_manager_vector

from impl.ascend import AContainer
from impl.ascend import TensorOperatorParam
from impl.ascend import VecCmd
from impl.ascend import VecExecutor


class BertAdam(object):

    def __init__(self, data_num, data_type, kernel_name, cont):
        self.data_num = data_num
        self.data_type = data_type
        self.kernel_name = kernel_name
        self.cont = cont

        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ai_core_use = self.cont.const_aicore_num
        self.ub_size = self.cont.const_ub_max_byte

        self.data_size, self.data_block_data_num, self.data_repeat_data_num = self.get_type_const(self.data_type)

        data_shape = (self.data_num,)
        data_shape_1 = (1,)
        self.var = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_gm, "var")
        self.m = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_gm, "m")
        self.v = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_gm, "v")
        self.lr = self.tik_inst.Tensor(self.data_type, data_shape_1, self.tik.scope_gm, "lr")

        self.beta1 = self.tik_inst.Tensor(self.data_type, data_shape_1, self.tik.scope_gm, "beta1")
        self.beta2 = self.tik_inst.Tensor(self.data_type, data_shape_1, self.tik.scope_gm, "beta2")
        self.epsilon = self.tik_inst.Tensor(self.data_type, data_shape_1, self.tik.scope_gm, "epsilon")
        self.grad = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_gm, "grad")

        self.max_grad_norm = self.tik_inst.Tensor(self.data_type, data_shape_1, self.tik.scope_gm, "max_grad_norm")
        self.global_grad_norm = self.tik_inst.Tensor(self.data_type, data_shape_1,
                                                     self.tik.scope_gm, "global_grad_norm")
        self.weight_decay = self.tik_inst.Tensor(self.data_type, data_shape_1, self.tik.scope_gm, "weight_decay")

        self.var_out = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_gm, "var_out")
        self.m_out = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_gm, "m_out")
        self.v_out = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_gm, "v_out")

    def ceil_div(self, dividend_, divisor_):
        result_ = (dividend_ + divisor_ - 1) // divisor_
        return result_

    def get_loop_info(self, all_data_num_, each_loop_num_):
        loop_times_ = self.ceil_div(all_data_num_, each_loop_num_)
        last_loop_num_ = all_data_num_ - each_loop_num_ * (loop_times_ - 1)
        return loop_times_, last_loop_num_

    def get_align_num(self, input_num_, align_num_, ceil=True):
        if ceil:
            result_ = self.ceil_div(input_num_, align_num_) * align_num_
        else:
            result_ = input_num_ // align_num_ * align_num_
        return result_

    def get_type_const(self, data_type):
        data_size = self.cont.const_dtype_byte.get(data_type)
        block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self.cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    def mode_compute(self):
        each_core_data_num = self.ceil_div(self.data_num, self.ai_core_use)
        each_core_data_num = self.get_align_num(each_core_data_num, self.data_repeat_data_num)
        self.ai_core_use, last_core_data_num = self.get_loop_info(self.data_num, each_core_data_num)
        with self.tik_inst.for_range(0, self.ai_core_use, block_num=self.ai_core_use) as core_index_s:
            data_index_core_s = each_core_data_num * core_index_s
            with self.tik_inst.if_scope(core_index_s != self.ai_core_use - 1):
                self._mode_compute_each_core(data_index_core_s, each_core_data_num)
            with self.tik_inst.else_scope():
                self._mode_compute_each_core(data_index_core_s, last_core_data_num)

        inputs_all = [self.var, self.m, self.v, self.lr,
                      self.beta1, self.beta2, self.epsilon, self.grad, self.max_grad_norm,
                      self.global_grad_norm, self.weight_decay]
        outputs_all = [self.var_out, self.m_out, self.v_out]
        self.tik_inst.BuildCCE(
            inputs=inputs_all,
            outputs=outputs_all,
            kernel_name=self.kernel_name)

    def _init_beta_scalar(self, scalar_all):
        """
        init beta1, beta2, ne_beta1, ne_beta2
        """
        mask, repeat_num, block_num = 1, 1, 1
        with self.tik_inst.new_stmt_scope():
            data_shape = (self.data_block_data_num,)
            beta1_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "beta1_ub")
            beta2_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "beta2_ub")
            one_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "one_ub")
            self.tik_inst.data_move(beta1_ub, self.beta1, 0, 1, block_num, 0, 0)
            self.tik_inst.data_move(beta2_ub, self.beta2, 0, 1, block_num, 0, 0)
            scalar_all.get("beta1").set_as(beta1_ub[0])
            scalar_all.get("beta2").set_as(beta2_ub[0])
            self.tik_inst.vector_dup(mask, one_ub, 1, repeat_num, 1, 8)
            self.tik_inst.vsub(mask, beta1_ub, one_ub, beta1_ub, repeat_num, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vsub(mask, beta2_ub, one_ub, beta2_ub, repeat_num, 1, 1, 1, 8, 8, 8)
            scalar_all.get("ne_beta1").set_as(beta1_ub[0])
            scalar_all.get("ne_beta2").set_as(beta2_ub[0])

    def _init_judge_scalar(self, scalar_all):
        """
        init max_grad_norm, max_grad_norm_i, rec_global_grad_norm, global_grad_norm_i, weight_decay, weight_decay_i
        """
        mask, repeat_num, block_num = 1, 1, 1
        int_type = scalar_all.get("weight_decay_i").dtype
        with self.tik_inst.new_stmt_scope():
            data_shape = (self.data_block_data_num,)
            max_grad_norm_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "max_grad_norm_ub")
            max_grad_norm_i_ub = self.tik_inst.Tensor(int_type, data_shape, self.tik.scope_ubuf, "max_grad_norm_i_ub")
            global_grad_norm_ub = self.tik_inst.Tensor(self.data_type, data_shape,
                                                       self.tik.scope_ubuf, "global_grad_norm_ub")
            global_grad_norm_i_ub = self.tik_inst.Tensor(int_type, data_shape,
                                                         self.tik.scope_ubuf, "global_grad_norm_i_ub")
            weight_decay_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "weight_decay_ub")
            weight_decay_i_ub = self.tik_inst.Tensor(int_type, data_shape, self.tik.scope_ubuf, "weight_decay_i_ub")
            self.tik_inst.data_move(max_grad_norm_ub, self.max_grad_norm, 0, 1, block_num, 0, 0)
            self.tik_inst.data_move(global_grad_norm_ub, self.global_grad_norm, 0, 1, block_num, 0, 0)
            self.tik_inst.data_move(weight_decay_ub, self.weight_decay, 0, 1, block_num, 0, 0)
            self.tik_inst.vconv(mask, "ceil", max_grad_norm_i_ub, max_grad_norm_ub, repeat_num, 0, 0, 0, 0)
            self.tik_inst.vconv(mask, "ceil", global_grad_norm_i_ub, global_grad_norm_ub, repeat_num, 0, 0, 0, 0)
            self.tik_inst.vconv(mask, "ceil", weight_decay_i_ub, weight_decay_ub, repeat_num, 0, 0, 0, 0)

            scalar_all.get("max_grad_norm").set_as(max_grad_norm_ub[0])
            scalar_all.get("max_grad_norm_i").set_as(max_grad_norm_i_ub[0])
            scalar_all.get("global_grad_norm_i").set_as(global_grad_norm_i_ub[0])
            scalar_all.get("weight_decay").set_as(weight_decay_ub[0])
            scalar_all.get("weight_decay_i").set_as(weight_decay_i_ub[0])

            self.tik_inst.vector_dup(mask, weight_decay_ub, 1, repeat_num, 1, 8)
            self.tik_inst.vdiv(mask, global_grad_norm_ub, weight_decay_ub, global_grad_norm_ub, repeat_num, 1, 1, 1, 8,
                               8, 8)
            scalar_all.get("rec_global_grad_norm").set_as(global_grad_norm_ub[0])

    def _init_other_scalar(self, scalar_all):
        """
        init epsilon, lr_scheduled
        """
        block_num = 1
        with self.tik_inst.new_stmt_scope():
            data_shape = (self.data_block_data_num,)
            epsilon_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "epsilon_ub")
            lr_scheduled_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "lr_scheduled_ub")
            self.tik_inst.data_move(epsilon_ub, self.epsilon, 0, 1, block_num, 0, 0)
            self.tik_inst.data_move(lr_scheduled_ub, self.lr, 0, 1, block_num, 0, 0)
            scalar_all.get("epsilon").set_as(epsilon_ub[0])
            scalar_all.get("lr_scheduled").set_as(lr_scheduled_ub[0])

    def _init_scalar_all(self):
        """
        init scalar
        """
        int_type = "int32"
        scalar_all = {
            "beta1": self.tik_inst.Scalar(self.data_type),
            "ne_beta1": self.tik_inst.Scalar(self.data_type),
            "beta2": self.tik_inst.Scalar(self.data_type),
            "ne_beta2": self.tik_inst.Scalar(self.data_type),
            "max_grad_norm": self.tik_inst.Scalar(self.data_type),
            "max_grad_norm_i": self.tik_inst.Scalar(int_type),
            "rec_global_grad_norm": self.tik_inst.Scalar(self.data_type),
            "global_grad_norm_i": self.tik_inst.Scalar(int_type),
            "weight_decay": self.tik_inst.Scalar(self.data_type),
            "weight_decay_i": self.tik_inst.Scalar(int_type),
            "epsilon": self.tik_inst.Scalar(self.data_type),
            "lr_scheduled": self.tik_inst.Scalar(self.data_type),
        }
        self._init_beta_scalar(scalar_all)
        self._init_judge_scalar(scalar_all)
        self._init_other_scalar(scalar_all)
        return scalar_all

    def _mode_compute_each_core(self, data_index_core_s, data_num):
        scalar_all = self._init_scalar_all()
        with self.tik_inst.if_scope(scalar_all.get("max_grad_norm_i") > 0):
            with self.tik_inst.if_scope(scalar_all.get("global_grad_norm_i") > 1):
                self._judge_weight_decay_each_core(scalar_all, data_index_core_s, data_num, 1)
            with self.tik_inst.else_scope():
                self._judge_weight_decay_each_core(scalar_all, data_index_core_s, data_num, 0)
        with self.tik_inst.else_scope():
            self._judge_weight_decay_each_core(scalar_all, data_index_core_s, data_num, 0)

    def _judge_weight_decay_each_core(self, scalar_all, data_index_core_s, data_num, compute_mode):
        with self.tik_inst.if_scope(scalar_all.get("weight_decay_i") > 0):
            self._compute_each_core(scalar_all, data_index_core_s, data_num, compute_mode + 2)
        with self.tik_inst.else_scope():
            self._compute_each_core(scalar_all, data_index_core_s, data_num, compute_mode)

    def _init_data_tensor(self, each_loop_data_num, last_loop_data_num):
        data_shape = (each_loop_data_num,)
        exp_avg_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "exp_avg_ub")
        exp_avg_sq_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "exp_avg_sq_ub")
        combined_param_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "combined_param_ub")
        combined_grad_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "combined_grad_ub")
        temp_tensor_ub = self.tik_inst.Tensor(self.data_type, data_shape, self.tik.scope_ubuf, "temp_tensor_ub")
        data_buf = {
            "exp_avg_ub": TensorOperatorParam(exp_avg_ub, each_loop_data_num, 0),
            "exp_avg_last_ub": TensorOperatorParam(exp_avg_ub, last_loop_data_num, 0),
            "exp_avg_sq_ub": TensorOperatorParam(exp_avg_sq_ub, each_loop_data_num, 0),
            "combined_param_ub": TensorOperatorParam(combined_param_ub, each_loop_data_num, 0),
            "combined_grad_ub": TensorOperatorParam(combined_grad_ub, each_loop_data_num, 0),
            "temp_tensor_ub": TensorOperatorParam(temp_tensor_ub, each_loop_data_num, 0),
        }
        return data_buf

    def _get_loop_thread_num(self, data_num):
        each_repeat_size = self.data_repeat_data_num * self.data_size * 5
        repeat_num = self.ub_size // each_repeat_size
        each_loop_data_num = min(repeat_num * self.data_repeat_data_num, data_num)
        return each_loop_data_num, 1

    def _compute_each_core(self, scalar_all, data_index_core_s, data_num, compute_mode):
        each_loop_data_num, thread_num = self._get_loop_thread_num(data_num)
        loop_times, last_loop_data_num = self.get_loop_info(data_num, each_loop_data_num)
        with self.tik_inst.for_range(0, loop_times, thread_num=min(loop_times, thread_num)) as loop_index_s:
            data_buf = self._init_data_tensor(each_loop_data_num, last_loop_data_num)
            data_index_loop_s = data_index_core_s + each_loop_data_num * loop_index_s
            with self.tik_inst.if_scope(loop_index_s != loop_times - 1):
                self._mode_compute_each_loop(data_buf, scalar_all, data_index_loop_s, each_loop_data_num,
                                             compute_mode, "exp_avg_ub")
            with self.tik_inst.else_scope():
                self._mode_compute_each_loop(data_buf, scalar_all, data_index_loop_s, last_loop_data_num,
                                             compute_mode, "exp_avg_last_ub")

    def _mode_compute_each_loop(self, data_buf, scalar_all, data_index_loop_s, data_num, compute_mode, drive_buf_name):
        self._data_move_in(data_buf, data_index_loop_s, data_num)
        self._start_compute(data_buf, scalar_all, compute_mode, drive_buf_name)
        self._data_move_out(data_buf, data_index_loop_s, data_num)

    def _data_move_in(self, data_buf, data_index_loop_s, data_num):
        """
        data move in
        """
        exp_avg_ub = data_buf.get("exp_avg_ub").const_tensor
        exp_avg_sq_ub = data_buf.get("exp_avg_sq_ub").const_tensor
        combined_param_ub = data_buf.get("combined_param_ub").const_tensor
        combined_grad_ub = data_buf.get("combined_grad_ub").const_tensor
        block_num = self.ceil_div(data_num, self.data_block_data_num)
        self.tik_inst.data_move(exp_avg_ub, self.m[data_index_loop_s], 0, 1, block_num, 0, 0)
        self.tik_inst.data_move(exp_avg_sq_ub, self.v[data_index_loop_s], 0, 1, block_num, 0, 0)
        self.tik_inst.data_move(combined_param_ub, self.var[data_index_loop_s], 0, 1, block_num, 0, 0)
        self.tik_inst.data_move(combined_grad_ub, self.grad[data_index_loop_s], 0, 1, block_num, 0, 0)

    def _start_compute(self, data_buf, scalar_all, compute_mode, drive_buf_name):
        """
        start compute:
            if compute_mode % 2 == 1: combined_grad /= global_grad_norm
            exp_avg = exp_avg * beta1 + combined_grad * (1 - beta1)
            exp_avg_sq = exp_avg_sq * beta2 + combined_grad * combined_grad * (1 - beta2)
            update = exp_avg / (exp_avg_sq.sqrt() + epsilon)
            if compute_mode // 2 == 1: update += weight_decay * combined_param
            update_with_lr = lr_scheduled * update
            combined_param -= update_with_lr
        """
        compute_cmd = []
        if compute_mode % 2 == 1:
            compute_cmd.append(
                VecCmd(cmd_name="vmuls", dst_name="combined_grad_ub",
                       src0_name="combined_grad_ub", scalar=scalar_all.get("rec_global_grad_norm")))
        compute_cmd.extend([
            VecCmd(cmd_name="vmuls", dst_name="exp_avg_ub",
                   src0_name="exp_avg_ub", scalar=scalar_all.get("beta1")),
            VecCmd(cmd_name="vmuls", dst_name="temp_tensor_ub",
                   src0_name="combined_grad_ub", scalar=scalar_all.get("ne_beta1")),
            VecCmd(cmd_name="vadd", dst_name="exp_avg_ub",
                   src0_name="exp_avg_ub", src1_name="temp_tensor_ub"),

            VecCmd(cmd_name="vmuls", dst_name="exp_avg_sq_ub",
                   src0_name="exp_avg_sq_ub", scalar=scalar_all.get("beta2")),
            VecCmd(cmd_name="vmul", dst_name="temp_tensor_ub",
                   src0_name="combined_grad_ub", src1_name="combined_grad_ub"),
            VecCmd(cmd_name="vmuls", dst_name="temp_tensor_ub",
                   src0_name="temp_tensor_ub", scalar=scalar_all.get("ne_beta2")),
            VecCmd(cmd_name="vadd", dst_name="exp_avg_sq_ub",
                   src0_name="exp_avg_sq_ub", src1_name="temp_tensor_ub"),

            VecCmd(cmd_name="vsqrt", dst_name="temp_tensor_ub",
                   src0_name="exp_avg_sq_ub"),
            VecCmd(cmd_name="vadds", dst_name="temp_tensor_ub",
                   src0_name="temp_tensor_ub", scalar=scalar_all.get("epsilon")),
            VecCmd(cmd_name="vdiv", dst_name="temp_tensor_ub",
                   src0_name="exp_avg_ub", src1_name="temp_tensor_ub")])

        if compute_mode // 2 == 1:
            compute_cmd.extend(
                [VecCmd(cmd_name="vmuls", dst_name="combined_grad_ub",
                        src0_name="combined_param_ub", scalar=scalar_all.get("weight_decay")),
                 VecCmd(cmd_name="vadd", dst_name="temp_tensor_ub",
                        src0_name="temp_tensor_ub", src1_name="combined_grad_ub")])

        compute_cmd.extend(
            [VecCmd(cmd_name="vmuls", dst_name="temp_tensor_ub",
                    src0_name="temp_tensor_ub", scalar=scalar_all.get("lr_scheduled")),
             VecCmd(cmd_name="vsub", dst_name="combined_param_ub",
                    src0_name="combined_param_ub", src1_name="temp_tensor_ub")])
        VecExecutor.exec_vec_cmd(data_buf, compute_cmd, drive_buf_name)

    def _data_move_out(self, data_buf, data_index_loop_s, data_num):
        """
        exp_avg_result, exp_avg_sq_result, combined_param_result, data move out
        """
        exp_avg_ub = data_buf.get("exp_avg_ub").const_tensor
        exp_avg_sq_ub = data_buf.get("exp_avg_sq_ub").const_tensor
        combined_param_ub = data_buf.get("combined_param_ub").const_tensor
        block_num = self.ceil_div(data_num, self.data_block_data_num)
        self.tik_inst.data_move(self.m_out[data_index_loop_s], exp_avg_ub, 0, 1, block_num, 0, 0)
        self.tik_inst.data_move(self.v_out[data_index_loop_s], exp_avg_sq_ub, 0, 1, block_num, 0, 0)
        self.tik_inst.data_move(self.var_out[data_index_loop_s], combined_param_ub, 0, 1, block_num, 0, 0)


def check_params(var, m, v, lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay,
                 var_out, m_out, v_out, kernel_name):
    """
    check params
    """
    input_shape = var.get("shape")
    input_dtype = var.get("dtype").lower()
    input_shape_1 = beta1.get("shape")
    if input_shape_1 != (1, ):
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "shape of exp_avg", "must be (1, )", str(input_shape_1))
    if input_dtype not in ("float16", "float32"):
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "dtype of exp_avg", "float32 or float16", input_dtype)
    param_list_0 = (var, m, v, grad, var_out, m_out, v_out)
    param_name_list_0 = ("var", "m", "v", "grad", "var_out", "m_out", "v_out")
    for param, param_name in zip(param_list_0, param_name_list_0):
        param_shape = param.get("shape")
        param_dtype = param.get("dtype").lower()
        if param_shape != input_shape:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "shape of {}".format(param_name), str(input_shape), str(param_shape))
        if param_dtype != input_dtype:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "dtype of {}".format(param_name), input_dtype, param_dtype)

    param_list_1 = (lr, beta1, beta2, epsilon, max_grad_norm, global_grad_norm, weight_decay)
    param_name_list_1 = ("lr", "beta1", "beta2", "epsilon", "max_grad_norm", "global_grad_norm", "weight_decay")
    for param, param_name in zip(param_list_1, param_name_list_1):
        param_shape = param.get("shape")
        param_dtype = param.get("dtype").lower()
        if param_shape != input_shape_1:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "shape of {}".format(param_name), str(input_shape_1), str(param_shape))
        if param_dtype != input_dtype:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "dtype of {}".format(param_name), input_dtype, param_dtype)


def get_data_num(data_shape):
    data_num = 1
    for data_dim in data_shape:
        data_num *= data_dim
    return data_num


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
# pylint: disable=unused-argument
def apply_adam_v2(var, m, v, lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay,
                  var_out, m_out, v_out, kernel_name="ApplyAdamV2"):
    """
    algorithm: assign positive bboxes
        if max_grad_norm > 0 and global_grad_norm > 1: combined_grad /= global_grad_norm
        m = m * beta1 + combined_grad * (1 - beta1)
        v = v * beta2 + combined_grad * combined_grad * (1 - beta2)
        update = m / (v.sqrt() + epsilon)
        if weight_decay > 0: update += weight_decay * var
        update_with_lr = lr * update
        var -= update_with_lr
    Parameters
    ----------
    var:
        A Tensor. Support float16/float32.
    m :
        A Tensor. Datatype and shape are same as var.
    v:
        A Tensor. Datatype and shape are same as var.
    lr:
        A Tensor. Datatype is same as var. Shape (1, )
    beta1 :
        A Tensor. Datatype is same as var. Shape (1, )
    beta2 :
        A Tensor. Datatype is same as var. Shape (1, )
    epsilon :
        A Tensor. Datatype is same as var. Shape (1, )
    grad :
        A Tensor. Datatype and shape are same as var.
    max_grad_norm:
        A Tensor. Datatype is same as var. Shape (1, )
    global_grad_norm :
        A Tensor. Datatype is same as var. Shape (1, )
    weight_decay :
        A Tensor. Datatype is same as var. Shape (1, )
    var_out:
        A Tensor. Datatype and shape are same as var.
    m_out:
        A Tensor. Datatype and shape are same as var.
    v_out:
        A Tensor. Datatype and shape are same as var.
    kernel_name : str
        cce kernel name, default value is ApplyAdamV2
    Returns
    -------
    None
    """
    check_params(var, m, v, lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay,
                 var_out, m_out, v_out, kernel_name)
    AContainer.reset_instance()
    cont = AContainer.get_instance()
    data_type = var.get("dtype").lower()
    data_shape = var.get("shape")
    data_num = get_data_num(data_shape)
    obj = BertAdam(data_num, data_type, kernel_name, cont)
    obj.mode_compute()
