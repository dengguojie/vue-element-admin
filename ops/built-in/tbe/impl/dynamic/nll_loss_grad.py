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
nll_loss_grad
"""
from impl.common_util import get_data_size
from impl.constant_util import MASK64
from impl.util import util_common
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_tik_comm_func import ceil_div
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    DIM2 = 2
    NUM_EIGHT = 8
    NUM_FOUR = 4
    NEGATIVE = -1
    MAX_REPEAT = 255
    BLOCK_BYTE_SIZE = 32
    NUM_SIXTYFOUR = MASK64
    MAX_INT64_VALUE = 2**64 - 1
    # used for tiling data
    TILING_CTRL_PARAM = ("int64", 128)


# 'pylint: disable=unused-argument, invalid-name, too-many-arguments
def check_supported(x, y_grad, target, weight, total_weight, x_grad, reduction="mean", ignore_index=-100,
                    kernel_name="nll_loss_grad"):
    """
    check nllloss grad supported

    Parameters
    ----------
    x : dict
        shape and dtype of input, the length of shape should be two or one.
    y_grad : dict
        shape and dtype of input, the length of shape must be one.
    target : dict
        shape and dtype of input, the length of shape only support one.
    total_weight : dict
        shape and dtype of input, it is a scalar.
    weight : dict or None
        the length of shape only support one when weight is dict.
    x_grad: dict
        It's a tensor with shape(minibatch, ) when reduction == 'none' and
        the input is 2D. Otherwise, the output is a scalar.
    reduction: str
        default value is "mean"
    ignore_index: int
        default value is -100
    kernel_name : str
        kernel name, default value is "nll_loss_grad"

    Returns
    -------
    (is_supported, description)
    """
    x_shape = x.get("ori_shape")

    if util_common.is_unknown([x, target, weight]):
        return True, ""

    if _dynamic_static_union(x_shape, reduction):
        return True, ""

    return False, ""


def _dynamic_static_union(shape, reduction):
    """
    for dynamic and static union fully verified
    """
    white_list_dict = {
        "none": [[818497, 2], [7353474, 2]],
        "sum": [[818497, 2], [7353474, 2]],
        "mean": [[818497, 2], [7353474, 2]]
    }

    if reduction not in white_list_dict:
        return False

    x_shape = list(shape)
    if x_shape in white_list_dict[reduction]:
        return True

    return False


# 'pylint: disable=unused-argument, invalid-name, too-many-arguments, too-many-locals
def _shape_and_dtype_check(x, y_grad, target, weight, total_weight, reduction, x_grad):
    """
    check shape and dtype
    """
    x_dtype = x.get("dtype").lower()
    y_grad_dtype = y_grad.get("dtype").lower()
    target_dtype = target.get("dtype").lower()
    total_weight_dtype = total_weight.get("dtype").lower()
    weight_dtype = weight.get("dtype").lower()
    x_grad_dtype = x_grad.get("dtype").lower()

    check_list = ("float32",)
    check_list_target = ("int32",)

    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_dtype(y_grad_dtype, check_list, param_name="y_grad")
    para_check.check_dtype(target_dtype, check_list_target, param_name="target")
    para_check.check_dtype(weight_dtype, check_list, param_name="weight")
    para_check.check_dtype(total_weight_dtype, check_list, param_name="total_weight")
    para_check.check_dtype(x_grad_dtype, check_list, param_name="x_grad")

    x_shape = x.get("shape")
    y_grad_shape = y_grad.get("shape")
    target_shape = target.get("shape")
    total_weight_shape = total_weight.get("shape")
    weight_shape = weight.get("shape")
    x_grad_shape = x_grad.get("shape")

    if len(x_shape) > Constant.DIM2:
        error_detail = "The dimension of x should be equal to or less than two."
        error_manager_vector.raise_err_input_shape_invalid("nll_loss_grad", "x", error_detail)
    if len(y_grad_shape) != 1:
        error_detail = "The dimension of y_grad should be 1D."
        error_manager_vector.raise_err_input_shape_invalid("nll_loss_grad", "y_grad", error_detail)
    if len(weight_shape) != 1:
        error_detail = "The dimension of weight should be 1D."
        error_manager_vector.raise_err_input_shape_invalid("nll_loss_grad", "weight", error_detail)
    if len(target_shape) != 1:
        error_detail = "The dimension of target should be 1D."
        error_manager_vector.raise_err_input_shape_invalid("nll_loss_grad", "target", error_detail)
    if len(total_weight_shape) != 1:
        error_detail = "The dimension of total_weight should be 1D."
        error_manager_vector.raise_err_input_shape_invalid("nll_loss_grad", "total_weight", error_detail)
    if len(x_shape) != len(x_grad_shape):
        error_detail = "The length of x and x_grad must be equal."
        error_manager_vector.raise_err_two_input_shape_invalid("nll_loss_grad", "x", "x_grad", error_detail)
    if reduction not in ("mean", "sum", "none"):
        error_detail = "The value of reduction should be in [\"mean\", \"sum\", \"none\"]."
        error_manager_vector.raise_err_specific_reson("nll_loss_grad", error_detail)


# 'pylint: disable=unused-argument, invalid-name, too-many-arguments, too-many-locals
# 'pylint: disable=too-many-instance-attributes
class NllLossGradCompute:
    """
    NLLLOSSGRAD
    """

    def __init__(self, x, target, reduction, ignore_index, kernel_name):
        self.tik_instance = tik.Tik()
        self.tik_profiling = tik.Dprofile()
        self.real_core_num = self.tik_profiling.get_aicore_num()
        self.tiling_dtype_bytes = get_data_size(Constant.TILING_CTRL_PARAM[0])
        self.ub_size = self.tik_profiling.get_unified_buffer_size() - 2048
        self.reduction = reduction
        self.ignore_idx = ignore_index
        self.kernel_name = kernel_name
        self.dtype = x.get("dtype").lower()
        self.dtype_target = target.get("dtype").lower()

        self.data_x = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT64_VALUE,),
                                               name="data_x",
                                               scope=tik.scope_gm)
        self.data_y_grad = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT64_VALUE,),
                                                    name="data_y_grad",
                                                    scope=tik.scope_gm)
        self.data_weight = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT64_VALUE,),
                                                    name="data_weight",
                                                    scope=tik.scope_gm)
        self.data_target = self.tik_instance.Tensor(self.dtype_target, (Constant.MAX_INT64_VALUE,),
                                                    name="data_target",
                                                    scope=tik.scope_gm)
        self.data_total_weight = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT64_VALUE,),
                                                          name="data_total_weight",
                                                          scope=tik.scope_gm)
        self.output = self.tik_instance.Tensor(self.dtype, (Constant.MAX_INT64_VALUE,),
                                               name="output",
                                               scope=tik.scope_gm)

        self.tiling_gm = self.tik_instance.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.tiling_ub = self.tik_instance.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tiling_params = [
            self.tik_instance.Scalar(Constant.TILING_CTRL_PARAM[0]) for i in range(Constant.TILING_CTRL_PARAM[1])
        ]
        self.get_tiling_params()
        self.init_tiling_params()
        self.y_grad_ub = None
        self.target_ub = None
        self.weight_ub = None
        self.target_value_ub = None
        self.refactor_weight_ub = None
        self.total_weight_ub = None
        self.dup_ub = None
        self.index_x = None

    def init_tiling_params(self):
        """
        init tiling parameters function
        """
        self.c_dim = self.tiling_params[0]
        self.n_dim = self.tiling_params[1]
        self.invalid_target = self.tiling_params[2]
        self.ignore_index = self.tiling_params[3]
        self.output_gm_size = self.tiling_params[4]
        self.x_gm_size = self.tiling_params[5]
        self.y_grad_gm_size = self.tiling_params[6]
        self.target_gm_size = self.tiling_params[7]
        self.data_total_weight_size = self.tiling_params[8]
        self.weight_gm_size = self.tiling_params[9]
        self.big_weight = self.tiling_params[10]
        self.core_num = self.tiling_params[11]
        self.max_line = self.tiling_params[12]
        self.lower_line = self.tiling_params[13]
        self.loop_time = self.tiling_params[14]
        self.fake_core = self.tiling_params[15]
        self.redundant_line = self.tiling_params[16]
        self.max_total_num = self.tiling_params[17]
        self.lower_total_num = self.tiling_params[18]
        self.dup_ub_size = self.tiling_params[19]
        self.target_ub_size = self.tiling_params[20]
        self.weight_ub_size = self.tiling_params[21]
        self.total_weight_ub_size = self.tiling_params[22]
        self.refactor_weight_ub_size = self.tiling_params[23]
        self.weight_burst = self.tiling_params[24]
        self.target_burst = self.tiling_params[25]
        self.lower_target_burst = self.tiling_params[26]
        self.max_vmul_repeat = self.tiling_params[27]
        self.lower_vmul_repeat = self.tiling_params[28]
        self.last_target_burst = self.tiling_params[29]
        self.last_vmul_repeat = self.tiling_params[30]
        self.core_dup_repeat = self.tiling_params[31]
        self.last_dup_repeat = self.tiling_params[32]
        self.max_out_burst = self.tiling_params[33]
        self.last_out_burst = self.tiling_params[34]
        self.y_grad_ub_size = self.tiling_params[35]
        self.key = self.tiling_params[36]
        self.align_repeat_size = self.tiling_params[37]
        self.move_out_time = self.tiling_params[38]
        self.single_max_repeat = self.tiling_params[39]
        self.tail_repeat = self.tiling_params[40]
        self.offet = self.tiling_params[41]

    def get_tiling_params(self):
        """
        get tiling parameters function
        """
        ele_per_block = Constant.BLOCK_BYTE_SIZE // self.tiling_dtype_bytes
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
                                    Constant.TILING_CTRL_PARAM[1] // ele_per_block, 0, 0)
        for reg_idx in range(Constant.TILING_CTRL_PARAM[1]):
            self.tiling_params[reg_idx].set_as(self.tiling_ub[reg_idx])

    def init_ub(self):
        """
        init the ub of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.y_grad_ub = self.tik_instance.Tensor(self.dtype, [self.y_grad_ub_size],
                                                  name="y_grad_ub",
                                                  scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor(self.dtype_target, [self.target_ub_size],
                                                  name="target_ub",
                                                  scope=tik.scope_ubuf)
        self.weight_ub = self.tik_instance.Tensor(self.dtype, [self.weight_ub_size],
                                                  name="weight_ub",
                                                  scope=tik.scope_ubuf)
        self.target_value_ub = self.tik_instance.Tensor(self.dtype_target, [Constant.NUM_EIGHT],
                                                        name="target_value_ub",
                                                        scope=tik.scope_ubuf)
        self.refactor_weight_ub = self.tik_instance.Tensor(self.dtype, [self.refactor_weight_ub_size],
                                                           name="refactor_weight_ub",
                                                           scope=tik.scope_ubuf)
        self.total_weight_ub = self.tik_instance.Tensor(self.dtype, [Constant.NUM_SIXTYFOUR],
                                                        name="total_weight_ub",
                                                        scope=tik.scope_ubuf)
        self.dup_ub = self.tik_instance.Tensor(self.dtype, [self.dup_ub_size], name="dup_ub", scope=tik.scope_ubuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")

    def vector_dup_process(self, dup_up, repeat):
        """
        vector dup process.

        Parameters
        ----------

        Returns
        -------
        None
        """
        max_repeat_num = Constant.MAX_REPEAT * Constant.NUM_SIXTYFOUR
        max_repeat_loop = ceil_div(repeat, Constant.MAX_REPEAT)
        last_repeat = self.tik_instance.Scalar(dtype="int64",
                                               name="last_repeat",
                                               init_value=repeat % Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(last_repeat == 0):
            last_repeat.set_as(Constant.MAX_REPEAT)
        with self.tik_instance.for_range(0, max_repeat_loop - 1) as i:
            self.tik_instance.vector_dup(MASK64, dup_up[i * max_repeat_num], 0, Constant.MAX_REPEAT, 1,
                                         Constant.NUM_EIGHT)

        self.tik_instance.vector_dup(MASK64, dup_up[(max_repeat_loop - 1) * max_repeat_num], 0, last_repeat, 1,
                                     Constant.NUM_EIGHT)

    def select_valid_value(self, line_num, line_size, dst, src, target, dst_need_index=True, src_need_index=True):
        """
        select valid value with .

        Parameters
        ----------

        Returns
        -------
        None
        """
        vars_num = self.tik_instance.Scalar("int32", name="vars_num", init_value=line_num)
        loop_num = self.tik_instance.Scalar("int32", name="loop_num", init_value=0)
        last_line = self.tik_instance.Scalar("int32", name="last_line", init_value=line_num)
        tmp_target_value = self.tik_instance.Scalar(self.dtype_target, name="tmp_target_value", init_value=0)
        with self.tik_instance.if_scope(line_num >= 8):
            vars_num.set_as(8)
            loop_num.set_as(line_num // 8)
            last_line.set_as(line_num % 8)
        with self.tik_instance.for_range(0, loop_num) as time:
            offset_set = 8 * time
            with self.tik_instance.for_range(0, vars_num) as i:
                self.target_value_ub[i].set_as(target[offset_set + i])
            with self.tik_instance.for_range(0, vars_num) as i:
                tmp_target_value.set_as(self.target_value_ub[i])
                dst_offset = (offset_set + i) * line_size + tmp_target_value
                src_offset = self.tik_instance.Scalar("int32", name="src_offset", init_value=self.target_value_ub[i])
                if not dst_need_index:
                    dst_offset = (offset_set + i) * line_size
                if not src_need_index:
                    src_offset.set_as(offset_set + i)
                with self.tik_instance.if_scope(tik.all(tmp_target_value >= 0, tmp_target_value < self.c_dim)):
                    dst[dst_offset].set_as(src[src_offset])
        with self.tik_instance.for_range(0, last_line) as i:
            self.target_value_ub[i].set_as(target[loop_num * 8 + i])
        with self.tik_instance.for_range(0, last_line) as i:
            tmp_target_value.set_as(self.target_value_ub[i])
            dst_offset = (loop_num * 8 + i) * line_size + tmp_target_value
            src_offset = self.tik_instance.Scalar("int32", name="src_offset", init_value=self.target_value_ub[i])
            if not dst_need_index:
                dst_offset = (loop_num * 8 + i) * line_size
            if not src_need_index:
                src_offset.set_as(loop_num * 8 + i)
            with self.tik_instance.if_scope(tik.all(tmp_target_value >= 0, tmp_target_value < self.c_dim)):
                dst[dst_offset].set_as(src[src_offset])

    def _normal_two_tim_process(self, line_num, core_offset, repeat, burst, output_burst):
        """
        deal with two normal dims function
        """
        if self.reduction == "none":
            self.tik_instance.data_move(self.y_grad_ub, self.data_y_grad[core_offset], 0, 1, burst, 0, 0)
        self.tik_instance.data_move(self.target_ub, self.data_target[core_offset], 0, 1, burst, 0, 0)
        self.vector_dup_process(self.dup_ub, self.core_dup_repeat)
        if self.reduction == "mean":
            total_weight = self.tik_instance.Scalar(dtype="float32")
            total_weight.set_as(self.total_weight_ub[0])
            self.tik_instance.vector_dup(MASK64, self.total_weight_ub, total_weight, 1, 1, 8)
        with self.tik_instance.if_scope(tik.all(self.reduction == "none", self.c_dim != 1)):
            self.vector_dup_process(self.refactor_weight_ub,
                                    ceil_div(self.refactor_weight_ub_size, Constant.NUM_SIXTYFOUR))
            self.select_valid_value(line_num,
                                    1,
                                    self.refactor_weight_ub,
                                    self.weight_ub,
                                    self.target_ub,
                                    dst_need_index=False)
        with self.tik_instance.else_scope():
            self.select_valid_value(line_num, self.c_dim, self.dup_ub, self.weight_ub, self.target_ub)
        vmul_repeat_times = ceil_div(repeat, Constant.MAX_REPEAT)
        max_repeat_num = Constant.MAX_REPEAT * Constant.NUM_SIXTYFOUR
        last_vmul_offset = max_repeat_num
        last_vmul_repeat = self.tik_instance.Scalar("int64",
                                                    name="last_vmul_repeat",
                                                    init_value=repeat % Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(last_vmul_repeat == 0):
            last_vmul_repeat.set_as(Constant.MAX_REPEAT)
        with self.tik_instance.if_scope(tik.all(self.reduction == "none", self.c_dim != 1)):
            with self.tik_instance.for_range(0, vmul_repeat_times - 1) as i:
                self.compute_valid_value(self.refactor_weight_ub, self.y_grad_ub, i, max_repeat_num,
                                         Constant.MAX_REPEAT)
            self.compute_valid_value(self.refactor_weight_ub, self.y_grad_ub, vmul_repeat_times - 1, last_vmul_offset,
                                     last_vmul_repeat)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, vmul_repeat_times - 1) as i:
                self.compute_valid_value(self.dup_ub, self.y_grad_ub, i, max_repeat_num, Constant.MAX_REPEAT)
            self.compute_valid_value(self.dup_ub, self.y_grad_ub, vmul_repeat_times - 1, last_vmul_offset,
                                     last_vmul_repeat)
        with self.tik_instance.if_scope(tik.all(self.reduction == "none", self.c_dim != 1)):
            self.select_valid_value(line_num,
                                    self.c_dim,
                                    self.dup_ub,
                                    self.refactor_weight_ub,
                                    self.target_ub,
                                    dst_need_index=True,
                                    src_need_index=False)
        with self.tik_instance.if_scope(tik.any(line_num * self.c_dim % 8 == 0, self.core_num == 1)):
            self.tik_instance.data_move(self.output[core_offset * self.c_dim], self.dup_ub, 0, 1, output_burst, 8, 8)
        with self.tik_instance.else_scope():
            temp_out_ub = self.tik_instance.Tensor(self.dtype, [Constant.NUM_EIGHT],
                                                   name="temp_out_ub",
                                                   scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.output[core_offset * self.c_dim], self.dup_ub, 0, 1, output_burst - 1, 8,
                                        8)
            for i in range(0, 8):
                temp_out_ub[i].set_as(self.dup_ub[line_num * self.c_dim - 8 + i])
            self.tik_instance.data_move(self.output[(core_offset + line_num) * self.c_dim - 8], temp_out_ub, 0, 1, 1,
                                        8, 8)

    def compute_valid_value(self, dst, src, index, offset, repeat):
        """
        compute valid value.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.reduction != "none":
            scalar = self.tik_instance.Scalar(dtype="float32")
            scalar.set_as(src[0])
            self.tik_instance.vmuls(MASK64, dst[index * offset], dst[index * offset], scalar, repeat, 1, 1, 8, 8)
        else:
            self.tik_instance.vmul(MASK64, dst[index * offset], dst[index * offset], src[index * offset], repeat, 1, 1,
                                   1, Constant.NUM_EIGHT, Constant.NUM_EIGHT, Constant.NUM_EIGHT)
        self.tik_instance.vmuls(MASK64, dst[index * offset], dst[index * offset], Constant.NEGATIVE, repeat, 1, 1, 8,
                                8)
        if self.reduction == "mean":
            if tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
                self.tik_instance.vdiv(MASK64, dst[index * offset], dst[index * offset], self.total_weight_ub, repeat,
                                       1, 1, 1, 8, 8, 0)
            else:
                self.tik_instance.vrec(MASK64, dst[index * offset], dst[index * offset], repeat, 1, 1, 8, 8)
                self.tik_instance.vmul(MASK64, dst[index * offset], dst[index * offset], self.total_weight_ub, repeat,
                                       1, 1, 1, 8, 8, 0)

    def normal_two_dim_compute(self, cycle):
        """
        calculate process of normal two dim.

        Parameters
        ----------

        Returns
        -------
        None
        """
        core_offset = cycle * self.max_line
        lower_core_offset = cycle * self.lower_line + self.redundant_line
        self.tik_instance.data_move(self.y_grad_ub, self.data_x, 0, 1, 1, 0, 0)
        if self.reduction != "none":
            self.tik_instance.data_move(self.total_weight_ub, self.data_total_weight, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.y_grad_ub, self.data_y_grad, 0, 1, 1, 0, 0)

        self.tik_instance.data_move(self.weight_ub, self.data_weight, 0, 1, self.weight_burst, 0, 0)
        with self.tik_instance.for_range(0, self.loop_time) as loop:
            loop_offset = loop * self.max_line * self.core_num
            compute_max_repeat = self.tik_instance.Scalar("int32",
                                                          name="compute_max_repeat",
                                                          init_value=self.core_dup_repeat)
            compute_last_repeat = self.tik_instance.Scalar("int32",
                                                           name="compute_last_repeat",
                                                           init_value=self.last_dup_repeat)
            with self.tik_instance.if_scope(tik.all(self.reduction == "none", self.c_dim != 1)):
                compute_max_repeat.set_as(self.max_vmul_repeat)
                compute_last_repeat.set_as(self.last_vmul_repeat)

            with self.tik_instance.if_scope(self.loop_time == 1):
                with self.tik_instance.if_scope(cycle < self.redundant_line):
                    self._normal_two_tim_process(self.max_line, core_offset, compute_max_repeat, self.target_burst,
                                                 self.max_out_burst)
                with self.tik_instance.else_scope():
                    self._normal_two_tim_process(self.lower_line, lower_core_offset, compute_last_repeat,
                                                 self.lower_target_burst, self.last_out_burst)
            with self.tik_instance.if_scope(self.loop_time > 1):
                with self.tik_instance.if_scope(loop * self.core_num + cycle < self.fake_core - 1):
                    self._normal_two_tim_process(self.max_line, loop_offset + core_offset, compute_max_repeat,
                                                 self.target_burst, self.max_out_burst)
                with self.tik_instance.if_scope(loop * self.core_num + cycle == self.fake_core - 1):
                    self._normal_two_tim_process(self.lower_line, loop_offset + core_offset, compute_last_repeat,
                                                 self.last_target_burst, self.last_out_burst)

    def tail_block_refactor(self, dst, src, valid_value, index, burst, start, offset):
        """
        refactor tail block.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.tik_instance.data_move(dst[offset], src, 0, 1, burst - 1, 0, 0)
        temp_out_ub = self.tik_instance.Tensor(self.dtype, [Constant.NUM_EIGHT],
                                               name="temp_out_ub",
                                               scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(8, temp_out_ub, 0, 1, 1, 0)

        with self.tik_instance.if_scope(tik.any(index < 0, index >= self.c_dim)):
            self.tik_instance.data_move(dst[start + self.c_dim - 8], temp_out_ub, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(index > self.c_dim - 8):
                temp_out_ub[index - self.c_dim + 8].set_as(valid_value[0])
                self.tik_instance.data_move(dst[start + self.c_dim - 8], temp_out_ub, 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(dst[start + self.c_dim - 8], temp_out_ub, 0, 1, 1, 0, 0)
                temp_out_ub[0].set_as(valid_value[0])
                self.tik_instance.data_move(dst[start + index], temp_out_ub, 0, 1, 1, 0, 0)

    def two_dim_with_big_weight_compute(self, cycle):
        """
        calculate process when x is 2D and the shape of weight
        is big.

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.move_out_time > 1):
            self.vector_dup_process(self.dup_ub, self.single_max_repeat)
        with self.tik_instance.else_scope():
            self.vector_dup_process(self.dup_ub, self.tail_repeat)
        with self.tik_instance.for_range(0, self.loop_time) as loop:
            line_num = cycle + loop * self.core_num
            with self.tik_instance.if_scope(line_num < self.n_dim):
                if self.reduction == "none":
                    self.tik_instance.data_move(self.y_grad_ub, self.data_y_grad[line_num], 0, 1, 1, 0, 0)
                else:
                    self.tik_instance.data_move(self.y_grad_ub, self.data_y_grad, 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.target_ub, self.data_target[line_num], 0, 1, 1, 0, 0)
                self.index_x.set_as(self.target_ub[0])
                with self.tik_instance.if_scope(tik.any(self.index_x < 0, self.index_x >= self.c_dim)):
                    self.tik_instance.vector_dup(8, self.weight_ub, 0, 1, 1, 8)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(self.weight_ub, self.data_weight[self.index_x], 0, 1, 1, 0, 0)

                self.tik_instance.data_move(self.total_weight_ub, self.data_total_weight, 0, 1, 1, 0, 0)
                if self.reduction == "mean":
                    if tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
                        self.tik_instance.vdiv(1, self.weight_ub, self.weight_ub, self.total_weight_ub, 1, 1, 1, 1,
                                               Constant.NUM_EIGHT, Constant.NUM_EIGHT, Constant.NUM_EIGHT)
                    else:
                        self.tik_instance.vrec(1, self.weight_ub, self.weight_ub, 1, 1, 1, 8, 8)
                        self.tik_instance.vmul(1, self.weight_ub, self.weight_ub, self.total_weight_ub, 1, 1, 1, 1,
                                               Constant.NUM_EIGHT, Constant.NUM_EIGHT, Constant.NUM_EIGHT)
                self.tik_instance.vmul(1, self.weight_ub, self.weight_ub, self.y_grad_ub, 1, 1, 1, 1,
                                       Constant.NUM_EIGHT, Constant.NUM_EIGHT, Constant.NUM_EIGHT)
                self.tik_instance.vmuls(1, self.weight_ub, self.weight_ub, Constant.NEGATIVE, 1, 1, 1,
                                        Constant.NUM_EIGHT, Constant.NUM_EIGHT)

                with self.tik_instance.for_range(0, self.move_out_time) as time:
                    out_put_offset = line_num * self.c_dim + time * self.offet
                    with self.tik_instance.if_scope(time < self.move_out_time - 1):
                        self.tik_instance.data_move(self.output[out_put_offset], self.dup_ub, 0, 1, self.max_out_burst,
                                                    0, 0)
                    with self.tik_instance.else_scope():
                        self.tail_block_refactor(self.output, self.dup_ub, self.weight_ub, self.index_x,
                                                 self.last_out_burst, line_num * self.c_dim, out_put_offset)

    def nll_loss_compute_start(self):
        """
        Different calculation methods

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.real_core_num, block_num=self.real_core_num) as cycle:
            with self.tik_instance.if_scope(cycle < self.core_num):
                self.init_ub()
                with self.tik_instance.if_scope(self.key == 2000):
                    with self.tik_instance.new_stmt_scope():
                        self.normal_two_dim_compute(cycle)
                with self.tik_instance.if_scope(self.key == 2001):
                    with self.tik_instance.new_stmt_scope():
                        self.two_dim_with_big_weight_compute(cycle)
        input_list = [self.data_x, self.data_y_grad, self.data_target, self.data_weight, self.data_total_weight]
        tbe_context.get_context().add_compile_info("vars", {"block_dim": self.real_core_num, "ub_size": self.ub_size,
                                                            "reduction": self.reduction})

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=input_list, outputs=[self.output],
                                   flowtable=[self.tiling_gm])
        return self.tik_instance


@register_operator("NLLLossGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def nll_loss_grad(x,
                  y_grad,
                  target,
                  weight,
                  total_weight,
                  x_grad,
                  reduction="mean",
                  ignore_index=-100,
                  kernel_name="nll_loss_grad"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input, the length of shape should be two or one.
    y_grad : dict
        shape and dtype of input, the length of shape must be one.
    target : dict
        shape and dtype of input, the length of shape only support one.
    total_weight : dict
        shape and dtype of input, it is a scalar.
    weight : dict or None
        the length of shape only support one when weight is dict.
    x_grad: dict
        It's a tensor with shape(minibatch, ) when reduction == 'none' and
        the input is 2D. Otherwise, the output is a scalar.
    reduction: str
        default value is "mean"
    ignore_index: int
        default value is -100
    kernel_name : str
        kernel name, default value is "nll_loss_grad"

    Returns
    -------
    None
    """
    _shape_and_dtype_check(x, y_grad, target, weight, total_weight, reduction, x_grad)
    nll_loss_function = NllLossGradCompute(x, target, reduction, ignore_index, kernel_name)
    return nll_loss_function.nll_loss_compute_start()
