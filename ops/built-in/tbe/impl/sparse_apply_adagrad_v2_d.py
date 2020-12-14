# Copyright 2019 Huawei Technologies Co., Ltd
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
sparse_apply_adagrad_v2_d
"""
from impl.sparse_apply_common import SparseApply
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


class SparseApplyAdagrad(SparseApply):
    """
    Sub class inherited form SparseApply for sparse_apply_adagrad op
    """

    # pylint: disable=invalid-name,too-many-arguments
    def __init__(self, var, accum, grad, indices, lr, epsilon, update_slot, kernel_name):
        """
        init sparse_apply_adagrad  base parameters

        Parameters
        ----------
        var: dict
        accum: dict
        lr: float
            scalar
        epsilon: float
            scalar
        update_slot: bool

        Returns
        -------
        None
        """
        super().__init__(var, grad, indices, kernel_name)
        self.lr = lr
        self.epsilon = epsilon
        self.update_slot = update_slot
        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()
        self.accum_shape = accum.get("shape")
        self.accum_dtype = accum.get("dtype").lower()
        self._check_param()

    def _check_param(self):
        """
        Check parameter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        para_check.check_shape(self.var_shape, param_name="var")
        para_check.check_shape(self.accum_shape, param_name="accum")

        para_check.check_dtype(self.var_dtype, ("float32", ), param_name="var")
        para_check.check_dtype(self.accum_dtype, ("float32", ), param_name="accum")

        if self.accum_shape != self.var_shape:
            error_manager_vector.raise_err_inputs_shape_not_equal("sparse_apply_adagrad_v2_d", "accum", "var",
                                                                  self.accum_shape, self.var_shape, self.var_shape)

    def _calculate(self, repeat_times, mask, offset):
        """
        calculate data according to the adagrad scheme will automated called by basic class function

        Parameters
        ----------
        repeat_times: repeat times of insn
        mask: mask for vector insn
        offset: offset of ub addr

        Returns
        -------
        None
        """
        tmp_ub = self._get_ub("tmp_ub")[offset]

        if self.each_row_data_num <= self.cache_threshold_col:
            var_ub = self._get_ub("var_align_ub")[offset]
            accum_ub = self._get_ub("accum_align_ub")[offset]
            grad_ub = self.grad_align_ub[offset]
        else:
            var_ub = self._get_ub("var_ub")[offset]
            grad_ub = self.grad_ub[offset]
            accum_ub = self._get_ub("accum_ub")[offset]

        if self.update_slot:
            self.tik_instance.vmul(mask, tmp_ub, grad_ub, grad_ub, repeat_times, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadd(mask, accum_ub, accum_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsqrt(mask, tmp_ub, accum_ub, repeat_times, 1, 1, 8, 8)

        if self.epsilon != 0:
            self.tik_instance.vadds(mask, tmp_ub, tmp_ub, self.epsilon, repeat_times, 1, 1, 8, 8)

        self.tik_instance.vdiv(mask, tmp_ub, grad_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, tmp_ub, tmp_ub, self.lr, repeat_times, 1, 1, 8, 8)
        self.tik_instance.vsub(mask, var_ub, var_ub, tmp_ub, repeat_times, 1, 1, 1, 8, 8, 8)


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_INT),
                            (para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_INT), para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_adagrad_v2_d(var,
                              accum,
                              grad,
                              indices,
                              var_out,
                              accum_out,
                              lr,
                              epsilon,
                              use_locking=False,
                              update_slots=True,
                              kernel_name="sparse_apply_adagrad_v2_d"):
    """
    Adds sparse updates to the variable referenced by resource.

    Parameters
    ----------
    var: dict
        data of input.
        source data type, support  "float32"
    accum: dict
        data of input.
        source data type, support "float32"
    grad: dict
        data of input
        source data type should ne same as var
    indices: dict
         A tensor of indices into var, support "int32"
    out: dict
        data of output
    lr: float
        scalar
    use_locking: bool
        not used in this compute
    kernel_name: str
        kernel name, default value is "sparse_apply_adagrad_v2_d"

    Returns:
    None
    """
    apply_adagrad = SparseApplyAdagrad(var, accum, grad, indices, lr, epsilon, update_slots, kernel_name)
    var_shape = var.get('shape')

    apply_adagrad.add_input("var_in_gm", "float32", var_shape)
    apply_adagrad.add_input("accum_in_gm", "float32", var_shape)
    apply_adagrad.add_output("var_out_gm", "float32", var_shape)
    apply_adagrad.add_output("accum_out_gm", "float32", var_shape)
    apply_adagrad.reserve_ub("var_ub", "float32", "var_align_ub")
    apply_adagrad.reserve_ub("accum_ub", "float32", "accum_align_ub")
    apply_adagrad.reserve_ub("tmp_ub", "float32")
    apply_adagrad.set_var_rows(var_shape[0])
    apply_adagrad.sparse_apply_operator()
