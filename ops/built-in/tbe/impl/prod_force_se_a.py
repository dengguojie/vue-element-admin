"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

prod_force_se_a
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max data num one repeat for dtype float32
    MASK_FLOAT32 = 64
    # max data num one block for dytpe float32
    BLOCK_FLOAT32 = 8


class ProdForceSeA:
    """Function: use to calc the force to those local atoms for all frames
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, net_deriv, in_deriv, nlist, natoms, nnei, kernel_name):
        """
        init ProdForceSeA.

        Parameters
        ----------
        net_deriv : dict. shape and dtype of input data net_deriv
        in_deriv : dict. shape and dtype of input data in_deriv
        nlist : dict. shape and dtype of input data nlist
        natoms : dict. shape and dtype of input data natoms
        nnei : value of neighbour
        kernel_name : str. cce kernel name

        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.tik_inst = tik.Tik(tik.Dprofile)

        self.op_data_type = net_deriv.get("dtype").lower()
        self.nlist_dtype = nlist.get("dtype").lower()
        self.natoms_dtype = natoms.get("dtype").lower()

        net_deriv_shape = net_deriv.get("shape")
        in_deriv_shape = in_deriv.get("shape")
        nlist_shape = nlist.get("shape")

        self.nframes = net_deriv_shape[0]
        self.nnei = nnei
        self.nloc = nlist_shape[1] // self.nnei
        self.nall = 28328
        self.ndescrpt = self.nnei * 4
        self.natoms_shape = natoms.get("shape")

        if any([self.nframes != in_deriv_shape[0], self.nframes != nlist_shape[0]]):
            rule = "shape[0] of {net_deriv, in_deriv, nlist} should match"
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, rule)

        if self.nloc * self.ndescrpt * 3 != in_deriv_shape[1]:
            rule = "shape[1] of in_deriv should be equal to nloc * ndescrpt * 3"
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, rule)

        if self.ndescrpt != 4 * self.nnei:
            rule = "ndescrpt should be equal to 4 * nnei"
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, rule)

        self.nei_total = self.nframes * self.nloc * self.nnei
        self.nei_burst = 768
        self.nei_rep_times = self.nei_total // self.nei_burst
        self.nei_tail = 0

        if self.nei_rep_times > 0:
            self.nei_tail = self.nei_total % self.nei_rep_times

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.pre_core_num = self.nei_rep_times % self.ai_core_num
        self.post_core_num = self.ai_core_num - self.pre_core_num
        self.nei_rep_times_pre_core = (self.nei_rep_times + self.ai_core_num - 1) // self.ai_core_num
        self.nei_rep_times_post_core = self.nei_rep_times // self.ai_core_num

    def _init_gm_data_fp32(self):
        """
        init gm data
        """
        self.net_deriv_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * self.nloc * self.ndescrpt,),
                                                 name="net_deriv_gm", scope=tik.scope_gm)
        self.in_deriv_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * self.nloc * self.ndescrpt * 3,),
                                                name="in_deriv_gm", scope=tik.scope_gm)
        self.nlist_gm = self.tik_inst.Tensor(self.nlist_dtype, (self.nframes * self.nloc * self.nnei,), name="nlist_gm",
                                             scope=tik.scope_gm)
        self.natoms_gm = self.tik_inst.Tensor(self.natoms_dtype, (self.natoms_shape[0],), name="natoms_gm",
                                              scope=tik.scope_gm)
        self.force_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * self.nall * 3,), name="force_gm",
                                             scope=tik.scope_gm, is_atomic_add=True)

    # 'pylint: disable=too-many-locals
    def _init_ub_data_fp32(self):
        """
        init ub data
        """
        op_ub_shape = (self.nei_burst * 4 * 3,)
        net_ub = self.tik_inst.Tensor(self.op_data_type, op_ub_shape, name="net_deriv_ub",
                                      scope=tik.scope_ubuf)
        in_ub = self.tik_inst.Tensor(self.op_data_type, op_ub_shape, name="in_deriv_ub",
                                     scope=tik.scope_ubuf)
        nlist_ub = self.tik_inst.Tensor(self.nlist_dtype, (self.nei_burst,), name="nlist_ub",
                                        scope=tik.scope_ubuf)
        trans_ub = self.tik_inst.Tensor(self.op_data_type, op_ub_shape, name="trans_ub",
                                        scope=tik.scope_ubuf)

        zero_scalar = self.tik_inst.Scalar(init_value=0, dtype=self.op_data_type)
        j_idx0 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx1 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx2 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx3 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx4 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx5 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx6 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx7 = self.tik_inst.Scalar(dtype=self.nlist_dtype)

        res = (net_ub, in_ub, nlist_ub, trans_ub, zero_scalar, j_idx0, j_idx1, j_idx2, j_idx3,
               j_idx4, j_idx5, j_idx6, j_idx7)

        return res

    # 'pylint: disable=too-many-statements,too-many-locals
    def _compute_force_fp32(self, nn, ub_tuple):
        """
        Support shape/datatype:
        type  param name   shape form                      enable shape          dtype     shape in ub
        in    net_deriv    (nframes, nloc * ndescrpt)      (_, _ * 768 * 4)      float32   (768, 4)
        in    in_deriv     (nframes, nloc * ndescrpt * 3)  (_, _ * 768 * 4 * 3)  float32   (768, 4, 3)
        in    nlist        (nframes, nloc * nnei)          (_, _ * 768)          int32     (768, )
        in    natoms       (2 + ntypes, )                  (4)                   int32
        out   force        (nframes, nall * 3)             (_, 3)                float32   (3, )
        """
        (net_ub, in_ub, nlist_ub, trans_ub, zero_scalar, j_idx0, j_idx1, j_idx2, j_idx3,
         j_idx4, j_idx5, j_idx6, j_idx7) = ub_tuple

        net_idx = nn * self.nei_burst * 4
        self.tik_inst.data_move(net_ub, self.net_deriv_gm[net_idx], 0, 1, self.nei_burst * 4 // Constant.BLOCK_FLOAT32,
                                0, 0)
        self.tik_inst.data_move(net_ub[self.nei_burst * 4], net_ub, 0, 1, self.nei_burst * 4 // Constant.BLOCK_FLOAT32,
                                0, 0)
        self.tik_inst.data_move(net_ub[self.nei_burst * 4 * 2], net_ub, 0, 1,
                                self.nei_burst * 4 // Constant.BLOCK_FLOAT32, 0, 0)

        in_idx = nn * self.nei_burst * 4 * 3
        self.tik_inst.data_move(trans_ub, self.in_deriv_gm[in_idx], 0, 1,
                                self.nei_burst * 4 * 3 // Constant.BLOCK_FLOAT32, 0, 0)
        self.tik_inst.v4dtrans(False, in_ub, trans_ub, self.nei_burst * 4, 3)

        nlist_idx = nn * self.nei_burst
        self.tik_inst.data_move(nlist_ub, self.nlist_gm[nlist_idx], 0, 1, self.nei_burst // Constant.BLOCK_FLOAT32, 0,
                                0)

        self.tik_inst.vmul(Constant.MASK_FLOAT32, net_ub, net_ub, in_ub,
                           self.nei_burst * 4 * 3 // Constant.MASK_FLOAT32, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vec_dup(Constant.MASK_FLOAT32, in_ub, zero_scalar, self.nei_burst * 8 // Constant.MASK_FLOAT32, 8)
        self.tik_inst.vcpadd(Constant.MASK_FLOAT32, net_ub, net_ub, self.nei_burst * 4 * 3 // Constant.MASK_FLOAT32, 1,
                             1, 8)
        self.tik_inst.vcpadd(Constant.MASK_FLOAT32, in_ub, net_ub, self.nei_burst * 4 * 3 // Constant.MASK_FLOAT32 // 2,
                             1, 1, 8)

        self.tik_inst.v4dtrans(True, net_ub, in_ub, self.nei_burst, 8)
        self.tik_inst.vmuls(Constant.MASK_FLOAT32, trans_ub, net_ub, -1, (self.nei_burst * 8) // Constant.MASK_FLOAT32,
                            1, 1, 8, 8)

        nframes = nn * self.nei_burst // (self.nloc * self.nnei)
        frame_offset = nframes * self.nall * 3
        burst_offset = nn * self.nei_burst

        self.tik_inst.vmuls(Constant.MASK_FLOAT32, nlist_ub, nlist_ub, 3, self.nei_burst // Constant.MASK_FLOAT32, 1, 1,
                            8, 8)
        self.tik_inst.vadds(Constant.MASK_FLOAT32, nlist_ub, nlist_ub, frame_offset,
                            self.nei_burst // Constant.MASK_FLOAT32, 1, 1, 8, 8)

        with self.tik_inst.for_range(0, self.nei_burst // 8, name="ii") as ii:
            j_idx0.set_as(nlist_ub[ii * 8])
            j_idx1.set_as(nlist_ub[ii * 8 + 1])
            j_idx2.set_as(nlist_ub[ii * 8 + 2])
            j_idx3.set_as(nlist_ub[ii * 8 + 3])
            j_idx4.set_as(nlist_ub[ii * 8 + 4])
            j_idx5.set_as(nlist_ub[ii * 8 + 5])
            j_idx6.set_as(nlist_ub[ii * 8 + 6])
            j_idx7.set_as(nlist_ub[ii * 8 + 7])

            batch_offset = ii * 64

            self.tik_inst.set_atomic_add(1)

            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8) // self.nnei * 3],
                                    trans_ub[batch_offset], 0, 1, 1, 0, 0)
            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8 + 1) // self.nnei * 3],
                                    trans_ub[batch_offset + 8], 0, 1, 1, 0, 0)
            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8 + 2) // self.nnei * 3],
                                    trans_ub[batch_offset + 16], 0, 1, 1, 0, 0)
            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8 + 3) // self.nnei * 3],
                                    trans_ub[batch_offset + 24], 0, 1, 1, 0, 0)
            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8 + 4) // self.nnei * 3],
                                    trans_ub[batch_offset + 32], 0, 1, 1, 0, 0)
            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8 + 5) // self.nnei * 3],
                                    trans_ub[batch_offset + 40], 0, 1, 1, 0, 0)
            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8 + 6) // self.nnei * 3],
                                    trans_ub[batch_offset + 48], 0, 1, 1, 0, 0)
            self.tik_inst.data_move(self.force_gm[frame_offset + (burst_offset + ii * 8 + 7) // self.nnei * 3],
                                    trans_ub[batch_offset + 56], 0, 1, 1, 0, 0)

            with self.tik_inst.if_scope(j_idx0 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx0], net_ub[batch_offset], 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(j_idx1 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx1], net_ub[batch_offset + 8], 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(j_idx2 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx2], net_ub[batch_offset + 16], 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(j_idx3 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx3], net_ub[batch_offset + 24], 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(j_idx4 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx4], net_ub[batch_offset + 32], 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(j_idx5 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx5], net_ub[batch_offset + 40], 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(j_idx6 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx6], net_ub[batch_offset + 48], 0, 1, 1, 0, 0)
            with self.tik_inst.if_scope(j_idx7 >= frame_offset):
                self.tik_inst.data_move(self.force_gm[j_idx7], net_ub[batch_offset + 56], 0, 1, 1, 0, 0)

            self.tik_inst.set_atomic_add(0)

    def _compute_db_fp32(self, nnei_start, nnei_end):
        """
        compute with double buffer
        """
        with self.tik_inst.if_scope(nnei_end == nnei_start + 1):
            ub_tuple = self._init_ub_data_fp32()
            self._compute_force_fp32(nnei_start, ub_tuple)
        with self.tik_inst.elif_scope(nnei_end > nnei_start + 1):
            with self.tik_inst.for_range(nnei_start, nnei_end, thread_num=2) as nn:
                ub_tuple = self._init_ub_data_fp32()
                self._compute_force_fp32(nn, ub_tuple)

    def _compute_fp32(self):
        """
        compute with multiple ai core
        """
        self._init_gm_data_fp32()

        nnei_start = self.tik_inst.Scalar(init_value=0, dtype="int32")
        nnei_end = self.tik_inst.Scalar(init_value=0, dtype="int32")

        with self.tik_inst.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as block_i:
            with self.tik_inst.if_scope(block_i < self.pre_core_num):
                nnei_start.set_as(block_i * self.nei_rep_times_pre_core)
                nnei_end.set_as((block_i + 1) * self.nei_rep_times_pre_core)
            with self.tik_inst.else_scope():
                nnei_start.set_as(self.pre_core_num + block_i * self.nei_rep_times_post_core)
                nnei_end.set_as(self.pre_core_num + (block_i + 1) * self.nei_rep_times_post_core)

            self._compute_db_fp32(nnei_start, nnei_end)

    def compute(self):
        """
        compute
        """
        if self.op_data_type == "float32":
            self._compute_fp32()

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.net_deriv_gm, self.in_deriv_gm, self.nlist_gm, self.natoms_gm],
                               outputs=[self.force_gm])


# 'pylint: disable=too-many-locals,too-many-arguments
def _check_params(net_deriv, in_deriv, nlist, natoms, force, n_a_sel, n_r_sel, kernel_name):
    net_deriv_dtype = net_deriv.get("dtype").lower()
    para_check.check_dtype(net_deriv_dtype, ("float16", "float32", "float64"), param_name="net_deriv")

    in_deriv_dtype = in_deriv.get("dtype").lower()
    para_check.check_dtype(in_deriv_dtype, ("float16", "float32", "float64"), param_name="in_deriv")

    nlist_dtype = nlist.get("dtype").lower()
    para_check.check_dtype(nlist_dtype, ("int32"), param_name="nlist")

    natoms_dtype = natoms.get("dtype").lower()
    para_check.check_dtype(natoms_dtype, ("int32"), param_name="natoms")

    force_dtype = force.get("dtype").lower()
    para_check.check_dtype(force_dtype, ("float16", "float32", "float64"), param_name="force")

    if net_deriv_dtype != in_deriv_dtype:
        rule = "Data type of {net_deriv, in_deriv} is not match."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule)

    net_deriv_shape = net_deriv.get("shape")
    para_check.check_shape(net_deriv_shape, min_rank=2, max_rank=2, param_name="net_deriv")

    in_deriv_shape = in_deriv.get("shape")
    para_check.check_shape(in_deriv_shape, min_rank=2, max_rank=2, param_name="in_deriv")

    nlist_shape = nlist.get("shape")
    para_check.check_shape(nlist_shape, min_rank=2, max_rank=2, param_name="nlist")

    natoms_shape = natoms.get("shape")
    para_check.check_shape(natoms_shape, min_rank=1, max_rank=1, min_size=3, param_name="natoms")

    if any([n_a_sel < 0, n_r_sel < 0, n_a_sel + n_r_sel <= 0]):
        rule = "The attributes {n_r_sel, n_r_sel} can not be minus value or both 0."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule)


# 'pylint: disable=too-many-arguments
@register_operator("ProdForceSeA")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def prod_force_se_a(net_deriv, in_deriv, nlist, natoms, force, n_a_sel, n_r_sel, kernel_name="prod_force_se_a"):
    """
    Compute ProdForceSeA.

    Parameters
    ----------
    net_deriv : dict. shape and dtype of input data net_deriv
    in_deriv : dict. shape and dtype of input data in_deriv
    nlist : dict. shape and dtype of input data nlist
    natoms : dict. shape and dtype of input data natoms
    force : dict. shape and dtype of output data force
    n_a_sel : value of attr n_a_sel
    n_r_sel : value of attr n_r_sel
    kernel_name : str. cce kernel name, default value is "prod_force_se_a"

    Returns
    -------
    None
    """
    _check_params(net_deriv, in_deriv, nlist, natoms, force, n_a_sel, n_r_sel, kernel_name)

    obj = ProdForceSeA(net_deriv, in_deriv, nlist, natoms, n_a_sel + n_r_sel, kernel_name)
    obj.compute()
