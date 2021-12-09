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

prod_virial_se_a
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik


# 'pylint: disable=too-few-public-methods
class ProdVirialSeA:
    """
    ProdVirialSeA compute
    """
    MAX_NNEI_IN_UB = 256

    # 'pylint: disable=too-many-arguments
    def __init__(self, net_deriv, in_deriv, rij, nlist, natoms, nnei, nall, kernel_name):
        self.kernel_name = kernel_name
        self.tik_inst = tik.Tik(tik.Dprofile)

        self.op_data_type = net_deriv.get("dtype").lower()
        self.nlist_dtype = nlist.get("dtype").lower()
        self.natoms_dtype = natoms.get("dtype").lower()

        net_deriv_shape = net_deriv.get("shape")
        in_deriv_shape = in_deriv.get("shape")
        rij_shape = rij.get("shape")
        nlist_shape = nlist.get("shape")
        natoms_shape = natoms.get("shape")

        self.nframes = net_deriv_shape[0]
        self.nnei = nnei
        self.nloc = nlist_shape[1] // self.nnei
        self.nall = nall
        self.ndescrpt = self.nnei * 4

        if any((self.nframes != in_deriv_shape[0], self.nframes != rij_shape[0], self.nframes != nlist_shape[0])):
            rule = "Number of samples should match"
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, rule)

        if self.nloc * self.ndescrpt * 3 != in_deriv_shape[1]:
            rule = "number of descriptors should match"
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, rule)

        if self.nloc * self.nnei * 3 != rij_shape[1]:
            rule = "dim of rij should be nnei * 3"
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, rule)

        if self.ndescrpt != 4 * self.nnei:
            rule = "ndescrpt should be 4 * nnei"
            error_manager_vector.raise_err_check_params_rules(self.kernel_name, rule)

        self.nei_total = self.nframes * self.nloc * self.nnei
        self.nei_burst = ProdVirialSeA.MAX_NNEI_IN_UB
        self.nei_rep_times = self.nei_total // self.nei_burst
        self.nei_tail = 0
        if self.nei_rep_times > 0:
            self.nei_tail = self.nei_total % self.nei_rep_times

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.pre_core_num = self.nei_rep_times % self.ai_core_num
        self.post_core_num = self.ai_core_num - self.pre_core_num
        self.nei_rep_times_pre_core = (self.nei_rep_times + self.ai_core_num - 1) // self.ai_core_num
        self.nei_rep_times_post_core = self.nei_rep_times // self.ai_core_num

        self.net_deriv_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * self.nloc * self.nnei * 4,),
                                                 name="net_deriv_gm", scope=tik.scope_gm)
        self.in_deriv_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * self.nloc * self.nnei * 4 * 3,),
                                                name="in_deriv_gm", scope=tik.scope_gm)
        self.rij_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * self.nloc * self.nnei * 3,),
                                           name="rij_gm", scope=tik.scope_gm)
        self.nlist_gm = self.tik_inst.Tensor(self.nlist_dtype, (self.nframes * self.nloc * self.nnei,),
                                             name="nlist_gm", scope=tik.scope_gm)
        self.natoms_gm = self.tik_inst.Tensor(self.natoms_dtype, (natoms_shape[0],), name="natoms_gm",
                                              scope=tik.scope_gm)
        self.virial_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * 9,), name="virial_gm",
                                              scope=tik.scope_gm, is_atomic_add=True)
        self.atom_virial_gm = self.tik_inst.Tensor(self.op_data_type, (self.nframes * self.nall * 9,),
                                                   name="atom_virial_gm", scope=tik.scope_gm, is_atomic_add=True)

    def _init_ub_data_fp32(self):
        """
        init ub data fp32
        """
        net_ub = self.tik_inst.Tensor(self.op_data_type, (self.nei_burst * 4,), name="net_ub",
                                           scope=tik.scope_ubuf)
        drv_ub = self.tik_inst.Tensor(self.op_data_type, (self.nei_burst * 4 * 4,), name="drv_ub",
                                           scope=tik.scope_ubuf)
        nlist_ub = self.tik_inst.Tensor(self.nlist_dtype, (self.nei_burst,), name="nlist_ub",
                                             scope=tik.scope_ubuf)

        op_ub_shape = (self.nei_burst * 4 * 3 * 3,)
        trans_ub = self.tik_inst.Tensor(self.op_data_type, op_ub_shape, name="trans_ub", scope=tik.scope_ubuf)
        tmpv_ub = self.tik_inst.Tensor(self.op_data_type, op_ub_shape, name="tmpv_ub", scope=tik.scope_ubuf)

        zero_scalar = self.tik_inst.Scalar(init_value=0, dtype=self.op_data_type)
        j_idx0 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx1 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx2 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx3 = self.tik_inst.Scalar(dtype=self.nlist_dtype)

        return net_ub, drv_ub, nlist_ub, trans_ub, tmpv_ub, zero_scalar, j_idx0, j_idx1, j_idx2, j_idx3

    # 'pylint: disable=too-many-locals,too-many-statements
    def _compute_virial_fp32(self, nn, ub_tuple):
        """
        Support shape/datatype:
        type  param name   shape form                      enable shape          dtype     shape in ub
        in    net_deriv    (nframes, nloc * ndescrpt)      (_, _ * 256 * 4)      float32   (256, 4)
        in    in_deriv     (nframes, nloc * ndescrpt * 3)  (_, _ * 256 * 4 * 3)  float32   (256, 4, 3)
        in    rij          (nframes, nloc * nnei * 3)      (_, _ * 256 * 3)      float32   (256, 3)
        in    nlist        (nframes, nloc * nnei)          (_, _ * 256)          int32     (256, )
        in    natoms       (2 + ntypes, )                  (3)                   int32
        out   virial       (nframes, 9)                    (_, 9)                float32   (9, )
        out   atom_virial  (nframes, nall * 9)             (_, _ * 9)            float32   (144, )
        """
        (net_ub, drv_ub, nlist_ub, trans_ub, tmpv_ub, zero_scalar, j_idx0, j_idx1, j_idx2, j_idx3) = ub_tuple

        net_idx = nn * self.nei_burst * 4
        self.tik_inst.data_move(trans_ub, self.net_deriv_gm[net_idx], 0, 1, 128, 0, 0)  # net_deriv -> (256, 4)
        self.tik_inst.v4dtrans(False, net_ub, trans_ub, 256, 4)                         # net_deriv -> (1, 4, 256)

        rij_idx = nn * self.nei_burst * 3
        self.tik_inst.data_move(trans_ub, self.rij_gm[rij_idx], 0, 1, 96, 0, 0)         # rij -> (256, 3)
        self.tik_inst.v4dtrans(False, drv_ub, trans_ub, 256, 3)                         # rij -> (3, 1, 256)

        with self.tik_inst.for_range(0, 3, name="rij_row") as rij_row:
            with self.tik_inst.for_range(0, 4, name="net_row") as net_row:
                self.tik_inst.vmul(64, trans_ub[(rij_row * 4 + net_row) * 256], drv_ub[rij_row * 256],
                                   net_ub[net_row * 256], 4, 1, 1, 1, 8, 8, 8)          # tmpv -> (1, 3, 4, 256)

        in_idx = nn * self.nei_burst * 4 * 3
        self.tik_inst.data_move(tmpv_ub, self.in_deriv_gm[in_idx], 0, 1, 384, 0, 0)     # in_deriv -> (256, 4, 3)
        self.tik_inst.v4dtrans(False, drv_ub, tmpv_ub, 256, 12)                         # in_deriv -> (4, 3, 256)

        with self.tik_inst.for_range(0, 3, name="mul_row") as mul_row:
            with self.tik_inst.for_range(0, 3, name="env_col") as env_col:              # in_deriv -> (3, 1, 4, 256)
                with self.tik_inst.for_range(0, 4, name="idx_tb") as idx_tb:
                    p_tmpv = (mul_row + env_col * 3) * 4 * 256 + idx_tb * 256
                    p_mul = (mul_row * 4 + idx_tb) * 256
                    p_env = (idx_tb * 3 + env_col) * 256
                    self.tik_inst.vmul(64, tmpv_ub[p_tmpv], trans_ub[p_mul], drv_ub[p_env],
                                       4, 1, 1, 1, 8, 8, 8)                             # tmpv -> (3, 3, 4, 256)

        self.tik_inst.vcadd(64, trans_ub, tmpv_ub, 144, 1, 1, 8)
        self.tik_inst.vec_dup(16, net_ub, zero_scalar, 1, 2)
        self.tik_inst.vcadd(16, net_ub, trans_ub, 9, 1, 1, 2)                           # virial -> (9, )

        kk = nn * self.nei_burst // (self.nloc * self.nnei)
        self.tik_inst.set_atomic_add(1)
        self.tik_inst.data_move(self.virial_gm[kk * 9], net_ub, 0, 1, 2, 0, 0)
        self.tik_inst.set_atomic_add(0)

        self.tik_inst.v4dtrans(True, trans_ub, tmpv_ub, 256, 36)                        # atom_virial -> (256, 9, 4)
        self.tik_inst.vcpadd(64, tmpv_ub, trans_ub, 144, 1, 1, 8)
        self.tik_inst.vcpadd(64, trans_ub, tmpv_ub, 72, 1, 1, 8)                        # atom_virial -> (256, 9)

        nlist_idx = nn * self.nei_burst
        self.tik_inst.data_move(nlist_ub, self.nlist_gm[nlist_idx], 0, 1, 32, 0, 0)     # nlist -> (256, )
        self.tik_inst.vadds(64, nlist_ub, nlist_ub, kk * self.nall, 4, 1, 1, 8, 8)
        self.tik_inst.vmuls(64, nlist_ub, nlist_ub, 9, 4, 1, 1, 8, 8)

        self.tik_inst.v4dtrans(False, drv_ub, trans_ub, 256, 9)
        self.tik_inst.vec_dup(64, drv_ub[2304], zero_scalar, 28, 8)                     # atom_virial padding 0
        self.tik_inst.v4dtrans(True, tmpv_ub, drv_ub, 256, 16)                          # atom_virial -> (256, 16)

        self.tik_inst.set_atomic_add(1)
        av_offset = kk * self.nall * 9
        with self.tik_inst.for_range(0, self.nei_burst // 4, name="jj") as jj:
            j_idx0.set_as(nlist_ub[jj * 4])
            with self.tik_inst.if_scope(j_idx0 >= av_offset):
                self.tik_inst.data_move(self.atom_virial_gm[j_idx0], tmpv_ub[jj * 64], 0, 1, 2, 0, 0)

            j_idx1.set_as(nlist_ub[jj * 4 + 1])
            with self.tik_inst.if_scope(j_idx1 >= av_offset):
                self.tik_inst.data_move(self.atom_virial_gm[j_idx1], tmpv_ub[jj * 64 + 16], 0, 1, 2, 0, 0)

            j_idx2.set_as(nlist_ub[jj * 4 + 2])
            with self.tik_inst.if_scope(j_idx2 >= av_offset):
                self.tik_inst.data_move(self.atom_virial_gm[j_idx2], tmpv_ub[jj * 64 + 32], 0, 1, 2, 0, 0)

            j_idx3.set_as(nlist_ub[jj * 4 + 3])
            with self.tik_inst.if_scope(j_idx3 >= av_offset):
                self.tik_inst.data_move(self.atom_virial_gm[j_idx3], tmpv_ub[jj * 64 + 48], 0, 1, 2, 0, 0)
        self.tik_inst.set_atomic_add(0)

    def _compute_db_fp32(self, nnei_start, nnei_end):
        """
        compute with double buffer
        """
        if nnei_end == nnei_start + 1:
            ub_tuple = self._init_ub_data_fp32()
            self._compute_virial_fp32(nnei_start, ub_tuple)
        elif nnei_end > nnei_start + 1:
            with self.tik_inst.for_range(nnei_start, nnei_end, thread_num=2) as nn:
                ub_tuple = self._init_ub_data_fp32()
                self._compute_virial_fp32(nn, ub_tuple)

    def _compute_fp32(self):
        """
        compute fp32
        """
        with self.tik_inst.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as block_i:
            with self.tik_inst.if_scope(block_i < self.pre_core_num):
                nnei_start = block_i * self.nei_rep_times_pre_core
                nnei_end = (block_i + 1) * self.nei_rep_times_pre_core
                self._compute_db_fp32(nnei_start, nnei_end)
            with self.tik_inst.else_scope():
                nnei_start = self.pre_core_num + block_i * self.nei_rep_times_post_core
                nnei_end = self.pre_core_num + (block_i + 1) * self.nei_rep_times_post_core
                self._compute_db_fp32(nnei_start, nnei_end)

    def compute(self):
        """
        compute
        """
        if self.op_data_type == "float32":
            self._compute_fp32()

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.net_deriv_gm, self.in_deriv_gm, self.rij_gm, self.nlist_gm, self.natoms_gm],
                               outputs=[self.virial_gm, self.atom_virial_gm])


# 'pylint: disable=too-many-arguments,too-many-locals
def _check_params(net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, n_a_sel, n_r_sel, nall, kernel_name):
    net_deriv_dtype = net_deriv.get("dtype").lower()
    para_check.check_dtype(net_deriv_dtype, ("float32"), param_name="net_deriv")

    in_deriv_dtype = in_deriv.get("dtype").lower()
    para_check.check_dtype(in_deriv_dtype, ("float32"), param_name="in_deriv")

    rij_dtype = rij.get("dtype").lower()
    para_check.check_dtype(rij_dtype, ("float32"), param_name="rij")

    nlist_dtype = nlist.get("dtype").lower()
    para_check.check_dtype(nlist_dtype, ("int32"), param_name="nlist")

    natoms_dtype = natoms.get("dtype").lower()
    para_check.check_dtype(natoms_dtype, ("int32"), param_name="natoms")

    virial_dtype = virial.get("dtype").lower()
    para_check.check_dtype(virial_dtype, ("float32"), param_name="virial")

    atom_virial_dtype = atom_virial.get("dtype").lower()
    para_check.check_dtype(atom_virial_dtype, ("float32"), param_name="atom_virial")

    if any((net_deriv_dtype != in_deriv_dtype, net_deriv_dtype != rij_dtype)):
        rule = "Data type of {net_deriv, in_deriv, rij} is not match."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule)

    net_deriv_shape = net_deriv.get("shape")
    para_check.check_shape(net_deriv_shape, min_rank=2, max_rank=2, param_name="net_deriv")

    in_deriv_shape = in_deriv.get("shape")
    para_check.check_shape(in_deriv_shape, min_rank=2, max_rank=2, param_name="in_deriv")

    rij_shape = rij.get("shape")
    para_check.check_shape(rij_shape, min_rank=2, max_rank=2, param_name="rij")

    nlist_shape = nlist.get("shape")
    para_check.check_shape(nlist_shape, min_rank=2, max_rank=2, param_name="nlist")

    natoms_shape = natoms.get("shape")
    para_check.check_shape(natoms_shape, min_rank=1, max_rank=1, min_size=3, param_name="natoms")

    if any((n_a_sel < 0, n_r_sel < 0, n_a_sel + n_r_sel <= 0)):
        rule = "The attributes {n_r_sel, n_r_sel} can not be minus value or all 0."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule)

    if nall <= 0:
        rule = "nall should be greater than 0."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule)


# 'pylint: disable=too-many-arguments,
@register_operator("ProdVirialSeA")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def prod_virial_se_a(net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, n_a_sel, n_r_sel, nall=28328,
                     kernel_name="prod_virial_se_a"):
    """
    Compute ProdVirialSeA.

    Parameters
    ----------
    net_deriv : dict. shape and dtype of input data net_deriv
    in_deriv : dict. shape and dtype of input data in_deriv
    rij : dict. shape and dtype of input data rij
    nlist : dict. shape and dtype of input data nlist
    natoms : dict. shape and dtype of input data natoms
    virial : dict. shape and dtype of output data virial
    atom_virial : dict. shape and dtype of output data atom_virial
    n_a_sel : value of attr n_a_sel
    n_r_sel : value of attr n_r_sel
    kernel_name : str. cce kernel name, default value is "prod_virial_se_a"

    Returns
    -------
    None
    """
    _check_params(net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, n_a_sel, n_r_sel, nall, kernel_name)

    nnei = n_a_sel + n_r_sel
    obj = ProdVirialSeA(net_deriv, in_deriv, rij, nlist, natoms, nnei, nall, kernel_name)
    obj.compute()
