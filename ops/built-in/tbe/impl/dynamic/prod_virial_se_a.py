"""
Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

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
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl import constant_util as constant


# 'pylint: disable=too-few-public-methods
class ProdVirialSeA:
    """
    ProdVirialSeA op implement
    """
    NNEI_UB = 256

    # 'pylint: disable=unused-argument,too-many-arguments
    def __init__(self, net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, nnei, split_count, split_index,
                 kernel_name):
        self.kernel_name = kernel_name
        self.tik_inst = tik.Tik(tik.Dprofile)

        self.op_data_type = net_deriv.get("dtype").lower()
        self.nlist_dtype = nlist.get("dtype").lower()
        self.natoms_dtype = natoms.get("dtype").lower()

        natoms_shape = natoms.get("shape")

        self.nnei = nnei
        self.split_count = split_count
        self.split_index = split_index

        self.tiling_gm = self.tik_inst.Tensor("int64", (constant.SIZE_SIXTEEN,), name="tiling_gm", scope=tik.scope_gm)
        self.net_deriv_gm = self.tik_inst.Tensor(self.op_data_type, (constant.SHAPE_SIZE_LIMIT,), name="net_deriv_gm",
                                                 scope=tik.scope_gm)
        self.in_deriv_gm = self.tik_inst.Tensor(self.op_data_type, (constant.SHAPE_SIZE_LIMIT,), name="in_deriv_gm",
                                                scope=tik.scope_gm)
        self.rij_gm = self.tik_inst.Tensor(self.op_data_type, (constant.SHAPE_SIZE_LIMIT,), name="rij_gm",
                                           scope=tik.scope_gm)
        self.nlist_gm = self.tik_inst.Tensor(self.nlist_dtype, (constant.SHAPE_SIZE_LIMIT,), name="nlist_gm",
                                             scope=tik.scope_gm)
        self.natoms_gm = self.tik_inst.Tensor(self.natoms_dtype, (natoms_shape[0],), name="natoms_gm",
                                              scope=tik.scope_gm)
        self.virial_gm = self.tik_inst.Tensor(self.op_data_type, (constant.SHAPE_SIZE_LIMIT,), name="virial_gm",
                                              scope=tik.scope_gm, is_atomic_add=True)
        self.atom_virial_gm = self.tik_inst.Tensor(self.op_data_type, (constant.SHAPE_SIZE_LIMIT,),
                                                   name="atom_virial_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.tiling_ub = None

        self.nframes = self.tik_inst.Scalar(dtype="int64", name="nframes")
        self.nnei_per_frame = self.tik_inst.Scalar(dtype="int64", name="nnei_per_frame")
        self.nall = self.tik_inst.Scalar(dtype="int64", name="nall")
        self.rep_times_offset = self.tik_inst.Scalar(dtype="int64", name="rep_times_offset")
        self.nei_rep_times = self.tik_inst.Scalar(dtype="int64", name="nei_rep_times")

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.pre_core_num = self.tik_inst.Scalar(dtype="int64", name="pre_core_num")
        self.post_core_num = self.tik_inst.Scalar(dtype="int64", name="post_core_num")
        self.nei_rep_times_pre_core = self.tik_inst.Scalar(dtype="int64", name="nei_rep_times_pre_core")
        self.nei_rep_times_post_core = self.tik_inst.Scalar(dtype="int64", name="nei_rep_times_post_core")

    def _tiling_args(self):
        """
        Get runtime tiling parameters from tiling.
        """
        self.nframes.set_as(self.tiling_ub[0])
        self.nnei_per_frame.set_as(self.tiling_ub[1])
        self.nall.set_as(self.tiling_ub[2])
        self.rep_times_offset.set_as(self.tiling_ub[3])
        self.nei_rep_times.set_as(self.tiling_ub[4])
        self.pre_core_num.set_as(self.tiling_ub[5])
        self.post_core_num.set_as(self.tiling_ub[6])
        self.nei_rep_times_pre_core.set_as(self.tiling_ub[7])
        self.nei_rep_times_post_core.set_as(self.tiling_ub[8])

    def _init_ub_data_fp32(self):
        """
        init ub data fp32
        """
        net_ub = self.tik_inst.Tensor(self.op_data_type, (ProdVirialSeA.NNEI_UB * 4,), name="net_ub",
                                      scope=tik.scope_ubuf)
        drv_ub = self.tik_inst.Tensor(self.op_data_type, (ProdVirialSeA.NNEI_UB * 4 * 4,), name="drv_ub",
                                      scope=tik.scope_ubuf)
        nlist_ub = self.tik_inst.Tensor(self.nlist_dtype, (ProdVirialSeA.NNEI_UB,), name="nlist_ub",
                                        scope=tik.scope_ubuf)

        op_ub_shape = (ProdVirialSeA.NNEI_UB * 4 * 3 * 3,)
        trans_ub = self.tik_inst.Tensor(self.op_data_type, op_ub_shape, name="trans_ub", scope=tik.scope_ubuf)
        tmpv_ub = self.tik_inst.Tensor(self.op_data_type, op_ub_shape, name="tmpv_ub", scope=tik.scope_ubuf)

        j_idx0 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx1 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx2 = self.tik_inst.Scalar(dtype=self.nlist_dtype)
        j_idx3 = self.tik_inst.Scalar(dtype=self.nlist_dtype)

        return net_ub, drv_ub, nlist_ub, trans_ub, tmpv_ub, j_idx0, j_idx1, j_idx2, j_idx3

    # 'pylint: disable=too-many-locals,too-many-statements
    def _compute_virial_fp32(self, ub_tuple, kk, nei_start, nnei_ub):
        """
        Support shape/datatype:
        type  param name   shape form                      dtype     shape in ub
        in    net_deriv    (nframes, nloc * nnei * 4)      float32   (256, 4)
        in    in_deriv     (nframes, nloc * nnei * 4 * 3)  float32   (256, 4, 3)
        in    rij          (nframes, nloc * nnei * 3)      float32   (256, 3)
        in    nlist        (nframes, nloc * nnei)          int32     (256, )
        in    natoms       (2 + ntypes, )                  int32
        out   virial       (nframes, 9)                    float32   (9, )
        out   atom_virial  (nframes, nall * 9)             float32   (256 * 9, )
        """
        (net_ub, drv_ub, nlist_ub, trans_ub, tmpv_ub, j_idx0, j_idx1, j_idx2, j_idx3) = ub_tuple

        with self.tik_inst.if_scope(nnei_ub < ProdVirialSeA.NNEI_UB):
            self.tik_inst.vec_dup(64, trans_ub, 0, 16, 8)
        self.tik_inst.data_move(trans_ub, self.net_deriv_gm[nei_start * 4], 0, 1, 128, 0, 0)
        self.tik_inst.v4dtrans(False, net_ub, trans_ub, ProdVirialSeA.NNEI_UB, 4)        # net_deriv -> (1, 4, 256)

        with self.tik_inst.if_scope(nnei_ub < ProdVirialSeA.NNEI_UB):
            self.tik_inst.vec_dup(64, trans_ub, 0, 12, 8)
        self.tik_inst.data_move(trans_ub, self.rij_gm[nei_start * 3], 0, 1, 96, 0, 0)
        self.tik_inst.v4dtrans(False, drv_ub, trans_ub, ProdVirialSeA.NNEI_UB, 3)        # rij -> (3, 1, 256)

        with self.tik_inst.for_range(0, 3, name="rij_row") as rij_row:
            with self.tik_inst.for_range(0, 4, name="net_row") as net_row:
                self.tik_inst.vmul(64, trans_ub[(rij_row * 4 + net_row) * ProdVirialSeA.NNEI_UB],
                                   drv_ub[rij_row * ProdVirialSeA.NNEI_UB],
                                   net_ub[net_row * ProdVirialSeA.NNEI_UB],
                                   4, 1, 1, 1, 8, 8, 8)                                  # tmpv -> (1, 3, 4, 256)

        with self.tik_inst.if_scope(nnei_ub < ProdVirialSeA.NNEI_UB):
            self.tik_inst.vec_dup(64, tmpv_ub, 0, 48, 8)
        self.tik_inst.data_move(tmpv_ub, self.in_deriv_gm[nei_start * 4 * 3], 0, 1, 384, 0, 0)
        self.tik_inst.v4dtrans(False, drv_ub, tmpv_ub, ProdVirialSeA.NNEI_UB, 12)        # in_deriv -> (4, 3, 256)

        with self.tik_inst.for_range(0, 3, name="mul_row") as mul_row:
            with self.tik_inst.for_range(0, 3, name="env_col") as env_col:               # in_deriv -> (3, 1, 4, 256)
                with self.tik_inst.for_range(0, 4, name="idx_tb") as idx_tb:
                    self.tik_inst.vmul(64, tmpv_ub[(mul_row + env_col * 3) * 4 * ProdVirialSeA.NNEI_UB +
                                                   idx_tb * ProdVirialSeA.NNEI_UB],
                                       trans_ub[(mul_row * 4 + idx_tb) * ProdVirialSeA.NNEI_UB],
                                       drv_ub[(idx_tb * 3 + env_col) * ProdVirialSeA.NNEI_UB],
                                       4, 1, 1, 1, 8, 8, 8)                              # tmpv -> (3, 3, 4, 256)

        self.tik_inst.vcadd(64, trans_ub, tmpv_ub, 144, 1, 1, 8)
        self.tik_inst.vec_dup(16, net_ub, 0, 1, 2)
        self.tik_inst.vcadd(16, net_ub, trans_ub, 9, 1, 1, 2)                            # virial -> (9, )

        self.tik_inst.set_atomic_add(1)
        self.tik_inst.data_move(self.virial_gm[kk * 9], net_ub, 0, 1, 2, 0, 0)
        self.tik_inst.set_atomic_add(0)

        self.tik_inst.v4dtrans(True, trans_ub, tmpv_ub, ProdVirialSeA.NNEI_UB, 36)       # atom_virial -> (256, 9, 4)
        self.tik_inst.vcpadd(64, tmpv_ub, trans_ub, 144, 1, 1, 8)
        self.tik_inst.vcpadd(64, trans_ub, tmpv_ub, 72, 1, 1, 8)                         # atom_virial -> (256, 9)

        self.tik_inst.data_move(nlist_ub, self.nlist_gm[nei_start], 0, 1, 32, 0, 0)      # nlist -> (256, )
        self.tik_inst.vadds(64, nlist_ub, nlist_ub, kk * self.nall, 4, 1, 1, 8, 8)
        self.tik_inst.vmuls(64, nlist_ub, nlist_ub, 9, 4, 1, 1, 8, 8)

        self.tik_inst.v4dtrans(False, drv_ub, trans_ub, ProdVirialSeA.NNEI_UB, 9)
        self.tik_inst.vec_dup(64, drv_ub[2304], 0, 28, 8)
        self.tik_inst.v4dtrans(True, tmpv_ub, drv_ub, ProdVirialSeA.NNEI_UB, 16)         # atom_virial -> (256, 16)

        self.tik_inst.set_atomic_add(1)
        with self.tik_inst.for_range(0, 64, name="jj") as jj:
            with self.tik_inst.if_scope(jj * 4 < nnei_ub):
                j_idx0.set_as(nlist_ub[jj * 4])
                with self.tik_inst.if_scope(j_idx0 >= kk * self.nall * 9):
                    self.tik_inst.data_move(self.atom_virial_gm[j_idx0], tmpv_ub[jj * 64], 0, 1, 2, 0, 0)

            with self.tik_inst.if_scope(jj * 4 + 1 < nnei_ub):
                j_idx1.set_as(nlist_ub[jj * 4 + 1])
                with self.tik_inst.if_scope(j_idx1 >= kk * self.nall * 9):
                    self.tik_inst.data_move(self.atom_virial_gm[j_idx1], tmpv_ub[jj * 64 + 16], 0, 1, 2, 0, 0)

            with self.tik_inst.if_scope(jj * 4 + 2 < nnei_ub):
                j_idx2.set_as(nlist_ub[jj * 4 + 2])
                with self.tik_inst.if_scope(j_idx2 >= kk * self.nall * 9):
                    self.tik_inst.data_move(self.atom_virial_gm[j_idx2], tmpv_ub[jj * 64 + 32], 0, 1, 2, 0, 0)

            with self.tik_inst.if_scope(jj * 4 + 3 < nnei_ub):
                j_idx3.set_as(nlist_ub[jj * 4 + 3])
                with self.tik_inst.if_scope(j_idx3 >= kk * self.nall * 9):
                    self.tik_inst.data_move(self.atom_virial_gm[j_idx3], tmpv_ub[jj * 64 + 48], 0, 1, 2, 0, 0)
        self.tik_inst.set_atomic_add(0)

    def _compute_simple_fp32(self, nnei_start, nnei_end):
        """
        compute with tail
        """
        ub_tuple = self._init_ub_data_fp32()
        with self.tik_inst.if_scope(self.nnei_per_frame % ProdVirialSeA.NNEI_UB == 0):
            kk_scalar = self.tik_inst.Scalar(dtype="int32")
            with self.tik_inst.for_range(nnei_start, nnei_end, name="nn") as nn:
                kk_scalar.set_as(nn * ProdVirialSeA.NNEI_UB // self.nnei_per_frame)
                self._compute_virial_fp32(ub_tuple, kk_scalar, nn * ProdVirialSeA.NNEI_UB, ProdVirialSeA.NNEI_UB)
        with self.tik_inst.elif_scope(self.nnei_per_frame > ProdVirialSeA.NNEI_UB):
            pre_kk_scalar = self.tik_inst.Scalar(dtype="int32")
            post_kk_scalar = self.tik_inst.Scalar(dtype="int32")
            last_frame_tail = self.tik_inst.Scalar(dtype="int32")
            next_frame_head = self.tik_inst.Scalar(dtype="int32")
            with self.tik_inst.for_range(nnei_start, nnei_end, name="nn") as nn:
                pre_kk_scalar.set_as(nn * ProdVirialSeA.NNEI_UB // self.nnei_per_frame)
                post_kk_scalar.set_as((nn * ProdVirialSeA.NNEI_UB + ProdVirialSeA.NNEI_UB - 1) // self.nnei_per_frame)
                with self.tik_inst.if_scope(pre_kk_scalar == post_kk_scalar):
                    self._compute_virial_fp32(ub_tuple, pre_kk_scalar, nn * ProdVirialSeA.NNEI_UB,
                                              ProdVirialSeA.NNEI_UB)
                with self.tik_inst.else_scope():
                    last_frame_tail.set_as(post_kk_scalar * self.nnei_per_frame - nn * ProdVirialSeA.NNEI_UB)
                    self._compute_virial_fp32(ub_tuple, pre_kk_scalar, nn * ProdVirialSeA.NNEI_UB, last_frame_tail)

                    with self.tik_inst.if_scope(post_kk_scalar < self.nframes):
                        next_frame_head.set_as(ProdVirialSeA.NNEI_UB - last_frame_tail)
                        self._compute_virial_fp32(ub_tuple, post_kk_scalar, post_kk_scalar * self.nnei_per_frame,
                                                  next_frame_head)
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(nnei_start, nnei_end, name="kk") as kk:
                self._compute_virial_fp32(ub_tuple, kk, kk * self.nnei_per_frame, self.nnei_per_frame)

    def _compute_fp32(self):
        """
        compute fp32
        """
        with self.tik_inst.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as block_i:
            self.tiling_ub = self.tik_inst.Tensor("int64", (constant.SIZE_SIXTEEN,), name="tiling_ub",
                                                  scope=tik.scope_ubuf)
            self.tik_inst.data_move(self.tiling_ub, self.tiling_gm, 0, 1, constant.SIZE_SIXTEEN * 8 // 32, 0, 0)
            self._tiling_args()

            nnei_start = self.tik_inst.Scalar(dtype="int32")
            nnei_end = self.tik_inst.Scalar(dtype="int32")
            with self.tik_inst.if_scope(block_i < self.pre_core_num):
                nnei_start.set_as(self.rep_times_offset + block_i * self.nei_rep_times_pre_core)
                nnei_end.set_as(nnei_start +  self.nei_rep_times_pre_core)
            with self.tik_inst.else_scope():
                nnei_start.set_as(self.rep_times_offset + self.pre_core_num + block_i * self.nei_rep_times_post_core)
                nnei_end.set_as(nnei_start + self.nei_rep_times_post_core)

            with self.tik_inst.if_scope(tik.all(self.nnei_per_frame % ProdVirialSeA.NNEI_UB == 0,
                                                self.nei_rep_times >= self.ai_core_num * 2)):
                with self.tik_inst.for_range(nnei_start, nnei_end, thread_num=2) as nn:
                    ub_tuple = self._init_ub_data_fp32()
                    kk_scalar = self.tik_inst.Scalar(dtype="int32")
                    kk_scalar.set_as(nn * ProdVirialSeA.NNEI_UB // self.nnei_per_frame)
                    self._compute_virial_fp32(ub_tuple, kk_scalar, nn * ProdVirialSeA.NNEI_UB, ProdVirialSeA.NNEI_UB)
            with self.tik_inst.else_scope():
                self._compute_simple_fp32(nnei_start, nnei_end)

    def compute(self):
        """
        compute
        """
        if self.op_data_type == "float32":
            self._compute_fp32()

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.ai_core_num,
                                                            "split_count": self.split_count,
                                                            "split_index": self.split_index})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.net_deriv_gm, self.in_deriv_gm, self.rij_gm, self.nlist_gm, self.natoms_gm],
                               outputs=[self.virial_gm, self.atom_virial_gm],
                               flowtable=(self.tiling_gm,),
                               config=opt_config)


# 'pylint: disable=too-many-locals,too-many-arguments
def _check_params(net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, n_a_sel, n_r_sel, split_count,
                  split_index, kernel_name):
    net_deriv_dtype = net_deriv.get("dtype").lower()
    para_check.check_dtype(net_deriv_dtype, ("float32", ), param_name="net_deriv")

    in_deriv_dtype = in_deriv.get("dtype").lower()
    para_check.check_dtype(in_deriv_dtype, ("float32", ), param_name="in_deriv")

    rij_dtype = rij.get("dtype").lower()
    para_check.check_dtype(rij_dtype, ("float32", ), param_name="rij")

    nlist_dtype = nlist.get("dtype").lower()
    para_check.check_dtype(nlist_dtype, ("int32", ), param_name="nlist")

    natoms_dtype = natoms.get("dtype").lower()
    para_check.check_dtype(natoms_dtype, ("int32", ), param_name="natoms")

    virial_dtype = virial.get("dtype").lower()
    para_check.check_dtype(virial_dtype, ("float32", ), param_name="virial")

    atom_virial_dtype = atom_virial.get("dtype").lower()
    para_check.check_dtype(atom_virial_dtype, ("float32", ), param_name="atom_virial")

    if any((net_deriv_dtype != in_deriv_dtype, net_deriv_dtype != rij_dtype)):
        rule = "Data type of {net_deriv, in_deriv, rij} is not match."
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

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

    virial_shape = virial.get("shape")
    para_check.check_shape(virial_shape, min_rank=2, max_rank=2, param_name="virial")

    atom_virial_shape = atom_virial.get("shape")
    para_check.check_shape(atom_virial_shape, min_rank=2, max_rank=2, param_name="atom_virial")

    if any((net_deriv_shape[0] != in_deriv_shape[0], net_deriv_shape[0] != rij_shape[0],
            net_deriv_shape[0] != nlist_shape[0])):
        rule = "Number of samples should match"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    if any((n_a_sel < 0, n_r_sel < 0, n_a_sel + n_r_sel <= 0)):
        rule = "The attributes {n_r_sel, n_r_sel} can not be minus value or all 0"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)

    if any((split_count < 1, split_index < 0, split_count <= split_index)):
        rule = "Failed to check split info"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)


# 'pylint: disable=too-many-arguments
@register_operator("ProdVirialSeA")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def prod_virial_se_a(net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, n_a_sel, n_r_sel, split_count=1,
                     split_index=0, kernel_name="prod_virial_se_a"):
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
    _check_params(net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, n_a_sel, n_r_sel, split_count,
                  split_index, kernel_name)

    nnei = n_a_sel + n_r_sel
    obj = ProdVirialSeA(net_deriv, in_deriv, rij, nlist, natoms, virial, atom_virial, nnei, split_count, split_index,
                        kernel_name)
    obj.compute()
