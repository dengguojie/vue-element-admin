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

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context


def _ceil_div(dividend, divisor):
    result = (dividend + divisor - 1) // divisor
    return result


def _ceil_fill(dividend, divisor):
    result = ((dividend + divisor - 1) // divisor) * divisor
    return result


def _floor_div(dividend, divisor):
    result = dividend // divisor
    return result


def _floor_fill(dividend, divisor):
    result = (dividend // divisor) * divisor
    return result


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    MAX_INT64 = 2**63 - 1
    MASK_FLOAT32 = 64
    BLOCK_FLOAT32 = 8
    NLOC_UNIT_LEN = 4
    TILING_LEN = 8
    UB_MAX_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) // 4
    NNEI_UNIT_MAX = _floor_fill(UB_MAX_SIZE // 2 // 36, BLOCK_FLOAT32)


class ProdForceSeA:
    """
    Function: use to calc the force to those local atoms for all frames
    """
    def __init__(self, attrs, dtypes, shapes):
        """
        initial function
        """
        self.tik_instance = tik.Tik()
        self.opt_config = {"out_of_bound_sync_check": True,
                           "enable_const_fold": True,
                           "double_buffer_non_reuse": True}
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_LEN,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        (net_deriv_dtype, in_deriv_dtype, nlist_dtype, natoms_dtype, force_dtype) = dtypes
        (natoms_shape) = shapes
        (n_a_sel, n_r_sel, split_count, split_index) = attrs
        self.n_a_sel =n_a_sel
        self.n_r_sel =n_r_sel
        self.split_count = split_count
        self.split_index = split_index
        self.nnei = n_a_sel + n_r_sel
        self.net_deriv_dtype = net_deriv_dtype
        self.in_deriv_dtype = in_deriv_dtype
        self.nlist_dtype = nlist_dtype
        self.natoms_dtype = natoms_dtype
        self.force_dtype = force_dtype
        self.natoms_shape = natoms_shape
        self.nall = self.tik_instance.Scalar("int64", name="nall")
        self.nloc = self.tik_instance.Scalar("int64", name="nloc")
        self.nframes = self.tik_instance.Scalar("int64", name="nframes")
        self.core_loop_unit = self.tik_instance.Scalar("int64", name="core_loop_unit")
        self.core_loop_left = self.tik_instance.Scalar("int64", name="core_loop_left")
        self.core_offset = self.tik_instance.Scalar("int64", name="core_offset")
        self.core_nums_used = self.tik_instance.Scalar("int64", name="core_nums_used")
        self.nnei_unit_len = _ceil_fill(self.nnei, 128)
        if self.nnei_unit_len > Constant.NNEI_UNIT_MAX:
            self.nnei_unit_len = Constant.NNEI_UNIT_MAX
        self.nall_ub_len = _floor_fill(Constant.UB_MAX_SIZE // 2 - self.nnei_unit_len * 14,
                                       Constant.MASK_FLOAT32)
        self.net_deriv_gm = None
        self.in_deriv_gm = None
        self.nlist_gm = None
        self.natoms_gm = None
        self.force_gm = None

    def _init_gm_data_fp32(self):
        """
        init gm data
        """
        self.net_deriv_gm = self.tik_instance.Tensor(self.net_deriv_dtype, (Constant.MAX_INT64,),
                                                     name="net_deriv_gm", scope=tik.scope_gm)
        self.in_deriv_gm = self.tik_instance.Tensor(self.in_deriv_dtype, (Constant.MAX_INT64,),
                                                    name="in_deriv_gm", scope=tik.scope_gm)
        self.nlist_gm = self.tik_instance.Tensor(self.nlist_dtype, (Constant.MAX_INT64,), name="nlist_gm",
                                                 scope=tik.scope_gm)
        self.natoms_gm = self.tik_instance.Tensor(self.natoms_dtype, self.natoms_shape, name="natoms_gm",
                                                  scope=tik.scope_gm)
        self.force_gm = self.tik_instance.Tensor(self.force_dtype, (Constant.MAX_INT64,), name="force_gm",
                                                 scope=tik.scope_gm, is_atomic_add=True)

    def _tiling_args(self):
        """
        tiling_args
        """
        tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_LEN,),
                                             name="tiling_ub",
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
        tiling_para_index = 0
        self.nloc.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.nall.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.core_loop_unit.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.core_loop_left.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.core_offset.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.nframes.set_as(tiling_ub[tiling_para_index])
        tiling_para_index = tiling_para_index + 1
        self.core_nums_used.set_as(tiling_ub[tiling_para_index])

    def _run_multi_core_loop(self):
        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_idx:
            with self.tik_instance.if_scope(core_idx < self.core_nums_used - 1):
                self._run_one_core_loop(core_idx, self.core_loop_unit)
            with self.tik_instance.if_scope(core_idx == self.core_nums_used - 1):
                with self.tik_instance.if_scope(self.core_loop_left == 0):
                    self._run_one_core_loop(core_idx, self.core_loop_unit)
                with self.tik_instance.else_scope():
                    self._run_one_core_loop(core_idx, self.core_loop_left)

    def _run_one_core_loop(self, core_idx, core_loop_unit):
        with self.tik_instance.for_range(0, self.nframes) as frame_idx:
            loop_offset = frame_idx * self.nloc + self.core_offset + core_idx * self.core_loop_unit
            nloc_loop = core_loop_unit // Constant.NLOC_UNIT_LEN
            nloc_left = core_loop_unit % Constant.NLOC_UNIT_LEN
            # nloc loop
            with self.tik_instance.for_range(0, nloc_loop, thread_num=2) as nloc_idx:
                nloc_offset = nloc_idx * Constant.NLOC_UNIT_LEN
                # nnei loop
                with self.tik_instance.for_range(0, self.nnei // self.nnei_unit_len) as nnei_idx:
                    ndescrpt = self.nnei_unit_len * 4
                    deriv_nnei_vcpadd_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                        (3, Constant.NLOC_UNIT_LEN, self.nnei_unit_len),name="deriv_nnei_vcpadd_ub_fp32",
                        scope=tik.scope_ubuf)
                    # prepare
                    with self.tik_instance.for_range(0, Constant.NLOC_UNIT_LEN) as i:
                        deriv_input_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                       (1, 1, ndescrpt, 3),
                                                                       name="deriv_input_ub_fp32",
                                                                       scope=tik.scope_ubuf)
                        deriv_trans_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                       (1, 3, 1, ndescrpt),
                                                                       name="deriv_trans_ub_fp32",
                                                                       scope=tik.scope_ubuf)
                        deriv_offset = (loop_offset + nloc_offset + i) * self.nnei * 4 + \
                            nnei_idx * self.nnei_unit_len * 4
                        self.tik_instance.data_move(deriv_input_ub_fp32,
                                                    self.in_deriv_gm[deriv_offset * 3],
                                                    0, 1, 3 * ndescrpt // Constant.BLOCK_FLOAT32, 0, 0)
                        self.tik_instance.v4dtrans(False, deriv_trans_ub_fp32[0, 0, 0, 0],
                                                   deriv_input_ub_fp32[0, 0, 0, 0], ndescrpt, 3)
                        self.tik_instance.data_move(deriv_input_ub_fp32,
                                                    self.net_deriv_gm[deriv_offset],
                                                    0, 1, ndescrpt // Constant.BLOCK_FLOAT32, 0, 0)
                        vmul_repeat = ndescrpt // Constant.MASK_FLOAT32
                        vmul_left = ndescrpt % Constant.MASK_FLOAT32
                        if vmul_repeat > 0:
                            self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32,
                                                   deriv_trans_ub_fp32,
                                                   deriv_input_ub_fp32,
                                                   vmul_repeat, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[ndescrpt],
                                                   deriv_trans_ub_fp32[ndescrpt],
                                                   deriv_input_ub_fp32,
                                                   vmul_repeat, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[2 * ndescrpt],
                                                   deriv_trans_ub_fp32[2 * ndescrpt],
                                                   deriv_input_ub_fp32,
                                                   vmul_repeat, 1, 1, 1, 8, 8, 8)
                        if vmul_left > 0:
                            vmul_floor = _floor_fill(ndescrpt, Constant.MASK_FLOAT32)
                            self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[vmul_floor],
                                                   deriv_trans_ub_fp32[vmul_floor],
                                                   deriv_input_ub_fp32[vmul_floor],
                                                   1, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[ndescrpt + vmul_floor],
                                                   deriv_trans_ub_fp32[ndescrpt + vmul_floor],
                                                   deriv_input_ub_fp32[vmul_floor],
                                                   1, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[2 * ndescrpt + vmul_floor],
                                                   deriv_trans_ub_fp32[2 * ndescrpt + vmul_floor],
                                                   deriv_input_ub_fp32[vmul_floor],
                                                   1, 1, 1, 1, 8, 8, 8)
                        vcpadd_repeat_1 = (3 * ndescrpt) // Constant.MASK_FLOAT32
                        vcpadd_left_1 = (3 * ndescrpt) % Constant.MASK_FLOAT32
                        if vcpadd_repeat_1 > 0:
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32, deriv_input_ub_fp32,
                                                     deriv_trans_ub_fp32,
                                                     vcpadd_repeat_1, 1, 1, 8)
                        if vcpadd_left_1 > 0:
                            vcpadd_floor = vcpadd_repeat_1 * Constant.MASK_FLOAT32
                            self.tik_instance.vcpadd(vcpadd_left_1, deriv_input_ub_fp32[vcpadd_floor // 2],
                                                     deriv_trans_ub_fp32[vcpadd_floor],
                                                     1, 1, 1, 8)
                        vcpadd_repeat_2 = (ndescrpt // 2) // Constant.MASK_FLOAT32
                        vcpadd_left_2 = (ndescrpt // 2) % Constant.MASK_FLOAT32
                        vcpadd_nloc_offset = (Constant.NLOC_UNIT_LEN + i) * self.nnei_unit_len
                        vcpadd_nloc_offset_1 = (Constant.NLOC_UNIT_LEN * 2 + i) * self.nnei_unit_len
                        if vcpadd_repeat_2 > 0:
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                     deriv_nnei_vcpadd_ub_fp32[i * self.nnei_unit_len],
                                                     deriv_input_ub_fp32,
                                                     vcpadd_repeat_2, 1, 1, 8)
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                     deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset],
                                                     deriv_input_ub_fp32[self.nnei_unit_len * 2],
                                                     vcpadd_repeat_2, 1, 1, 8)
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                     deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset_1],
                                                     deriv_input_ub_fp32[self.nnei_unit_len * 4],
                                                     vcpadd_repeat_2, 1, 1, 8)
                        if vcpadd_left_2 > 0:
                            vcpadd_offset = vcpadd_repeat_2 * Constant.MASK_FLOAT32
                            self.tik_instance.vcpadd(vcpadd_left_2,
                                deriv_nnei_vcpadd_ub_fp32[i * self.nnei_unit_len + vcpadd_offset // 2],
                                deriv_input_ub_fp32[vcpadd_offset],
                                1, 1, 1, 8)
                            self.tik_instance.vcpadd(vcpadd_left_2,
                                deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset + vcpadd_offset // 2],
                                deriv_input_ub_fp32[self.nnei_unit_len * 2 + vcpadd_offset],
                                1, 1, 1, 8)
                            self.tik_instance.vcpadd(vcpadd_left_2,
                                deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset_1 + vcpadd_offset // 2],
                                deriv_input_ub_fp32[self.nnei_unit_len * 4 + vcpadd_offset],
                                1, 1, 1, 8)

                    # first
                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        force_add_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                            (Constant.NLOC_UNIT_LEN * 3 + Constant.MASK_FLOAT32, ), name="force_add_ub_fp32",
                            scope=tik.scope_ubuf)
                        force_assis_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                                                       (Constant.BLOCK_FLOAT32 * 3, ),
                                                                       name="force_assis_ub_fp32",
                                                                       scope=tik.scope_ubuf)
                        force_vcadd_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                            (Constant.NLOC_UNIT_LEN * 3 + Constant.MASK_FLOAT32,), name="force_vcadd_ub_fp32",
                            scope=tik.scope_ubuf)
                        force_reduce_loop = self.nnei_unit_len // Constant.MASK_FLOAT32
                        force_reduce_left = self.nnei_unit_len % Constant.MASK_FLOAT32
                        for i in range(0, force_reduce_loop):
                            self.tik_instance.vcadd(Constant.MASK_FLOAT32, force_vcadd_ub_fp32,
                                deriv_nnei_vcpadd_ub_fp32[Constant.MASK_FLOAT32 * i],
                                Constant.NLOC_UNIT_LEN * 3, 1, 0, self.nnei_unit_len, stride_unit=2)
                            self.tik_instance.vadd(Constant.NLOC_UNIT_LEN * 3, force_add_ub_fp32,
                                force_add_ub_fp32, force_vcadd_ub_fp32, 1, 1, 1, 1, 8, 8, 8)
                        if force_reduce_left > 0:
                            force_reduce_floor = _floor_fill(self.nnei_unit_len, Constant.MASK_FLOAT32)
                            self.tik_instance.vector_dup(Constant.NLOC_UNIT_LEN * 3, force_vcadd_ub_fp32, 0, 1, 1, 8)
                            self.tik_instance.vcadd(force_reduce_left, force_vcadd_ub_fp32,
                                deriv_nnei_vcpadd_ub_fp32[force_reduce_floor],
                                Constant.NLOC_UNIT_LEN * 3, 1, 0, self.nnei_unit_len, stride_unit=2)
                            self.tik_instance.vadd(Constant.NLOC_UNIT_LEN * 3, force_add_ub_fp32,
                                force_add_ub_fp32, force_vcadd_ub_fp32,
                                1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vmuls(Constant.NLOC_UNIT_LEN * 3, force_add_ub_fp32,
                            force_add_ub_fp32, -1, 1, 1, 1, 8, 8)
                        force_offset = frame_idx * self.nall * 3 + core_idx * self.core_loop_unit + nloc_offset
                        self.tik_instance.vadds(Constant.NLOC_UNIT_LEN, force_assis_ub_fp32,
                            force_add_ub_fp32, 0, 3, 0, 0, 8, Constant.NLOC_UNIT_LEN, stride_unit=2)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.force_gm[force_offset],
                                                    force_assis_ub_fp32, 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(self.force_gm[force_offset + self.nall],
                                                    force_assis_ub_fp32[Constant.BLOCK_FLOAT32], 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(self.force_gm[force_offset + self.nall * 2],
                                                    force_assis_ub_fp32[Constant.BLOCK_FLOAT32 * 2], 0, 1, 1, 0, 0)
                        self.tik_instance.set_atomic_add(0)

                    #second
                    with self.tik_instance.for_range(0, 3) as local_idx:
                        with self.tik_instance.for_range(0, Constant.NLOC_UNIT_LEN) as nu_idx:
                            nlist_assis_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                            (self.nnei_unit_len, ),
                                                                            name="nlist_assis_ub_int32",
                                                                            scope=tik.scope_ubuf)
                            with self.tik_instance.new_stmt_scope(disable_sync=False):
                                nlist_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                          (self.nnei_unit_len, ),
                                                                          name="nlist_ub_int32",
                                                                          scope=tik.scope_ubuf)
                                nlist_offset = (loop_offset + nloc_offset + nu_idx) * self.nnei + \
                                    nnei_idx * self.nnei_unit_len
                                self.tik_instance.data_move(nlist_ub_int32,
                                                            self.nlist_gm[nlist_offset],
                                                            0, 1, self.nnei_unit_len // Constant.BLOCK_FLOAT32, 0, 0)
                                if self.nnei_unit_len // Constant.MASK_FLOAT32 > 0:
                                    self.tik_instance.vmuls(Constant.MASK_FLOAT32, nlist_assis_ub_int32,
                                                            nlist_ub_int32, 4,
                                                            self.nnei_unit_len // Constant.MASK_FLOAT32, 1, 1, 8, 8)
                                if self.nnei_unit_len % Constant.MASK_FLOAT32 > 0:
                                    nlist_floor = _floor_fill(self.nnei_unit_len, Constant.MASK_FLOAT32)
                                    self.tik_instance.vmuls(self.nnei_unit_len % Constant.MASK_FLOAT32,
                                                            nlist_assis_ub_int32[nlist_floor],
                                                            nlist_ub_int32[nlist_floor],
                                                            4, 1, 1, 1, 8, 8)
                            force_out_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                (self.nall_ub_len,), name="force_out_ub_fp32",
                                scope=tik.scope_ubuf)
                            full_loop = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) // 255
                            full_left = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) % 255
                            for i in range(0, full_loop):
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                                             force_out_ub_fp32[i * 255 * Constant.MASK_FLOAT32],
                                                             0, 255, 1, 8)
                            if full_left > 0:
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                    force_out_ub_fp32[full_loop * 255 * Constant.MASK_FLOAT32],
                                    0, full_left, 1, 8)
                            offset = \
                                Constant.NLOC_UNIT_LEN * self.nnei_unit_len * local_idx + self.nnei_unit_len * nu_idx
                            self.tik_instance.vscatter(self.nnei_unit_len, force_out_ub_fp32,
                                                       deriv_nnei_vcpadd_ub_fp32[offset],
                                                       nlist_assis_ub_int32,
                                                       1, 8, 0, 0, "counter")
                            force_offset_2 = frame_idx * self.nall * 3 + self.nall * local_idx
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.force_gm[force_offset_2],
                                                        force_out_ub_fp32,
                                                        0, 1, _ceil_div(self.nall, Constant.BLOCK_FLOAT32), 0, 0)
                            self.tik_instance.set_atomic_add(0)

                # nnei tail
                with self.tik_instance.if_scope(self.nnei % self.nnei_unit_len > 0):
                    nnei_loop = self.nnei // self.nnei_unit_len
                    nnei_tail = self.nnei % self.nnei_unit_len
                    nnei_tail_fill = _ceil_fill(nnei_tail, Constant.BLOCK_FLOAT32)
                    ndescrpt_tail = nnei_tail_fill * 4
                    deriv_nnei_vcpadd_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                        (3 * nnei_tail_fill * Constant.NLOC_UNIT_LEN + 32,), name="deriv_nnei_vcpadd_ub_fp32",
                        scope=tik.scope_ubuf)
                    # prepare
                    with self.tik_instance.for_range(0, Constant.NLOC_UNIT_LEN) as i:
                        deriv_input_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                       (1, 1, ndescrpt_tail, 3),
                                                                       name="deriv_input_ub_fp32",
                                                                       scope=tik.scope_ubuf)
                        deriv_trans_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                       (1, 3, 1, ndescrpt_tail),
                                                                       name="deriv_trans_ub_fp32",
                                                                       scope=tik.scope_ubuf)
                        deriv_offset = (loop_offset + nloc_offset + i) * self.nnei * 4 + \
                            nnei_loop * self.nnei_unit_len * 4
                        self.tik_instance.data_move(deriv_input_ub_fp32,
                                                    self.in_deriv_gm[deriv_offset * 3],
                                                    0, 1, 3 * ndescrpt_tail // Constant.BLOCK_FLOAT32, 0, 0)
                        self.tik_instance.v4dtrans(False, deriv_trans_ub_fp32[0, 0, 0, 0],
                                                   deriv_input_ub_fp32[0, 0, 0, 0], ndescrpt_tail, 3)
                        self.tik_instance.data_move(deriv_input_ub_fp32,
                                                    self.net_deriv_gm[deriv_offset],
                                                    0, 1, ndescrpt_tail // Constant.BLOCK_FLOAT32, 0, 0)
                        vmul_repeat = ndescrpt_tail // Constant.MASK_FLOAT32
                        vmul_left = ndescrpt_tail % Constant.MASK_FLOAT32
                        if vmul_repeat > 0:
                            self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32,
                                                   deriv_trans_ub_fp32,
                                                   deriv_input_ub_fp32,
                                                   vmul_repeat, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[ndescrpt_tail],
                                                   deriv_trans_ub_fp32[ndescrpt_tail],
                                                   deriv_input_ub_fp32,
                                                   vmul_repeat, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[2 * ndescrpt_tail],
                                                   deriv_trans_ub_fp32[2 * ndescrpt_tail],
                                                   deriv_input_ub_fp32,
                                                   vmul_repeat, 1, 1, 1, 8, 8, 8)
                        if vmul_left > 0:
                            vmul_floor = vmul_repeat * Constant.MASK_FLOAT32
                            self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[vmul_floor],
                                                   deriv_trans_ub_fp32[vmul_floor],
                                                   deriv_input_ub_fp32[vmul_floor],
                                                   1, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[ndescrpt_tail + vmul_floor],
                                                   deriv_trans_ub_fp32[ndescrpt_tail + vmul_floor],
                                                   deriv_input_ub_fp32[vmul_floor],
                                                   1, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[2 * ndescrpt_tail + vmul_floor],
                                                   deriv_trans_ub_fp32[2 * ndescrpt_tail + vmul_floor],
                                                   deriv_input_ub_fp32[vmul_floor],
                                                   1, 1, 1, 1, 8, 8, 8)
                        vcpadd_repeat_1 = (3 * ndescrpt_tail) // Constant.MASK_FLOAT32
                        vcpadd_left_1 = (3 * ndescrpt_tail) % Constant.MASK_FLOAT32
                        if vcpadd_repeat_1 > 0:
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32, deriv_input_ub_fp32,
                                                     deriv_trans_ub_fp32,
                                                     vcpadd_repeat_1, 1, 1, 8)
                        if vcpadd_left_1 > 0:
                            vcpadd_floor = vcpadd_repeat_1 * Constant.MASK_FLOAT32
                            self.tik_instance.vcpadd(vcpadd_left_1, deriv_input_ub_fp32[vcpadd_floor // 2],
                                                     deriv_trans_ub_fp32[vcpadd_floor],
                                                     1, 1, 1, 8)
                        vcpadd_repeat_2 = (nnei_tail * 2) // Constant.MASK_FLOAT32
                        vcpadd_left_2 = (nnei_tail * 2) % Constant.MASK_FLOAT32
                        vcpadd_nloc_offset = (Constant.NLOC_UNIT_LEN + i) * nnei_tail_fill
                        vcpadd_nloc_offset_1 = (Constant.NLOC_UNIT_LEN * 2 + i) * nnei_tail_fill
                        if vcpadd_repeat_2 > 0:
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                     deriv_nnei_vcpadd_ub_fp32[i * nnei_tail_fill],
                                                     deriv_input_ub_fp32,
                                                     vcpadd_repeat_2, 1, 1, 8)
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                     deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset],
                                                     deriv_input_ub_fp32[nnei_tail_fill * 2],
                                                     vcpadd_repeat_2, 1, 1, 8)
                            self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                     deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset_1],
                                                     deriv_input_ub_fp32[nnei_tail_fill * 4],
                                                     vcpadd_repeat_2, 1, 1, 8)
                        if vcpadd_left_2 > 0:
                            vcpadd_offset = vcpadd_repeat_2 * Constant.MASK_FLOAT32
                            self.tik_instance.vcpadd(vcpadd_left_2,
                                                     deriv_nnei_vcpadd_ub_fp32[i * nnei_tail_fill + vcpadd_offset // 2],
                                                     deriv_input_ub_fp32[vcpadd_offset],
                                                     1, 1, 1, 8)
                            self.tik_instance.vcpadd(vcpadd_left_2,
                                                     deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset + vcpadd_offset // 2],
                                                     deriv_input_ub_fp32[nnei_tail_fill * 2 + vcpadd_offset],
                                                     1, 1, 1, 8)
                            self.tik_instance.vcpadd(vcpadd_left_2,
                                deriv_nnei_vcpadd_ub_fp32[vcpadd_nloc_offset_1 + vcpadd_offset // 2],
                                deriv_input_ub_fp32[nnei_tail_fill * 4 + vcpadd_offset],
                                1, 1, 1, 8)

                    # first
                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        force_add_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                            (Constant.NLOC_UNIT_LEN * 3 + Constant.MASK_FLOAT32,), name="force_add_ub_fp32",
                            scope=tik.scope_ubuf)
                        force_assis_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                                                       (Constant.BLOCK_FLOAT32 * 3, ),
                                                                       name="force_assis_ub_fp32",
                                                                       scope=tik.scope_ubuf)
                        self.tik_instance.vector_dup(Constant.MASK_FLOAT32, force_add_ub_fp32, 0, 1, 1, 8)
                        self.tik_instance.vector_dup(Constant.NLOC_UNIT_LEN * 3, force_add_ub_fp32, 0, 1, 1, 8)
                        self.tik_instance.vector_dup(Constant.BLOCK_FLOAT32 * 3, force_assis_ub_fp32, 0, 1, 1, 8)
                        force_vcadd_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                            (Constant.NLOC_UNIT_LEN * 3 + Constant.MASK_FLOAT32,), name="force_vcadd_ub_fp32",
                            scope=tik.scope_ubuf)
                        force_reduce_loop = nnei_tail // Constant.MASK_FLOAT32
                        force_reduce_left = nnei_tail % Constant.MASK_FLOAT32
                        for i in range(0, force_reduce_loop):
                            self.tik_instance.vcadd(Constant.MASK_FLOAT32, force_vcadd_ub_fp32,
                                                    deriv_nnei_vcpadd_ub_fp32[Constant.MASK_FLOAT32 * i],
                                                    Constant.NLOC_UNIT_LEN * 3, 1, 0, nnei_tail_fill, stride_unit=2)
                            self.tik_instance.vadd(Constant.NLOC_UNIT_LEN * 3, force_add_ub_fp32,
                                                   force_add_ub_fp32, force_vcadd_ub_fp32,
                                                   1, 1, 1, 1, 8, 8, 8)
                        if force_reduce_left > 0:
                            force_reduce_floor = _floor_fill(nnei_tail, Constant.MASK_FLOAT32)
                            self.tik_instance.vector_dup(Constant.NLOC_UNIT_LEN * 3, force_vcadd_ub_fp32, 0, 1, 1, 8)
                            self.tik_instance.vcadd(force_reduce_left, force_vcadd_ub_fp32,
                                                    deriv_nnei_vcpadd_ub_fp32[force_reduce_floor],
                                                    Constant.NLOC_UNIT_LEN * 3, 1, 0, nnei_tail_fill, stride_unit=2)
                            self.tik_instance.vadd(Constant.NLOC_UNIT_LEN * 3, force_add_ub_fp32,
                                                   force_add_ub_fp32, force_vcadd_ub_fp32,
                                                   1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.vmuls(Constant.NLOC_UNIT_LEN * 3, force_add_ub_fp32,
                                                force_add_ub_fp32, -1, 1, 1, 1, 8, 8)
                        force_offset = frame_idx * self.nall * 3 + core_idx * self.core_loop_unit + nloc_offset
                        self.tik_instance.vadds(Constant.NLOC_UNIT_LEN, force_assis_ub_fp32,
                                                force_add_ub_fp32, 0, 3, 0, 0, 8,
                                                Constant.NLOC_UNIT_LEN, stride_unit=2)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.force_gm[force_offset],
                                                    force_assis_ub_fp32, 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(self.force_gm[force_offset + self.nall],
                                                    force_assis_ub_fp32[Constant.BLOCK_FLOAT32], 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(self.force_gm[force_offset + self.nall * 2],
                                                    force_assis_ub_fp32[Constant.BLOCK_FLOAT32 * 2], 0, 1, 1, 0, 0)
                        self.tik_instance.set_atomic_add(0)

                    #second
                    with self.tik_instance.for_range(0, 3) as local_idx:
                        nnei_mask_len = _ceil_fill(nnei_tail, Constant.MASK_FLOAT32)
                        with self.tik_instance.for_range(0, Constant.NLOC_UNIT_LEN) as nu_idx:
                            nlist_assis_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                            (nnei_mask_len, ),
                                                                            name="nlist_assis_ub_int32",
                                                                            scope=tik.scope_ubuf)
                            with self.tik_instance.new_stmt_scope(disable_sync=False):
                                nlist_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                          (nnei_mask_len, ),
                                                                          name="nlist_ub_int32",
                                                                          scope=tik.scope_ubuf)
                                nlist_offset = (loop_offset + nloc_offset + nu_idx) * self.nnei + \
                                    nnei_loop * self.nnei_unit_len
                                self.tik_instance.data_move(nlist_ub_int32,
                                                            self.nlist_gm[nlist_offset],
                                                            0, 1, nnei_tail_fill // Constant.BLOCK_FLOAT32, 0, 0)
                                if nnei_tail // Constant.MASK_FLOAT32 > 0:
                                    self.tik_instance.vmuls(Constant.MASK_FLOAT32, nlist_assis_ub_int32,
                                                            nlist_ub_int32, 4,
                                                            nnei_tail // Constant.MASK_FLOAT32, 1, 1, 8, 8)
                                if nnei_tail % Constant.MASK_FLOAT32 > 0:
                                    nlist_floor = _floor_fill(nnei_tail, Constant.MASK_FLOAT32)
                                    self.tik_instance.vmuls(nnei_tail % Constant.MASK_FLOAT32,
                                                            nlist_assis_ub_int32[nlist_floor],
                                                            nlist_ub_int32[nlist_floor],
                                                            4, 1, 1, 1, 8, 8)
                            force_out_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                                                         (self.nall_ub_len, ),
                                                                         name="force_out_ub_fp32",
                                                                         scope=tik.scope_ubuf)
                            full_loop = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) // 255
                            full_left = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) % 255
                            for i in range(0, full_loop):
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                                             force_out_ub_fp32[i * 255 * Constant.MASK_FLOAT32],
                                                             0, 255, 1, 8)
                            if full_left > 0:
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                    force_out_ub_fp32[full_loop * 255 * Constant.MASK_FLOAT32],
                                    0, full_left, 1, 8)
                            offset = Constant.NLOC_UNIT_LEN * nnei_tail_fill * local_idx + nnei_tail_fill * nu_idx
                            self.tik_instance.vscatter(nnei_tail, force_out_ub_fp32,
                                                       deriv_nnei_vcpadd_ub_fp32[offset],
                                                       nlist_assis_ub_int32,
                                                       1, 8, 0, 0, "counter")
                            force_offset_2 = frame_idx * self.nall * 3 + self.nall * local_idx
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.force_gm[force_offset_2],
                                                        force_out_ub_fp32,
                                                        0, 1, _ceil_div(self.nall, Constant.BLOCK_FLOAT32), 0, 0)
                            self.tik_instance.set_atomic_add(0)

            # nloc left
            with self.tik_instance.if_scope(nloc_left > 0):
                nloc_offset = nloc_loop * Constant.NLOC_UNIT_LEN
                # nnei loop
                with self.tik_instance.for_range(0, self.nnei // self.nnei_unit_len) as nnei_idx:
                    ndescrpt = self.nnei_unit_len * 4
                    with self.tik_instance.for_range(0, nloc_left) as l_idx:
                        deriv_nnei_vcpadd_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                            (3 * self.nnei_unit_len + 32,),name="deriv_nnei_vcpadd_ub_fp32",
                            scope=tik.scope_ubuf)
                        # prepare
                        with self.tik_instance.new_stmt_scope(disable_sync=False):
                            deriv_input_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                        (1, 1, ndescrpt, 3),
                                                                        name="deriv_input_ub_fp32",
                                                                        scope=tik.scope_ubuf)
                            deriv_trans_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                        (1, 3, 1, ndescrpt),
                                                                        name="deriv_trans_ub_fp32",
                                                                        scope=tik.scope_ubuf)
                            deriv_offset = (loop_offset + nloc_offset + l_idx) * self.nnei * 4 + \
                                nnei_idx * self.nnei_unit_len * 4
                            self.tik_instance.data_move(deriv_input_ub_fp32,
                                                        self.in_deriv_gm[deriv_offset * 3],
                                                        0, 1, 3 * ndescrpt // Constant.BLOCK_FLOAT32, 0, 0)
                            self.tik_instance.v4dtrans(False, deriv_trans_ub_fp32[0, 0, 0, 0],
                                                    deriv_input_ub_fp32[0, 0, 0, 0], ndescrpt, 3)
                            self.tik_instance.data_move(deriv_input_ub_fp32,
                                                        self.net_deriv_gm[deriv_offset],
                                                        0, 1, ndescrpt // Constant.BLOCK_FLOAT32, 0, 0)
                            vmul_repeat = ndescrpt // Constant.MASK_FLOAT32
                            vmul_left = ndescrpt % Constant.MASK_FLOAT32
                            if vmul_repeat > 0:
                                self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32,
                                                    deriv_trans_ub_fp32,
                                                    deriv_input_ub_fp32,
                                                    vmul_repeat, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[ndescrpt],
                                                    deriv_trans_ub_fp32[ndescrpt],
                                                    deriv_input_ub_fp32,
                                                    vmul_repeat, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[2 * ndescrpt],
                                                    deriv_trans_ub_fp32[2 * ndescrpt],
                                                    deriv_input_ub_fp32,
                                                    vmul_repeat, 1, 1, 1, 8, 8, 8)
                            if vmul_left > 0:
                                vmul_floor = _floor_fill(ndescrpt, Constant.MASK_FLOAT32)
                                self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[vmul_floor],
                                                    deriv_trans_ub_fp32[vmul_floor],
                                                    deriv_input_ub_fp32[vmul_floor],
                                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[ndescrpt + vmul_floor],
                                                    deriv_trans_ub_fp32[ndescrpt + vmul_floor],
                                                    deriv_input_ub_fp32[vmul_floor],
                                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[2 * ndescrpt + vmul_floor],
                                                    deriv_trans_ub_fp32[2 * ndescrpt + vmul_floor],
                                                    deriv_input_ub_fp32[vmul_floor],
                                                    1, 1, 1, 1, 8, 8, 8)
                            vcpadd_repeat_1 = (3 * ndescrpt) // Constant.MASK_FLOAT32
                            vcpadd_left_1 = (3 * ndescrpt) % Constant.MASK_FLOAT32
                            if vcpadd_repeat_1 > 0:
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32, deriv_input_ub_fp32,
                                                        deriv_trans_ub_fp32,
                                                        vcpadd_repeat_1, 1, 1, 8)
                            if vcpadd_left_1 > 0:
                                vcpadd_floor = vcpadd_repeat_1 * Constant.MASK_FLOAT32
                                self.tik_instance.vcpadd(vcpadd_left_1, deriv_input_ub_fp32[vcpadd_floor // 2],
                                                        deriv_trans_ub_fp32[vcpadd_floor],
                                                        1, 1, 1, 8)
                            vcpadd_repeat_2 = (ndescrpt // 2) // Constant.MASK_FLOAT32
                            vcpadd_left_2 = (ndescrpt // 2) % Constant.MASK_FLOAT32
                            if vcpadd_repeat_2 > 0:
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                        deriv_nnei_vcpadd_ub_fp32,
                                                        deriv_input_ub_fp32,
                                                        vcpadd_repeat_2, 1, 1, 8)
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                        deriv_nnei_vcpadd_ub_fp32[self.nnei_unit_len],
                                                        deriv_input_ub_fp32[self.nnei_unit_len * 2],
                                                        vcpadd_repeat_2, 1, 1, 8)
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                        deriv_nnei_vcpadd_ub_fp32[self.nnei_unit_len * 2],
                                                        deriv_input_ub_fp32[self.nnei_unit_len * 4],
                                                        vcpadd_repeat_2, 1, 1, 8)
                            if vcpadd_left_2 > 0:
                                vcpadd_offset = vcpadd_repeat_2 * Constant.MASK_FLOAT32
                                self.tik_instance.vcpadd(vcpadd_left_2,
                                    deriv_nnei_vcpadd_ub_fp32[vcpadd_offset // 2],
                                    deriv_input_ub_fp32[vcpadd_offset],
                                    1, 1, 1, 8)
                                self.tik_instance.vcpadd(vcpadd_left_2,
                                    deriv_nnei_vcpadd_ub_fp32[(self.nnei_unit_len * 2 + vcpadd_offset) // 2],
                                    deriv_input_ub_fp32[self.nnei_unit_len * 2 + vcpadd_offset],
                                    1, 1, 1, 8)
                                self.tik_instance.vcpadd(vcpadd_left_2,
                                    deriv_nnei_vcpadd_ub_fp32[(self.nnei_unit_len * 4 + vcpadd_offset) // 2],
                                    deriv_input_ub_fp32[self.nnei_unit_len * 4 + vcpadd_offset],
                                    1, 1, 1, 8)

                        # first
                        with self.tik_instance.new_stmt_scope(disable_sync=False):
                            force_add_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                (3 + Constant.MASK_FLOAT32, ), name="force_add_ub_fp32",
                                scope=tik.scope_ubuf)
                            force_assis_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                                                        (Constant.BLOCK_FLOAT32 * 3, ),
                                                                        name="force_assis_ub_fp32",
                                                                        scope=tik.scope_ubuf)
                            self.tik_instance.vector_dup(Constant.MASK_FLOAT32, force_add_ub_fp32, 0, 1, 1, 8)
                            self.tik_instance.vector_dup(3, force_add_ub_fp32, 0, 1, 1, 8)
                            self.tik_instance.vector_dup(Constant.BLOCK_FLOAT32 * 3, force_assis_ub_fp32, 0, 1, 1, 8)
                            force_vcadd_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                (3 + Constant.MASK_FLOAT32,), name="force_vcadd_ub_fp32",
                                scope=tik.scope_ubuf)
                            force_reduce_loop = self.nnei_unit_len // Constant.MASK_FLOAT32
                            force_reduce_left = self.nnei_unit_len % Constant.MASK_FLOAT32
                            for i in range(0, force_reduce_loop):
                                self.tik_instance.vcadd(Constant.MASK_FLOAT32, force_vcadd_ub_fp32,
                                    deriv_nnei_vcpadd_ub_fp32[Constant.MASK_FLOAT32 * i],
                                    3, 1, 0, self.nnei_unit_len, stride_unit=2)
                                self.tik_instance.vadd(3, force_add_ub_fp32,
                                    force_add_ub_fp32, force_vcadd_ub_fp32, 1, 1, 1, 1, 8, 8, 8)
                            if force_reduce_left > 0:
                                force_reduce_floor = _floor_fill(self.nnei_unit_len, Constant.MASK_FLOAT32)
                                self.tik_instance.vector_dup(1 * 3, force_vcadd_ub_fp32, 0, 1, 1, 8)
                                self.tik_instance.vcadd(force_reduce_left, force_vcadd_ub_fp32,
                                    deriv_nnei_vcpadd_ub_fp32[force_reduce_floor],
                                    1 * 3, 1, 0, self.nnei_unit_len, stride_unit=2)
                                self.tik_instance.vadd(1 * 3, force_add_ub_fp32,
                                    force_add_ub_fp32, force_vcadd_ub_fp32,
                                    1, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmuls(1 * 3, force_add_ub_fp32,
                                force_add_ub_fp32, -1, 1, 1, 1, 8, 8)
                            force_offset = \
                                frame_idx * self.nall * 3 + core_idx * self.core_loop_unit + nloc_offset + l_idx
                            self.tik_instance.vadds(1, force_assis_ub_fp32,
                                force_add_ub_fp32, 0, 3, 0, 0, 8, 1, stride_unit=2)
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.force_gm[force_offset],
                                                        force_assis_ub_fp32, 0, 1, 1, 0, 0)
                            self.tik_instance.data_move(self.force_gm[force_offset + self.nall],
                                                        force_assis_ub_fp32[Constant.BLOCK_FLOAT32], 0, 1, 1, 0, 0)
                            self.tik_instance.data_move(self.force_gm[force_offset + self.nall * 2],
                                                        force_assis_ub_fp32[Constant.BLOCK_FLOAT32 * 2], 0, 1, 1, 0, 0)
                            self.tik_instance.set_atomic_add(0)

                        #second
                        with self.tik_instance.for_range(0, 3) as local_idx:
                            nlist_assis_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                            (self.nnei_unit_len, ),
                                                                            name="nlist_assis_ub_int32",
                                                                            scope=tik.scope_ubuf)
                            with self.tik_instance.new_stmt_scope(disable_sync=False):
                                nlist_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                        (self.nnei_unit_len, ),
                                                                        name="nlist_ub_int32",
                                                                        scope=tik.scope_ubuf)
                                nlist_offset = (loop_offset + nloc_offset) * self.nnei + \
                                    nnei_idx * self.nnei_unit_len
                                self.tik_instance.data_move(nlist_ub_int32,
                                                            self.nlist_gm[nlist_offset],
                                                            0, 1, self.nnei_unit_len // Constant.BLOCK_FLOAT32, 0, 0)
                                if self.nnei_unit_len // Constant.MASK_FLOAT32 > 0:
                                    self.tik_instance.vmuls(Constant.MASK_FLOAT32, nlist_assis_ub_int32,
                                                            nlist_ub_int32, 4,
                                                            self.nnei_unit_len // Constant.MASK_FLOAT32, 1, 1, 8, 8)
                                if self.nnei_unit_len % Constant.MASK_FLOAT32 > 0:
                                    nlist_floor = _floor_fill(self.nnei_unit_len, Constant.MASK_FLOAT32)
                                    self.tik_instance.vmuls(self.nnei_unit_len % Constant.MASK_FLOAT32,
                                                            nlist_assis_ub_int32[nlist_floor],
                                                            nlist_ub_int32[nlist_floor],
                                                            4, 1, 1, 1, 8, 8)
                            force_out_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                (self.nall_ub_len,), name="force_out_ub_fp32",
                                scope=tik.scope_ubuf)
                            full_loop = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) // 255
                            full_left = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) % 255
                            for i in range(0, full_loop):
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                                            force_out_ub_fp32[i * 255 * Constant.MASK_FLOAT32],
                                                            0, 255, 1, 8)
                            if full_left > 0:
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                    force_out_ub_fp32[full_loop * 255 * Constant.MASK_FLOAT32],
                                    0, full_left, 1, 8)
                            offset = self.nnei_unit_len * local_idx
                            self.tik_instance.vscatter(self.nnei_unit_len, force_out_ub_fp32,
                                                    deriv_nnei_vcpadd_ub_fp32[offset],
                                                    nlist_assis_ub_int32,
                                                    1, 8, 0, 0, "counter")
                            force_offset_2 = frame_idx * self.nall * 3 + self.nall * local_idx
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.force_gm[force_offset_2],
                                                        force_out_ub_fp32,
                                                        0, 1, _ceil_div(self.nall, Constant.BLOCK_FLOAT32), 0, 0)
                            self.tik_instance.set_atomic_add(0)

                # nnei tail
                with self.tik_instance.if_scope(self.nnei % self.nnei_unit_len > 0):
                    nnei_loop = self.nnei // self.nnei_unit_len
                    nnei_tail = self.nnei % self.nnei_unit_len
                    nnei_tail_fill = _ceil_fill(nnei_tail, Constant.BLOCK_FLOAT32)
                    ndescrpt_tail = nnei_tail_fill * 4
                    with self.tik_instance.for_range(0, nloc_left) as l_idx:
                        deriv_nnei_vcpadd_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                            (3 * nnei_tail_fill + 32,),name="deriv_nnei_vcpadd_ub_fp32",
                            scope=tik.scope_ubuf)
                        # prepare
                        with self.tik_instance.new_stmt_scope(disable_sync=False):
                            deriv_input_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                        (1, 1, ndescrpt_tail, 3),
                                                                        name="deriv_input_ub_fp32",
                                                                        scope=tik.scope_ubuf)
                            deriv_trans_ub_fp32 = self.tik_instance.Tensor(self.net_deriv_dtype,
                                                                        (1, 3, 1, ndescrpt_tail),
                                                                        name="deriv_trans_ub_fp32",
                                                                        scope=tik.scope_ubuf)
                            deriv_offset = (loop_offset + nloc_offset + l_idx) * self.nnei * 4 + \
                                nnei_loop * self.nnei_unit_len * 4
                            self.tik_instance.data_move(deriv_input_ub_fp32,
                                                        self.in_deriv_gm[deriv_offset * 3],
                                                        0, 1, 3 * ndescrpt_tail // Constant.BLOCK_FLOAT32, 0, 0)
                            self.tik_instance.v4dtrans(False, deriv_trans_ub_fp32[0, 0, 0, 0],
                                                    deriv_input_ub_fp32[0, 0, 0, 0], ndescrpt_tail, 3)
                            self.tik_instance.data_move(deriv_input_ub_fp32,
                                                        self.net_deriv_gm[deriv_offset],
                                                        0, 1, ndescrpt_tail // Constant.BLOCK_FLOAT32, 0, 0)
                            vmul_repeat = ndescrpt_tail // Constant.MASK_FLOAT32
                            vmul_left = ndescrpt_tail % Constant.MASK_FLOAT32
                            if vmul_repeat > 0:
                                self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32,
                                                    deriv_trans_ub_fp32,
                                                    deriv_input_ub_fp32,
                                                    vmul_repeat, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[ndescrpt_tail],
                                                    deriv_trans_ub_fp32[ndescrpt_tail],
                                                    deriv_input_ub_fp32,
                                                    vmul_repeat, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(Constant.MASK_FLOAT32, deriv_trans_ub_fp32[2 * ndescrpt_tail],
                                                    deriv_trans_ub_fp32[2 * ndescrpt_tail],
                                                    deriv_input_ub_fp32,
                                                    vmul_repeat, 1, 1, 1, 8, 8, 8)
                            if vmul_left > 0:
                                vmul_floor =  vmul_repeat * Constant.MASK_FLOAT32
                                self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[vmul_floor],
                                                    deriv_trans_ub_fp32[vmul_floor],
                                                    deriv_input_ub_fp32[vmul_floor],
                                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[ndescrpt_tail + vmul_floor],
                                                    deriv_trans_ub_fp32[ndescrpt_tail + vmul_floor],
                                                    deriv_input_ub_fp32[vmul_floor],
                                                    1, 1, 1, 1, 8, 8, 8)
                                self.tik_instance.vmul(vmul_left, deriv_trans_ub_fp32[2 * ndescrpt_tail + vmul_floor],
                                                    deriv_trans_ub_fp32[2 * ndescrpt_tail + vmul_floor],
                                                    deriv_input_ub_fp32[vmul_floor],
                                                    1, 1, 1, 1, 8, 8, 8)
                            vcpadd_repeat_1 = (3 * ndescrpt_tail) // Constant.MASK_FLOAT32
                            vcpadd_left_1 = (3 * ndescrpt_tail) % Constant.MASK_FLOAT32
                            if vcpadd_repeat_1 > 0:
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32, deriv_input_ub_fp32,
                                                        deriv_trans_ub_fp32,
                                                        vcpadd_repeat_1, 1, 1, 8)
                            if vcpadd_left_1 > 0:
                                vcpadd_floor = vcpadd_repeat_1 * Constant.MASK_FLOAT32
                                self.tik_instance.vcpadd(vcpadd_left_1, deriv_input_ub_fp32[vcpadd_floor // 2],
                                                        deriv_trans_ub_fp32[vcpadd_floor],
                                                        1, 1, 1, 8)
                            vcpadd_repeat_2 = (nnei_tail * 2) // Constant.MASK_FLOAT32
                            vcpadd_left_2 = (nnei_tail * 2) % Constant.MASK_FLOAT32
                            if vcpadd_repeat_2 > 0:
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                        deriv_nnei_vcpadd_ub_fp32,
                                                        deriv_input_ub_fp32,
                                                        vcpadd_repeat_2, 1, 1, 8)
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                        deriv_nnei_vcpadd_ub_fp32[nnei_tail_fill],
                                                        deriv_input_ub_fp32[nnei_tail_fill * 2],
                                                        vcpadd_repeat_2, 1, 1, 8)
                                self.tik_instance.vcpadd(Constant.MASK_FLOAT32,
                                                        deriv_nnei_vcpadd_ub_fp32[nnei_tail_fill * 2],
                                                        deriv_input_ub_fp32[nnei_tail_fill * 4],
                                                        vcpadd_repeat_2, 1, 1, 8)
                            if vcpadd_left_2 > 0:
                                vcpadd_offset = vcpadd_repeat_2 * Constant.MASK_FLOAT32
                                self.tik_instance.vcpadd(vcpadd_left_2,
                                    deriv_nnei_vcpadd_ub_fp32[vcpadd_offset // 2],
                                    deriv_input_ub_fp32[vcpadd_offset],
                                    1, 1, 1, 8)
                                self.tik_instance.vcpadd(vcpadd_left_2,
                                    deriv_nnei_vcpadd_ub_fp32[(nnei_tail_fill * 2 + vcpadd_offset) // 2],
                                    deriv_input_ub_fp32[nnei_tail_fill * 2 + vcpadd_offset],
                                    1, 1, 1, 8)
                                self.tik_instance.vcpadd(vcpadd_left_2,
                                    deriv_nnei_vcpadd_ub_fp32[(nnei_tail_fill * 4 + vcpadd_offset) // 2],
                                    deriv_input_ub_fp32[nnei_tail_fill * 4 + vcpadd_offset],
                                    1, 1, 1, 8)

                        # first
                        with self.tik_instance.new_stmt_scope(disable_sync=False):
                            force_add_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                (3 + Constant.MASK_FLOAT32, ), name="force_add_ub_fp32",
                                scope=tik.scope_ubuf)
                            force_assis_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                                                        (Constant.BLOCK_FLOAT32 * 3, ),
                                                                        name="force_assis_ub_fp32",
                                                                        scope=tik.scope_ubuf)
                            self.tik_instance.vector_dup(Constant.MASK_FLOAT32, force_add_ub_fp32, 0, 1, 1, 8)
                            self.tik_instance.vector_dup(3, force_add_ub_fp32, 0, 1, 1, 8)
                            self.tik_instance.vector_dup(Constant.BLOCK_FLOAT32 * 3, force_assis_ub_fp32, 0, 1, 1, 8)
                            force_vcadd_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                (3 + Constant.MASK_FLOAT32,), name="force_vcadd_ub_fp32",
                                scope=tik.scope_ubuf)
                            force_reduce_loop = nnei_tail // Constant.MASK_FLOAT32
                            force_reduce_left = nnei_tail % Constant.MASK_FLOAT32
                            for i in range(0, force_reduce_loop):
                                self.tik_instance.vcadd(Constant.MASK_FLOAT32, force_vcadd_ub_fp32,
                                    deriv_nnei_vcpadd_ub_fp32[Constant.MASK_FLOAT32 * i],
                                    3, 1, 0, nnei_tail_fill, stride_unit=2)
                                self.tik_instance.vadd(3, force_add_ub_fp32,
                                    force_add_ub_fp32, force_vcadd_ub_fp32, 1, 1, 1, 1, 8, 8, 8)
                            if force_reduce_left > 0:
                                force_reduce_floor = _floor_fill(nnei_tail, Constant.MASK_FLOAT32)
                                self.tik_instance.vector_dup(1 * 3, force_vcadd_ub_fp32, 0, 1, 1, 8)
                                self.tik_instance.vcadd(force_reduce_left, force_vcadd_ub_fp32,
                                    deriv_nnei_vcpadd_ub_fp32[force_reduce_floor],
                                    1 * 3, 1, 0, nnei_tail_fill, stride_unit=2)
                                self.tik_instance.vadd(1 * 3, force_add_ub_fp32,
                                    force_add_ub_fp32, force_vcadd_ub_fp32,
                                    1, 1, 1, 1, 8, 8, 8)
                            self.tik_instance.vmuls(1 * 3, force_add_ub_fp32,
                                force_add_ub_fp32, -1, 1, 1, 1, 8, 8)
                            force_offset = \
                                frame_idx * self.nall * 3 + core_idx * self.core_loop_unit + nloc_offset + l_idx
                            self.tik_instance.vadds(1, force_assis_ub_fp32,
                                force_add_ub_fp32, 0, 3, 0, 0, 8, 1, stride_unit=2)
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.force_gm[force_offset],
                                                        force_assis_ub_fp32, 0, 1, 1, 0, 0)
                            self.tik_instance.data_move(self.force_gm[force_offset + self.nall],
                                                        force_assis_ub_fp32[Constant.BLOCK_FLOAT32], 0, 1, 1, 0, 0)
                            self.tik_instance.data_move(self.force_gm[force_offset + self.nall * 2],
                                                        force_assis_ub_fp32[Constant.BLOCK_FLOAT32 * 2], 0, 1, 1, 0, 0)
                            self.tik_instance.set_atomic_add(0)

                        #second
                        with self.tik_instance.for_range(0, 3) as local_idx:
                            nnei_mask_len = _ceil_fill(nnei_tail, Constant.MASK_FLOAT32)
                            nlist_assis_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                            (nnei_mask_len, ),
                                                                            name="nlist_assis_ub_int32",
                                                                            scope=tik.scope_ubuf)
                            with self.tik_instance.new_stmt_scope(disable_sync=False):
                                nlist_ub_int32 = self.tik_instance.Tensor(self.nlist_dtype,
                                                                        (nnei_mask_len, ),
                                                                        name="nlist_ub_int32",
                                                                        scope=tik.scope_ubuf)
                                nlist_offset = (loop_offset + nloc_offset) * self.nnei + \
                                    nnei_loop * self.nnei_unit_len
                                self.tik_instance.data_move(nlist_ub_int32,
                                                            self.nlist_gm[nlist_offset],
                                                            0, 1, nnei_tail_fill // Constant.BLOCK_FLOAT32, 0, 0)
                                if nnei_tail // Constant.MASK_FLOAT32 > 0:
                                    self.tik_instance.vmuls(Constant.MASK_FLOAT32, nlist_assis_ub_int32,
                                                            nlist_ub_int32, 4,
                                                            nnei_tail // Constant.MASK_FLOAT32, 1, 1, 8, 8)
                                if nnei_tail % Constant.MASK_FLOAT32 > 0:
                                    nlist_floor = _floor_fill(nnei_tail, Constant.MASK_FLOAT32)
                                    self.tik_instance.vmuls(nnei_tail % Constant.MASK_FLOAT32,
                                                            nlist_assis_ub_int32[nlist_floor],
                                                            nlist_ub_int32[nlist_floor],
                                                            4, 1, 1, 1, 8, 8)
                            force_out_ub_fp32 = self.tik_instance.Tensor(self.force_dtype,
                                (self.nall_ub_len,), name="force_out_ub_fp32",
                                scope=tik.scope_ubuf)
                            full_loop = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) // 255
                            full_left = _ceil_div(self.nall_ub_len, Constant.MASK_FLOAT32) % 255
                            for i in range(0, full_loop):
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                                            force_out_ub_fp32[i * 255 * Constant.MASK_FLOAT32],
                                                            0, 255, 1, 8)
                            if full_left > 0:
                                self.tik_instance.vector_dup(Constant.MASK_FLOAT32,
                                    force_out_ub_fp32[full_loop * 255 * Constant.MASK_FLOAT32],
                                    0, full_left, 1, 8)
                            offset = nnei_tail_fill * local_idx
                            self.tik_instance.vscatter(nnei_tail, force_out_ub_fp32,
                                                    deriv_nnei_vcpadd_ub_fp32[offset],
                                                    nlist_assis_ub_int32,
                                                    1, 8, 0, 0, "counter")
                            force_offset_2 = frame_idx * self.nall * 3 + self.nall * local_idx
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.force_gm[force_offset_2],
                                                        force_out_ub_fp32,
                                                        0, 1, _ceil_div(self.nall, Constant.BLOCK_FLOAT32), 0, 0)
                            self.tik_instance.set_atomic_add(0)

    def prod_force_se_a_operator(self, kernel_name):
        """
        prod_force_se_a_operator
        """
        self._tiling_args()
        self._init_gm_data_fp32()
        self._run_multi_core_loop()
        # Build CCE
        # this "global_variable_link" flag suggest ccec.py do link without "-r" option
        # which will result in global variable in cce file with wrong address
        tbe_context.get_context().add_compile_info("vars", {
            "core_nums": self.core_nums,
            "n_a_sel" : self.n_a_sel,
            "n_r_sel" : self.n_r_sel,
            "split_count" : self.split_count,
            "split_index" : self.split_index
        })
        input_list = [
            self.net_deriv_gm, self.in_deriv_gm, self.nlist_gm, self.natoms_gm
        ]
        output_list = [
            self.force_gm
        ]
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=input_list,
                                   outputs=output_list,
                                   flowtable=(self.tiling_gm,),
                                   config=self.opt_config)

        return self.tik_instance


def _para_dtype_check(args_list):
    (net_deriv, in_deriv, nlist, natoms, force) = args_list
    net_deriv_dtype = net_deriv.get("dtype").lower()
    in_deriv_dtype = in_deriv.get("dtype").lower()
    nlist_dtype = nlist.get("dtype").lower()
    natoms_dtype = natoms.get("dtype").lower()
    force_dtype = force.get("dtype").lower()
    para_check.check_dtype(net_deriv_dtype, ("float32"), param_name="net_deriv")
    para_check.check_dtype(in_deriv_dtype, ("float32"),
                           param_name="in_deriv")
    para_check.check_dtype(nlist_dtype, ("int32"),
                           param_name="nlist")
    para_check.check_dtype(natoms_dtype, ("int32"),
                           param_name="natoms")
    para_check.check_dtype(force_dtype, ("float32"),
                           param_name="force")


@register_operator("ProdForceSeA")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def prod_force_se_a(net_deriv, in_deriv, nlist, natoms,
                    force, n_a_sel, n_r_sel, split_count, split_index,
                    kernel_name="prod_force_se_a"):
    args_list = (net_deriv, in_deriv, nlist, natoms, force)
    _para_dtype_check(args_list)
    net_deriv_dtype = net_deriv.get("dtype").lower()
    in_deriv_dtype = in_deriv.get("dtype").lower()
    nlist_dtype = nlist.get("dtype").lower()
    natoms_dtype = natoms.get("dtype").lower()
    force_dtype = force.get("dtype").lower()
    natoms_shape = natoms.get("shape")
    dtypes = (net_deriv_dtype, in_deriv_dtype, nlist_dtype,
              natoms_dtype, force_dtype)
    shapes = (natoms_shape)
    attrs = (n_a_sel, n_r_sel, split_count, split_index)
    obj = ProdForceSeA(attrs, dtypes, shapes)
    return obj.prod_force_se_a_operator(kernel_name)
