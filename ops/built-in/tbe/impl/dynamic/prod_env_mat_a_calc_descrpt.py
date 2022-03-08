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

prod_env_mat_a_calc_descrpt
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from te.utils import para_check


# 'pylint: disable=too-many-public-methods
class ProdEnvMatACalcDescrpt:
    """
    ProdEnvMatACalcDescrpt class
    """
    DTYPE_BYTES = {"float32": 4, "float16": 2}

    # 'pylint: disable=unused-argument
    def __init__(self, distance, rij_x, rij_y, rij_z, types, natoms, mesh, davg, dstd,
                 sel_a, rcut, rcut_smth,  kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tik = tik
        self.natoms_shape = natoms["shape"]
        self.coord_dtype = distance.get("dtype").lower()
        self.type_dtype = types.get("dtype").lower()

        self.nsample = 1
        self.max_nall = 31040
        self.nnei = sum(sel_a)

        self.max_gm_size = 2 ** 31 - 1

        self.max_nbor_size = 256
        self.type_num = len(sel_a)
        if self.type_num != 2:
            self.max_nbor_size = 1024

        self.sel_a = self.tik_instance.ScalarArray(self.type_dtype, name="sel_a",
                                                   length=self.type_num,
                                                   init_value=sel_a)

        self.sel_a_back_list = [0]
        for i in range(1, self.type_num):
            self.sel_a_back_list.append((sum(sel_a[:i]) + 7) // 8 * 8)

        self.sel_a_back = self.tik_instance.ScalarArray(self.type_dtype, name="sel_a_back",
                                                        length=self.type_num,
                                                        init_value=self.sel_a_back_list)

        self.rcut = rcut_smth
        self.rcut_smth = rcut

        self.coord_dim_num = 3

        self.block_num = 32 // ProdEnvMatACalcDescrpt.DTYPE_BYTES.get(self.coord_dtype)

        self.nnei_align = (self.nnei + self.block_num - 1) // self.block_num * self.block_num

        self.descrpt_size = self.nnei * 4

        self.kernel_name = kernel_name

        self.repeat_once_size = self.block_num * self.block_num
        self.block_stride = 1
        self.repeat_stride = 8
        self.default_mask_fp32 = 64

        self.init_scalar()
        self.init_buffer()

    def init_scalar(self):
        """
        init scalars
        """
        self.cur_type_neighbour_nums = self.tik_instance.Scalar("uint32",
                                                                "cur_type_neighbour_nums",
                                                                init_value=0)
        self.cur_loc_nei_num = self.tik_instance.Scalar("uint32",
                                                        "cur_loc_nei_num",
                                                        init_value=self.max_nbor_size)
        self.int32_type = "int32"
        self.nloc_d = self.tik_instance.Scalar(self.int32_type, "nloc_d", init_value=0)
        self.nall_d = self.tik_instance.Scalar(self.int32_type, "nall_d", init_value=0)
        self.total_nloc = self.tik_instance.Scalar(self.int32_type, "total_nloc", init_value=0)
        self.nnei_once_repeat_nums = self.block_num * self.block_num
        self.nnei_repeat_times = self.nnei // self.nnei_once_repeat_nums

        self.cur_op_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.cur_op_core_num_scalar = self.tik_instance.Scalar(self.int32_type, "cur_op_core_num_scalar",
                                                               init_value=self.cur_op_core_num)

    def init_buffer(self):
        """
        init the gm input buffer and the output buffer
        """
        self.distance_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size], name="distance_gm",
                                                    scope=self.tik.scope_gm)
        self.rij_x_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size], name="rij_x_gm",
                                                 scope=self.tik.scope_gm)
        self.rij_y_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size], name="rij_y_gm",
                                                 scope=self.tik.scope_gm)
        self.rij_z_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size], name="rij_z_gm",
                                                 scope=self.tik.scope_gm)
        self.type_gm = self.tik_instance.Tensor(self.type_dtype, [self.max_gm_size],
                                                 name="type_gm", scope=self.tik.scope_gm)
        self.natoms_gm = self.tik_instance.Tensor(self.type_dtype, self.natoms_shape, name="natoms_gm",
                                                  scope=self.tik.scope_gm)
        self.mesh_gm = self.tik_instance.Tensor(self.type_dtype, [self.max_gm_size], name="mesh_gm",
                                                scope=self.tik.scope_gm)
        self.davg_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size], name="davg_gm",
                                                scope=self.tik.scope_gm)
        self.dstd_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size], name="dstd_gm",
                                                scope=self.tik.scope_gm)

        self.descrpt_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size], name="descrpt_gm",
                                                   scope=self.tik.scope_gm)
        self.descrpt_deriv_gm = self.tik_instance.Tensor(self.coord_dtype, [self.max_gm_size],
                                                         name="descrpt_deriv_gm", scope=self.tik.scope_gm)

    def data_move_to_ub_phase1(self, cur_nsample_index, type_ub):
        """
        copy type data in ub from gm.
        """
        src_offset = self.tik_instance.Scalar(self.int32_type, "src_offset",
                                              init_value=cur_nsample_index * self.nall_d * self.coord_dim_num)
        self.tik_instance.data_move(type_ub[0], self.type_gm[src_offset // 3],
                                    sid=0, nburst=1, burst=self.nall_d // self.block_num,
                                    src_stride=0, dst_stride=0)

    def data_move_avg_to_ub(self, cur_type_idx, davg_ub, dstd_ub):
        """
        copy davg data in ub from gm.
        """
        avg_offset = self.tik_instance.Scalar(self.int32_type, "avg_offset",
                                               init_value=cur_type_idx * self.descrpt_size)
        self.tik_instance.data_move(davg_ub[0], self.davg_gm[avg_offset],
                                    sid=0, nburst=1, burst=self.descrpt_size // self.block_num,
                                    src_stride=0, dst_stride=0)
        self.tik_instance.data_move(dstd_ub[0], self.dstd_gm[avg_offset],
                                    sid=0, nburst=1, burst=self.descrpt_size // self.block_num,
                                    src_stride=0, dst_stride=0)

    def set_last_block_value(self, tensor, last_block_tensor, offset):
        """
        set the last block nums when copy ub to out.
        """
        for i in range(0, self.block_num):
            index = self.nnei * offset - self.block_num + i
            last_block_tensor[i].set_as(tensor[index])

    def data_move_res_to_gm(self, src_tensor, dst_gm, last_block_tensor,
                            block_nums, gm_offset, stride_offset):
        """
        copy res data to gm from ub buffer.
        """
        self.tik_instance.data_move(dst_gm[gm_offset], src_tensor,
                                    sid=0, nburst=1, burst=block_nums,
                                    src_stride=0, dst_stride=0)

        self.set_last_block_value(src_tensor, last_block_tensor, stride_offset)

        self.tik_instance.data_move(dst_gm[gm_offset + self.nnei * stride_offset - self.block_num],
                                    last_block_tensor,
                                    sid=0, nburst=1, burst=1,
                                    src_stride=0, dst_stride=0)

    def data_move_to_gm_last_loc(self, res_descrpt_a_tensor,
                                 res_descrpt_a_deriv_tensor,
                                 cur_nsample_index, cur_nloc_index):
        """
        copy nlis, descrpt, descrpt_deriv to gm.
        """
        last_block_tensor = self.tik_instance.Tensor(self.coord_dtype, [self.block_num],
                                                     name="last_block_tensor", scope=self.tik.scope_ubuf)
        descrpt_offset = self.tik_instance.Scalar(self.int32_type, "rij_dst_offset",
                                                  init_value=cur_nsample_index * self.nloc_d * self.nnei * 4 +
                                                             cur_nloc_index * self.nnei * 4)

        descrpt_move_blocks = (self.nnei * 4) // self.block_num

        self.data_move_res_to_gm(res_descrpt_a_tensor, self.descrpt_gm, last_block_tensor, descrpt_move_blocks,
                                 descrpt_offset, 4)

        descrpt_deriv_offset = self.tik_instance.Scalar(self.int32_type, "descrpt_deriv_offset",
                                                        init_value=cur_nsample_index * self.nloc_d * self.nnei * 12 +
                                                             cur_nloc_index * self.nnei * 12)

        descrpt_deriv_move_blocks = (self.nnei * 12) // self.block_num

        self.data_move_res_to_gm(res_descrpt_a_deriv_tensor, self.descrpt_deriv_gm, last_block_tensor,
                                 descrpt_deriv_move_blocks, descrpt_deriv_offset, 12)

    def data_move_to_gm(self, res_descrpt_a_tensor,
                        res_descrpt_a_deriv_tensor,
                        cur_nsample_index, cur_nloc_index):
        """
        copy nlist descrpt descrpt_derive data to gm from ub buffer when block align.
        """
        descrpt_offset = self.tik_instance.Scalar(self.int32_type, "descrpt_offset",
                                                  init_value=cur_nsample_index * self.nloc_d * self.nnei * 4 +
                                                             cur_nloc_index * self.nnei * 4)

        descrpt_move_blocks = self.tik_instance.Scalar(self.int32_type, "descrpt_move_blocks",
                                                       init_value=(self.nnei * 4 + self.block_num - 1)
                                                                   // self.block_num)
        self.tik_instance.data_move(self.descrpt_gm[descrpt_offset], res_descrpt_a_tensor,
                                    sid=0, nburst=1, burst=descrpt_move_blocks, src_stride=0, dst_stride=0)

        descrpt_deriv_offset = self.tik_instance.Scalar(self.int32_type, "descrpt_deriv_offset",
                                                        init_value=cur_nsample_index * self.nloc_d * self.nnei * 12 +
                                                                   cur_nloc_index * self.nnei * 12)

        descrpt_deriv_move_blocks = self.tik_instance.Scalar(self.int32_type, "descrpt_deriv_move_blocks",
                                                             init_value=(self.nnei * 12 + self.block_num - 1)
                                                                  // self.block_num)
        self.tik_instance.data_move(self.descrpt_deriv_gm[descrpt_deriv_offset], res_descrpt_a_deriv_tensor,
                                    sid=0, nburst=1, burst=descrpt_deriv_move_blocks, src_stride=0, dst_stride=0)

    def simple_trans_fp32_8x_4(self, dst_ub, dst_offset, src_ub, hw):
        """
        Simple transpose fp32 data from (8x, 4) to (4, 8x)
        Suppose:
        (1) A is an array, shape is (8m, n)
        (2) AT is transpose of A, shape is (n, 8m)
        (3) {A1, A2, ... } is slice of A, A1 shape is (8m1, n), A2 shape is (8m2, n), ... , 8m1 + 8m2 + ... = 8m
        (4) A1T is transpose of A1, shape is (n, 8m1), A2T is transpose of A2, shape is (n, 8m2), ...
        then the equation holds: `AT = concatenate({A1T, A2T, ... }, axis=1}`
        """
        src_ub_fp16 = src_ub.reinterpret_cast_to("float16")
        trans_ub_fp16 = self.tik_instance.Tensor("float16", [1024], name="trans_ub_fp16",
                                                 scope=self.tik.scope_ubuf)  # shape: (4, 128)
        dst_ub_fp16 = dst_ub.reinterpret_cast_to("float16")
        if hw >= 128:
            with self.tik_instance.for_range(0, hw // 128, name="trans_idx") as trans_idx:
                src_list0 = [src_ub_fp16[trans_idx * 1024 + 64 * i] for i in range(16)]
                dst_list0 = [trans_ub_fp16[16 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list0, src_list0, 4, 16, 1)

                src_tuples1 = [(trans_ub_fp16[128 * i], trans_ub_fp16[128 * i + 16]) for i in range(8)]
                src_list1 = [i for tup in src_tuples1 for i in tup]
                dst_list1 = [dst_ub_fp16[dst_offset * 2 + trans_idx * 256 + 16 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list1, src_list1, 4, hw // 8, 2)
        if hw % 128 > 0:
            src_tail_ub_fp16 = self.tik_instance.Tensor("float16", [1024], name="src_tail_ub_fp16",
                                                        scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(src_tail_ub_fp16, src_ub_fp16[hw // 128 * 1024], 0, 1,
                                        hw % 128 // 8 * 4, 0, 0)

            src_list0 = [src_tail_ub_fp16[64 * i] for i in range(16)]
            dst_list0 = [trans_ub_fp16[16 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list0, src_list0, 4, 16, 1)

            src_tuples1 = [(trans_ub_fp16[128 * i], trans_ub_fp16[128 * i + 16]) for i in range(8)]
            src_list1 = [i for tup in src_tuples1 for i in tup]
            dst_list1 = [src_tail_ub_fp16[16 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list1, src_list1, 4, hw % 128, 2)

            self.tik_instance.data_move(dst_ub_fp16[dst_offset * 2 + hw // 128 * 256], src_tail_ub_fp16, 0, 4,
                                        hw % 128 // 8, 16 - hw % 128 // 8, hw // 128 * 16)

    def simple_trans_fp32_4_8x(self, dst_ub, src_ub, hw):
        """
        Simple transpose fp32 data from (4, 8x) to (8x, 4)
        Suppose:
        (1) A is an array, shape is (n, 8m)
        (2) AT is transpose of A, shape is (8m, n)
        (3) {A1, A2, ... } is slice of A, A1 shape is (n, 8m1), A2 shape is (n, 8m2), ... , 8m1 + 8m2 + ... = 8m
        (4) A1T is transpose of A1, shape is (8m1, n), A2T is transpose of A2, shape is (8m2, n), ...
        then the equation holds: `AT = concatenate({A1T, A2T, ... }, axis=0}`
        """
        src_ub_fp16 = src_ub.reinterpret_cast_to("float16")
        trans_ub_fp16 = self.tik_instance.Tensor("float16", [1024], name="trans_ub_fp16",
                                                 scope=self.tik.scope_ubuf)  # shape: (4, 128)
        dst_ub_fp16 = dst_ub.reinterpret_cast_to("float16")
        if hw >= 128:
            with self.tik_instance.for_range(0, hw // 128, name="trans_idx") as trans_idx:
                src_list0 = [src_ub_fp16[trans_idx * 256 + 16 * i] for i in range(16)]
                dst_list0 = [trans_ub_fp16[16 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list0, src_list0, 4, 16, hw // 8)

                src_tuples1_left = [(trans_ub_fp16[256 * i], trans_ub_fp16[256 * i + 16]) for i in range(4)]
                src_tuples1_right = [(trans_ub_fp16[256 * i + 32], trans_ub_fp16[256 * i + 48]) for i in range(4)]
                src_list1 = [i for tup in src_tuples1_left + src_tuples1_right for i in tup]
                dst_list1 = [dst_ub_fp16[trans_idx * 1024 + 64 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list1, src_list1, 4, 1, 4)
        if hw % 128 > 0:
            src_tail_ub_fp16 = self.tik_instance.Tensor("float16", [1024], name="src_tail_ub_fp16",
                                                        scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(src_tail_ub_fp16, src_ub_fp16[hw // 128 * 256],
                                        0, 4, hw % 128 // 8, hw // 128 * 16, 0)

            src_list0 = [src_tail_ub_fp16[16 * i] for i in range(16)]
            dst_list0 = [trans_ub_fp16[16 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list0, src_list0, 4, 16, hw % 128 // 8)

            src_tuples1_left = [(trans_ub_fp16[256 * i], trans_ub_fp16[256 * i + 16]) for i in range(4)]
            src_tuples1_right = [(trans_ub_fp16[256 * i + 32], trans_ub_fp16[256 * i + 48]) for i in range(4)]
            src_list1 = [i for tup in src_tuples1_left + src_tuples1_right for i in tup]
            dst_list1 = [src_tail_ub_fp16[64 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list1, src_list1, 4, 1, 4)

            self.tik_instance.data_move(dst_ub_fp16[hw // 128 * 1024], src_tail_ub_fp16,
                                        0, 1, hw % 128 * 4 // 8, 0, 0)

    def simple_trans_fp32_12_8x(self, dst_ub, src_ub, hw):
        """
        Simple transpose fp32 data from (12, 8x) to (8x, 12)
        """
        src_ub_fp16 = src_ub.reinterpret_cast_to("float16")
        trans_ub_fp16 = self.tik_instance.Tensor("float16", [6144], name="trans_ub_fp16",
                                                 scope=self.tik.scope_ubuf)  # shape: (12, 128)
        dst_ub_fp16 = dst_ub.reinterpret_cast_to("float16")
        if hw >= 128:
            with self.tik_instance.for_range(0, hw // 128, name="trans_idx") as trans_idx:
                src_list0 = [src_ub_fp16[trans_idx * 256 + 16 * i] for i in range(16)]
                dst_list0 = [trans_ub_fp16[16 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list0, src_list0, 12, 16, hw // 8)

                src_tuples1 = [(trans_ub_fp16[256 * i], trans_ub_fp16[256 * i + 16]) for i in range(8)]
                src_list1 = [i for tup in src_tuples1 for i in tup]
                dst_list1 = [dst_ub_fp16[trans_idx * 3072 + 192 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list1, src_list1, 4, 3, 4)

                src_tuples2_left = [(trans_ub_fp16[2048 + 256 * i],
                                     trans_ub_fp16[2048 + 256 * i + 16]) for i in range(4)]
                src_tuples2_right = [(trans_ub_fp16[32 + 256 * i],
                                      trans_ub_fp16[32 + 256 * i + 16]) for i in range(4)]
                src_list2 = [i for tup in src_tuples2_left + src_tuples2_right for i in tup]
                dst_list2 = [dst_ub_fp16[trans_idx * 3072 + 16 + 192 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list2, src_list2, 4, 3, 4)

                src_tuples3 = [(trans_ub_fp16[1056 + 256 * i], trans_ub_fp16[1056 + 256 * i + 16]) for i in range(8)]
                src_list3 = [i for tup in src_tuples3 for i in tup]
                dst_list3 = [dst_ub_fp16[trans_idx * 3072 + 32 + 192 * i] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list3, src_list3, 4, 3, 4)
        if hw % 128 > 0:
            src_tail_ub_fp16 = self.tik_instance.Tensor("float16", [6144], name="src_tail_ub_fp16",
                                                        scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(src_tail_ub_fp16, src_ub_fp16[hw // 128 * 256],
                                        0, 12, hw % 128 // 8, hw // 128 * 16, (128 - hw % 128) // 8)

            src_list0 = [src_tail_ub_fp16[16 * i] for i in range(16)]
            dst_list0 = [trans_ub_fp16[16 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list0, src_list0, 12, 16, 16)

            src_tuples1 = [(trans_ub_fp16[256 * i], trans_ub_fp16[256 * i + 16]) for i in range(8)]
            src_list1 = [i for tup in src_tuples1 for i in tup]
            dst_list1 = [src_tail_ub_fp16[192 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list1, src_list1, 4, 3, 4)

            src_tuples2_left = [(trans_ub_fp16[2048 + 256 * i], trans_ub_fp16[2048 + 256 * i + 16]) for i in range(4)]
            src_tuples2_right = [(trans_ub_fp16[32 + 256 * i],
                                  trans_ub_fp16[32 + 256 * i + 16]) for i in range(4)]
            src_list2 = [i for tup in src_tuples2_left + src_tuples2_right for i in tup]
            dst_list2 = [src_tail_ub_fp16[16 + 192 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list2, src_list2, 4, 3, 4)

            src_tuples3 = [(trans_ub_fp16[1056 + 256 * i], trans_ub_fp16[1056 + 256 * i + 16]) for i in range(8)]
            src_list3 = [i for tup in src_tuples3 for i in tup]
            dst_list3 = [src_tail_ub_fp16[32 + 192 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list3, src_list3, 4, 3, 4)

            self.tik_instance.data_move(dst_ub_fp16[hw // 128 * 3072], src_tail_ub_fp16,
                                        0, 1, hw % 128 * 12 // 8, 0, 0)

    def simple_move_fp32(self, dst_ub, dst_offset, src_ub, src_offset, counter):
        """
        simple move fp32 data from ub to ub by vadds
        """
        if counter >= self.default_mask_fp32:
            self.tik_instance.vadds(self.default_mask_fp32, dst_ub[dst_offset], src_ub[src_offset],
                                    0, counter // self.default_mask_fp32,
                                    self.block_stride, self.block_stride, self.repeat_stride, self.repeat_stride)
        if counter % self.default_mask_fp32 > 0:
            self.tik_instance.vadds(counter % self.default_mask_fp32,
                                    dst_ub[dst_offset + counter // self.default_mask_fp32 * self.default_mask_fp32],
                                    src_ub[src_offset + counter // self.default_mask_fp32 * self.default_mask_fp32],
                                    0, 1, self.block_stride, self.block_stride, self.repeat_stride, self.repeat_stride)

    def do_avg_process(self, descrpt_tensor, descrpt_deriv_tensor, cur_type_idx):
        """
        do avg std process for descrpt and descrpt_deriv.
        """
        nnei_repeat_align = (self.nnei + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums \
                            * self.nnei_once_repeat_nums
        dstd_ub = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 12],
                                           name="dstd_ub", scope=self.tik.scope_ubuf)
        davg_ub = dstd_ub[nnei_repeat_align * 4]

        self.data_move_avg_to_ub(cur_type_idx, davg_ub, dstd_ub)

        descrpt_align = self.descrpt_size // self.nnei_once_repeat_nums
        descrpt_tail = self.descrpt_size % self.nnei_once_repeat_nums

        self.tik_instance.vsub(self.nnei_once_repeat_nums, descrpt_tensor, descrpt_tensor, davg_ub, descrpt_align,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if descrpt_tail != 0:
            self.tik_instance.vsub(descrpt_tail, descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   dstd_ub[nnei_repeat_align * 4 + descrpt_align * self.nnei_once_repeat_nums], 1,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_tensor, descrpt_tensor, dstd_ub, descrpt_align,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if descrpt_tail != 0:
            self.tik_instance.vdiv(descrpt_tail, descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   descrpt_tensor[descrpt_align * self.nnei_once_repeat_nums],
                                   dstd_ub[descrpt_align * self.nnei_once_repeat_nums], 1,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.simple_trans_fp32_8x_4(dstd_ub, nnei_repeat_align * 12 - self.nnei_align * 3 - nnei_repeat_align,
                                    dstd_ub, self.nnei_align)

        for des in range(3):
            for nn in range(3):
                self.simple_move_fp32(dstd_ub, des * 3 * nnei_repeat_align + nn * nnei_repeat_align, dstd_ub,
                                      nnei_repeat_align * 12 - self.nnei_align * 3 - nnei_repeat_align +
                                          des * self.nnei_align,
                                      self.nnei)

        self.simple_move_fp32(dstd_ub, 9 * nnei_repeat_align, dstd_ub, nnei_repeat_align * 11, self.nnei)
        self.simple_move_fp32(dstd_ub, 10 * nnei_repeat_align, dstd_ub, nnei_repeat_align * 11, self.nnei)

        descrpt_deriv_align = (nnei_repeat_align * 12) // self.nnei_once_repeat_nums
        descrpt_deriv_align_tail = 0

        max_repeat_times = 255
        if descrpt_deriv_align > max_repeat_times:
            descrpt_deriv_align_tail = descrpt_deriv_align % max_repeat_times
            descrpt_deriv_align = max_repeat_times

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_deriv_tensor, descrpt_deriv_tensor, dstd_ub,
                               descrpt_deriv_align, self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        if descrpt_deriv_align_tail != 0:
            self.tik_instance.vdiv(self.nnei_once_repeat_nums,
                                   descrpt_deriv_tensor[descrpt_deriv_align * self.nnei_once_repeat_nums],
                                   descrpt_deriv_tensor[descrpt_deriv_align * self.nnei_once_repeat_nums],
                                   dstd_ub[descrpt_deriv_align * self.nnei_once_repeat_nums],
                                   descrpt_deriv_align_tail, self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            descrpt_a_deriv_tensor = dstd_ub
            for deriv_idx in range(12):
                self.simple_move_fp32(descrpt_a_deriv_tensor, self.nnei_align * deriv_idx,
                                      descrpt_deriv_tensor, nnei_repeat_align * deriv_idx, self.nnei_align)
            self.simple_trans_fp32_12_8x(descrpt_deriv_tensor, descrpt_a_deriv_tensor, self.nnei_align)

    def concat_result(self, res_descrpt_a_tensor, nnei_repeat_times, free_buffer):
        """
        concat rij descrpt descrpt_deriv data in res ub buffer.
        """
        nnei_align = (self.nnei + self.block_num - 1) // self.block_num * self.block_num

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            descrpt_a_tensor = free_buffer

            self.simple_move_fp32(descrpt_a_tensor, 0, res_descrpt_a_tensor, 0, nnei_align)

            self.simple_move_fp32(descrpt_a_tensor, nnei_align,
                                  res_descrpt_a_tensor, nnei_repeat_times * self.nnei_once_repeat_nums,
                                  nnei_align)

            self.simple_move_fp32(descrpt_a_tensor, nnei_align * 2,
                                  res_descrpt_a_tensor, nnei_repeat_times * self.nnei_once_repeat_nums * 2,
                                  nnei_align)

            self.simple_move_fp32(descrpt_a_tensor, nnei_align * 3,
                                  res_descrpt_a_tensor, nnei_repeat_times * self.nnei_once_repeat_nums * 3,
                                  nnei_align)
            self.simple_trans_fp32_4_8x(res_descrpt_a_tensor, descrpt_a_tensor, self.nnei_align)

    def compute_descrpt_a_deriv_three(self, left_value_save, rr0_factor_tensor,
                                      rr0_tensor, rr1_tensor, rr2_tensor,
                                      sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                      descrpt_a_1, dsw_tensor, dis_rev_tensor,
                                      not_one, descrpt_a_deriv_3, descrpt_a_deriv_4,
                                      descrpt_a_deriv_5):
        """
        compute the group res for deriv.
        """
        compute_nums = self.nnei
        nnei_repeat_times = (compute_nums + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums

        self.tik_instance.vmuls(self.nnei_once_repeat_nums, left_value_save, rr0_factor_tensor, 2,
                                nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, left_value_save, sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, left_value_save, res_vec_tensor_2,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_3, descrpt_a_1, dsw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_3, descrpt_a_deriv_3, dis_rev_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmuls(self.nnei_once_repeat_nums, descrpt_a_deriv_3, descrpt_a_deriv_3, not_one,
                                nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadd(self.nnei_once_repeat_nums, left_value_save, left_value_save, descrpt_a_deriv_3,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_3, left_value_save, rr0_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_4, left_value_save, rr1_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_5, left_value_save, rr2_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, sw_tensor, res_vec_tensor_1,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vsub(self.nnei_once_repeat_nums, descrpt_a_deriv_3, descrpt_a_deriv_3, left_value_save,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

    def vector_compute_last_process(self, dis_rev_tensor, neighbour_coords,
                                    dis_dot_tensor,
                                    sw_tensor, dsw_tensor,
                                    res_vec_tensor, res_vec_tensor_1, res_vec_tensor_2, res_vec_tensor_3,
                                    res_descrpt_a_tensor,
                                    res_descrpt_a_deriv_tensor, free_buffer):
        """
        compute descrpt and descrpt_deriv last process.
        """
        compute_nums = self.nnei
        nnei_repeat_times = (compute_nums + 63) // self.nnei_once_repeat_nums

        descrpt_a_0 = res_descrpt_a_tensor[0]
        descrpt_a_1 = res_descrpt_a_tensor[nnei_repeat_times * self.nnei_once_repeat_nums]
        descrpt_a_2 = res_descrpt_a_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 2]
        descrpt_a_3 = res_descrpt_a_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 3]

        self.tik_instance.vrec(self.nnei_once_repeat_nums, descrpt_a_0, res_vec_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_a_1, neighbour_coords[0],
                               dis_dot_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_a_2, neighbour_coords[1],
                               dis_dot_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vdiv(self.nnei_once_repeat_nums, descrpt_a_3, neighbour_coords[2],
                               dis_dot_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        left_value_save = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_times * self.nnei_once_repeat_nums],
                                                   name="left_value_save", scope=self.tik.scope_ubuf)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, left_value_save, res_vec_tensor_3,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        descrpt_a_deriv_0 = res_descrpt_a_deriv_tensor[0]
        descrpt_a_deriv_1 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums]
        descrpt_a_deriv_2 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 2]
        descrpt_a_deriv_3 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 3]
        descrpt_a_deriv_4 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 4]
        descrpt_a_deriv_5 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 5]
        descrpt_a_deriv_6 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 6]
        descrpt_a_deriv_7 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 7]
        descrpt_a_deriv_8 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 8]
        descrpt_a_deriv_9 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 9]
        descrpt_a_deriv_10 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 10]
        descrpt_a_deriv_11 = res_descrpt_a_deriv_tensor[nnei_repeat_times * self.nnei_once_repeat_nums * 11]

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_0, descrpt_a_0,
                               dsw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_0, descrpt_a_deriv_0,
                               dis_rev_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        not_one = self.tik_instance.Scalar("float32", "not_one", init_value=-1)

        self.tik_instance.vmuls(self.nnei_once_repeat_nums, descrpt_a_deriv_0,
                                descrpt_a_deriv_0,
                                not_one,
                                nnei_repeat_times,
                                self.block_stride, self.block_stride,
                                self.repeat_stride, self.repeat_stride)

        self.tik_instance.vadd(self.nnei_once_repeat_nums, left_value_save, left_value_save,
                               descrpt_a_deriv_0,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_0, left_value_save,
                               neighbour_coords[0],
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_1, left_value_save,
                               neighbour_coords[1],
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_deriv_2, left_value_save,
                               neighbour_coords[2],
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        # compute res 456
        self.compute_descrpt_a_deriv_three(left_value_save, neighbour_coords[0],
                                           neighbour_coords[0], neighbour_coords[1], neighbour_coords[2],
                                           sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                           descrpt_a_1, dsw_tensor, dis_rev_tensor,
                                           not_one, descrpt_a_deriv_3, descrpt_a_deriv_4,
                                           descrpt_a_deriv_5)

        # compute res 789
        self.compute_descrpt_a_deriv_three(left_value_save, neighbour_coords[1],
                                           neighbour_coords[1], neighbour_coords[0], neighbour_coords[2],
                                           sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                           descrpt_a_2, dsw_tensor, dis_rev_tensor,
                                           not_one, descrpt_a_deriv_7, descrpt_a_deriv_6,
                                           descrpt_a_deriv_8)

        # compute res 10 11 12
        self.compute_descrpt_a_deriv_three(left_value_save, neighbour_coords[2],
                                           neighbour_coords[2], neighbour_coords[0], neighbour_coords[1],
                                           sw_tensor, res_vec_tensor_2, res_vec_tensor_1,
                                           descrpt_a_3, dsw_tensor, dis_rev_tensor,
                                           not_one, descrpt_a_deriv_11, descrpt_a_deriv_9,
                                           descrpt_a_deriv_10)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_0, descrpt_a_0,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_1, descrpt_a_1,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_2, descrpt_a_2,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, descrpt_a_3, descrpt_a_3,
                               sw_tensor,
                               nnei_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.concat_result(res_descrpt_a_tensor, nnei_repeat_times, free_buffer)

    def vector_compute_process(self, dis_tensor, dis_dot_tensor,
                               sw_tensor, dsw_tensor, res_vec_tensor, res_vec_tensor_1,
                               res_vec_tensor_2, res_vec_tensor_3, dis_revert_tensor,
                               res_descrpt_a_deriv_tensor, sorted_coords,
                               rcut_ub, free_buffer, nnei_repeat_align):
        """
        the first vec compute process for descrpt and descrpt_deriv.
        """
        compute_nums = self.nnei
        nnei_mask_repeat = (compute_nums + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums

        self.tik_instance.vec_dup(self.nnei_once_repeat_nums, sw_tensor, 0,
                                  nnei_mask_repeat, self.repeat_stride)

        self.tik_instance.vec_dup(self.nnei_once_repeat_nums, dsw_tensor, 0,
                                  nnei_mask_repeat, self.repeat_stride)

        self.simple_move_fp32(free_buffer, nnei_repeat_align * 2,
                              dis_tensor, 0, compute_nums)  # dis_tensor -> res_vec_tensor

        self.tik_instance.vrec(self.nnei_once_repeat_nums, dis_revert_tensor, dis_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vrec(self.nnei_once_repeat_nums, res_vec_tensor_1, dis_dot_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, res_vec_tensor_2, dis_dot_tensor,
                               dis_dot_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        self.tik_instance.vrec(self.nnei_once_repeat_nums, res_vec_tensor_2, res_vec_tensor_2,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vmul(self.nnei_once_repeat_nums, res_vec_tensor_3, res_vec_tensor_1,
                               dis_revert_tensor,
                               nnei_mask_repeat,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        mask_length = (compute_nums + self.nnei_once_repeat_nums - 1) // self.nnei_once_repeat_nums

        mask_tensor = self.tik_instance.Tensor("uint64", [mask_length],
                                               name="mask_tensor", scope=self.tik.scope_ubuf)

        mask_tensor_1 = self.tik_instance.Tensor("uint64", [mask_length],
                                                 name="mask_tensor_1", scope=self.tik.scope_ubuf)

        mask_tensor_2 = self.tik_instance.Tensor("uint64", [mask_length],
                                                 name="mask_tensor_2", scope=self.tik.scope_ubuf)

        mask_tensor_3 = self.tik_instance.Tensor("uint64", [mask_length],
                                                 name="mask_tensor_3", scope=self.tik.scope_ubuf)

        self.tik_instance.vec_dup(8, rcut_ub, self.rcut, 1, 0)
        self.tik_instance.vcmpv_lt(mask_tensor, res_vec_tensor, rcut_ub, mask_length,
                                   self.block_stride, 0, self.repeat_stride, 0)

        self.tik_instance.vec_dup(8, rcut_ub, self.rcut_smth, 1, 0)
        self.tik_instance.vcmpv_ge(mask_tensor_1, res_vec_tensor, rcut_ub, mask_length,
                                   self.block_stride, 0, self.repeat_stride, 0)

        mask_tensor_2_int16 = mask_tensor_2.reinterpret_cast_to("int16")
        mask_tensor_int16 = mask_tensor.reinterpret_cast_to("int16")

        mask_repeat_times = 1
        self.tik_instance.vnot(mask_length * 4, mask_tensor_2_int16, mask_tensor_int16,
                               mask_repeat_times,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        mask_tensor_3_int16 = mask_tensor_3.reinterpret_cast_to("int16")
        mask_tensor_1_int16 = mask_tensor_1.reinterpret_cast_to("int16")
        self.tik_instance.vnot(mask_length * 4, mask_tensor_3_int16, mask_tensor_1_int16,
                               mask_repeat_times,
                               self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride)

        self.tik_instance.vand(mask_length * 4, mask_tensor_2_int16, mask_tensor_2_int16, mask_tensor_3_int16,
                               mask_repeat_times,
                               self.block_stride, self.block_stride, self.block_stride,
                               self.repeat_stride, self.repeat_stride, self.repeat_stride)

        mask_left = self.tik_instance.Scalar(dtype="uint64", name="mask_left", init_value=0)
        mask_right = self.tik_instance.Scalar(dtype="uint64", name="mask_right", init_value=0)
        mask_right_one = self.tik_instance.Scalar(dtype="uint64", name="mask_right_one", init_value=0)
        mask_nlist_one = self.tik_instance.Scalar(dtype="uint64", name="mask_nlist_one", init_value=0)

        uu_tensor = res_descrpt_a_deriv_tensor[0]

        uu_tensor_back = res_descrpt_a_deriv_tensor[nnei_mask_repeat * self.nnei_once_repeat_nums]

        uu_tensor_pow = res_descrpt_a_deriv_tensor[nnei_mask_repeat * self.nnei_once_repeat_nums * 2]

        revert_min = self.tik_instance.Scalar(dtype="float32", name="revert_min",
                                              init_value=self.rcut * (-1))

        rev_max_sub_min = self.tik_instance.Scalar(dtype="float32", name="rev_max_sub_min",
                                                   init_value=1 / (self.rcut_smth - self.rcut))

        not_six = self.tik_instance.Scalar(dtype="float32", name="not_six",
                                           init_value=6 * (-1))

        not_twl = self.tik_instance.Scalar(dtype="float32", name="not_twl",
                                           init_value=12 * (-1))

        not_ten = self.tik_instance.Scalar(dtype="float32", name="not_ten",
                                           init_value=10 * (-1))

        fifthen = self.tik_instance.Scalar(dtype="float32", name="fifthen",
                                           init_value=15)

        with self.tik_instance.for_range(0, mask_length) as i:
            mask_right.set_as(mask_tensor_2[i])
            mask_right_one.set_as(mask_tensor[i])
            mask_nlist_one.set_as(mask_tensor_1[i])
            with self.tik_instance.if_scope(mask_right != 0):
                self.tik_instance.vadds([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        res_vec_tensor[self.nnei_once_repeat_nums * i],
                                        revert_min,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        rev_max_sub_min,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], sw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                        uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                        not_six,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor_back[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        15,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadd([mask_left, mask_right], uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadds([mask_left, mask_right], uu_tensor_back[self.nnei_once_repeat_nums * i],
                                        uu_tensor_back[self.nnei_once_repeat_nums * i],
                                        not_ten,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], sw_tensor[self.nnei_once_repeat_nums * i],
                                       sw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadds([mask_left, mask_right], sw_tensor[self.nnei_once_repeat_nums * i],
                                        sw_tensor[self.nnei_once_repeat_nums * i],
                                        1,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                        3,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                       dsw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_back[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        not_twl,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadds([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                        uu_tensor[self.nnei_once_repeat_nums * i],
                                        fifthen,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmul([mask_left, mask_right], uu_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor_pow[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadd([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                       dsw_tensor[self.nnei_once_repeat_nums * i],
                                       uu_tensor[self.nnei_once_repeat_nums * i],
                                       1,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)

                self.tik_instance.vmuls([mask_left, mask_right], dsw_tensor[self.nnei_once_repeat_nums * i],
                                        dsw_tensor[self.nnei_once_repeat_nums * i],
                                        rev_max_sub_min,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

            with self.tik_instance.if_scope(mask_right_one != 0):
                self.tik_instance.vadds([mask_left, mask_right_one], sw_tensor[self.nnei_once_repeat_nums * i],
                                        sw_tensor[self.nnei_once_repeat_nums * i],
                                        1,
                                        1,
                                        self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)

            with self.tik_instance.if_scope(mask_nlist_one != 0):
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             sorted_coords[0][self.nnei_once_repeat_nums * i], -1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             sorted_coords[1][self.nnei_once_repeat_nums * i], -1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             sorted_coords[2][self.nnei_once_repeat_nums * i], -1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             dis_dot_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             dis_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             dis_revert_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor_1[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor_2[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)
                self.tik_instance.vector_dup([mask_left, mask_nlist_one],
                                             res_vec_tensor_3[self.nnei_once_repeat_nums * i], 1,
                                             1, self.block_stride,
                                             self.repeat_stride)

    def apply_ub_tensor_phase1(self):
        """
        Apply ub buffer for all neighbours type.
        """
        type_ub = self.tik_instance.Tensor(self.type_dtype, [self.max_nall],
                                           name="type_ub", scope=self.tik.scope_ubuf)

        return type_ub

    def get_core_nums(self):
        """
        Get core nums for comppute process.
        """
        one_cor_nloc_num = self.tik_instance.Scalar(self.int32_type, "one_cor_nloc_num",
                                                    init_value=self.nloc_d // self.cur_op_core_num)

        last_core_num = self.tik_instance.Scalar(self.int32_type, "last_core_num",
                                                 init_value=self.nloc_d -
                                                            one_cor_nloc_num * (self.cur_op_core_num - 1))

        with self.tik_instance.if_scope(self.nloc_d < self.cur_op_core_num):
            self.cur_op_core_num_scalar.set_as(self.nloc_d)
            one_cor_nloc_num.set_as(self.nloc_d - 1)
            last_core_num.set_as(1)

        return one_cor_nloc_num, last_core_num

    def gm_data_to_ub(self, gm_tensor, ub_tensor, gm_offset, default_value,
                      extract_length, extract_length_align, rep_tail_ub):
        if extract_length == extract_length_align:
            self.tik_instance.data_move(ub_tensor, gm_tensor[gm_offset], 0, 1,
                                        extract_length_align // self.repeat_stride, 0, 0)
        else:
            align_num = extract_length_align - self.repeat_once_size
            tail_num = extract_length - align_num

            if extract_length > self.repeat_once_size:
                self.tik_instance.data_move(ub_tensor, gm_tensor[gm_offset], 0, 1,
                                            align_num // self.block_num, 0, 0)
            self.tik_instance.vector_dup(self.repeat_once_size, ub_tensor[align_num], default_value,
                                            1, self.block_stride, self.repeat_stride)
            self.tik_instance.data_move(rep_tail_ub, gm_tensor[gm_offset + align_num], 0, 1,
                                        (tail_num + self.block_num - 1) // self.block_num, 0, 0)
            self.tik_instance.vadds(tail_num, ub_tensor[align_num], rep_tail_ub, 0, 1,
                                    self.block_stride, self.block_stride,
                                    self.repeat_stride, self.repeat_stride)

    def compute_one_core(self, start_loc, stop_loc, block_idx, cur_sample, one_cor_nloc_num):
        """
        Compute process, the main fuc.
        """
        with self.tik_instance.for_range(start_loc, stop_loc) as loc_nei_idx:
            extract_length = max(self.max_nbor_size, self.nnei)
            extract_length_align = (extract_length + self.nnei_once_repeat_nums - 1) // \
                                   self.nnei_once_repeat_nums * self.nnei_once_repeat_nums

            sorted_neighbour_coord_x = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                                name="sorted_neighbour_coord_x",
                                                                scope=self.tik.scope_ubuf)

            sorted_neighbour_coord_y = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                                name="sorted_neighbour_coord_y",
                                                                scope=self.tik.scope_ubuf)

            sorted_neighbour_coord_z = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                                name="sorted_neighbour_coord_z",
                                                                scope=self.tik.scope_ubuf)

            rep_tail_ub = self.tik_instance.Tensor(self.coord_dtype, [self.nnei_once_repeat_nums],
                                                   name="rij_coord_ub", scope=self.tik.scope_ubuf)

            sorted_dis_tensor = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                         name="sorted_dis_tensor",
                                                         scope=self.tik.scope_ubuf)

            sorted_dis_dot_tensor = self.tik_instance.Tensor(self.coord_dtype, [extract_length_align],
                                                             name="sorted_dis_dot_tensor",
                                                             scope=self.tik.scope_ubuf)
            init_dis = self.tik_instance.Scalar("float32", "init_dis",
                                                init_value=self.rcut_smth + 1)

            cur_loc = self.tik_instance.Scalar(self.int32_type, "loc_idx",
                                               init_value=block_idx * one_cor_nloc_num + loc_nei_idx)

            for_input_cur_loc = self.tik_instance.Scalar(self.int32_type, "for_input_cur_loc",
                                                         init_value=block_idx * one_cor_nloc_num + loc_nei_idx)

            cur_loc_index = self.tik_instance.Scalar(self.int32_type, "cur_loc_index",
                                                     init_value=for_input_cur_loc + 1)
            mesh_ub = self.tik_instance.Tensor(self.type_dtype, [8], name="mesh_ub", scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(mesh_ub, self.mesh_gm[cur_loc_index], 0, 1, 1, 0, 0)
            cur_loc_index.set_as(mesh_ub[0])

            cur_loc_type_id = self.tik_instance.Scalar(self.int32_type, "cur_loc_type_id",
                                                       init_value=0)
            nei_num_index = self.tik_instance.Scalar(self.int32_type, "nei_num_index",
                                                     init_value=1 + self.total_nloc + for_input_cur_loc)
            self.tik_instance.data_move(mesh_ub, self.mesh_gm[nei_num_index], 0, 1, 1, 0, 0)
            self.cur_loc_nei_num.set_as(mesh_ub[0])
            with self.tik_instance.if_scope(self.cur_loc_nei_num != -1):
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    type_ub = self.apply_ub_tensor_phase1()
                    self.data_move_to_ub_phase1(cur_sample, type_ub)
                    cur_loc_type_id.set_as(type_ub[cur_loc_index])

            nn_offset = self.tik_instance.Scalar(self.int32_type, "nn_offset",
                                                 init_value=cur_sample * self.nloc_d * self.nnei + cur_loc * self.nnei)
            self.gm_data_to_ub(self.rij_x_gm, sorted_neighbour_coord_x, nn_offset, -1,
                               extract_length, extract_length_align, rep_tail_ub)
            self.gm_data_to_ub(self.rij_y_gm, sorted_neighbour_coord_y, nn_offset, -1,
                               extract_length, extract_length_align, rep_tail_ub)
            self.gm_data_to_ub(self.rij_z_gm, sorted_neighbour_coord_z, nn_offset, -1,
                               extract_length, extract_length_align, rep_tail_ub)
            self.gm_data_to_ub(self.distance_gm, sorted_dis_tensor, nn_offset, init_dis,
                               extract_length, extract_length_align, rep_tail_ub)
            self.tik_instance.vmul(self.repeat_once_size, sorted_dis_dot_tensor, sorted_dis_tensor, sorted_dis_tensor,
                                   extract_length_align // self.nnei_once_repeat_nums,
                                   self.block_stride, self.block_stride, self.block_stride,
                                   self.repeat_stride, self.repeat_stride, self.repeat_stride)

            nnei_repeat_align = (self.nnei + self.repeat_once_size - 1) // self.repeat_once_size * self.repeat_once_size
            res_descrpt_a_tensor = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 4],
                                                            name="res_descrpt_a_tensor",
                                                            scope=self.tik.scope_ubuf)

            res_descrpt_a_deriv_tensor = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 12],
                                                                  name="res_descrpt_a_deriv_tensor",
                                                                  scope=self.tik.scope_ubuf)
            with self.tik_instance.if_scope(self.cur_loc_nei_num == -1):
                self.tik_instance.vector_dup(self.repeat_once_size, res_descrpt_a_tensor, 0,
                                             nnei_repeat_align * 4 // self.nnei_once_repeat_nums, self.block_stride,
                                             self.repeat_stride)
                vd_repeat_times = nnei_repeat_align * 12 // self.nnei_once_repeat_nums
                max_repeat_times = 255
                if vd_repeat_times > max_repeat_times:
                    tail_repeat_times = vd_repeat_times - max_repeat_times
                    self.tik_instance.vector_dup(self.repeat_once_size, res_descrpt_a_deriv_tensor, 0,
                                                 max_repeat_times, self.block_stride,
                                                 self.repeat_stride)
                    self.tik_instance.vector_dup(self.repeat_once_size,
                                                 res_descrpt_a_deriv_tensor[max_repeat_times * self.repeat_once_size],
                                                 0,
                                                 tail_repeat_times, self.block_stride,
                                                 self.repeat_stride)
                else:
                    self.tik_instance.vector_dup(self.repeat_once_size, res_descrpt_a_deriv_tensor, 0,
                                                 vd_repeat_times, self.block_stride,
                                                 self.repeat_stride)
            with self.tik_instance.else_scope():
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    free_buffer = self.tik_instance.Tensor(self.coord_dtype, [nnei_repeat_align * 12],
                                                           name="free_buffer",
                                                           scope=self.tik.scope_ubuf)
                    rcut_ub = self.tik_instance.Tensor(self.coord_dtype, [8], name="rcut_ub",
                                                       scope=self.tik.scope_ubuf)

                    sw_tensor = free_buffer[0]
                    dsw_tensor = free_buffer[nnei_repeat_align]
                    res_vec_tensor = free_buffer[nnei_repeat_align * 2]
                    res_vec_tensor_1 = free_buffer[nnei_repeat_align * 3]
                    res_vec_tensor_2 = free_buffer[nnei_repeat_align * 4]
                    res_vec_tensor_3 = free_buffer[nnei_repeat_align * 5]
                    dis_revert_tensor = free_buffer[nnei_repeat_align * 6]

                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        self.vector_compute_process(sorted_dis_tensor,
                                                    sorted_dis_dot_tensor,
                                                    sw_tensor, dsw_tensor, res_vec_tensor, res_vec_tensor_1,
                                                    res_vec_tensor_2, res_vec_tensor_3, dis_revert_tensor,
                                                    res_descrpt_a_deriv_tensor,
                                                    [sorted_neighbour_coord_x,
                                                     sorted_neighbour_coord_y,
                                                     sorted_neighbour_coord_z],
                                                    rcut_ub, free_buffer, nnei_repeat_align)

                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        self.vector_compute_last_process(dis_revert_tensor,
                                                         [sorted_neighbour_coord_x,
                                                          sorted_neighbour_coord_y,
                                                          sorted_neighbour_coord_z],
                                                         sorted_dis_dot_tensor,
                                                         sw_tensor, dsw_tensor,
                                                         res_vec_tensor, res_vec_tensor_1,
                                                         res_vec_tensor_2, res_vec_tensor_3,
                                                         res_descrpt_a_tensor,
                                                         res_descrpt_a_deriv_tensor, free_buffer)

                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    self.do_avg_process(res_descrpt_a_tensor, res_descrpt_a_deriv_tensor, cur_loc_type_id)

            if self.nnei % 8 == 0:
                self.data_move_to_gm(res_descrpt_a_tensor,
                                     res_descrpt_a_deriv_tensor,
                                     cur_sample, cur_loc)
            else:
                with self.tik_instance.if_scope(loc_nei_idx != (one_cor_nloc_num - 1)):
                    self.data_move_to_gm(res_descrpt_a_tensor,
                                         res_descrpt_a_deriv_tensor,
                                         cur_sample, cur_loc)

                with self.tik_instance.if_scope(loc_nei_idx == (one_cor_nloc_num - 1)):
                    self.data_move_to_gm_last_loc(res_descrpt_a_tensor,
                                                  res_descrpt_a_deriv_tensor,
                                                  cur_sample, cur_loc)

    def get_dynamic_args(self):
        """
        Get the nloc value in natoms data.
        """
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            natoms_ub = self.tik_instance.Tensor(self.type_dtype, [self.block_num],
                                                 name="natoms_ub", scope=self.tik.scope_ubuf)

            self.tik_instance.data_move(natoms_ub, self.natoms_gm,
                                        sid=0, nburst=1, burst=1,
                                        src_stride=0, dst_stride=0)

            self.nloc_d.set_as(natoms_ub[0])
            self.total_nloc.set_as(natoms_ub[0])
            self.nall_d.set_as(natoms_ub[1])
            self.nall_d.set_as((self.nall_d + self.block_num - 1) // self.block_num * self.block_num)

    def compute_process(self):
        """
        Get core nums and nloc value, then dis the compute_process to every core.
        """
        self.get_dynamic_args()
        one_cor_nloc_num, last_core_num = self.get_core_nums()

        with self.tik_instance.for_range(0, self.nsample) as cur_sample:
            with self.tik_instance.for_range(0, self.cur_op_core_num, block_num=self.cur_op_core_num) as block_idx:
                with self.tik_instance.if_scope(block_idx < self.cur_op_core_num_scalar - 1):
                    self.compute_one_core(0, one_cor_nloc_num, block_idx, cur_sample, one_cor_nloc_num)
                with self.tik_instance.else_scope():
                    self.compute_one_core(0, last_core_num, block_idx, cur_sample, one_cor_nloc_num)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.distance_gm, self.rij_x_gm, self.rij_y_gm, self.rij_z_gm,
                                           self.type_gm, self.natoms_gm, self.mesh_gm, self.davg_gm, self.dstd_gm],
                                   outputs=[self.descrpt_gm, self.descrpt_deriv_gm],
                                   config={})


# 'pylint: disable=unused-argument
def _check_params(distance, rij_x, rij_y, rij_z, types, natoms, mesh, davg, dstd, descrpt, descrpt_deriv,
                  rcut_a, rcut_r, rcut_r_smth, sel_a, kernel_name):
    """
    Check the input args value.
    """
    para_check.check_dtype(distance.get("dtype").lower(), ("float32"), param_name="distance")

    para_check.check_dtype(rij_x.get("dtype").lower(), ("float32"), param_name="rij_x")

    para_check.check_dtype(rij_y.get("dtype").lower(), ("float32"), param_name="rij_y")

    para_check.check_dtype(rij_z.get("dtype").lower(), ("float32"), param_name="rij_z")

    para_check.check_dtype(types.get("dtype").lower(), ("int32"), param_name="type")

    para_check.check_dtype(natoms.get("dtype").lower(), ("int32"), param_name="natoms")

    para_check.check_dtype(mesh.get("dtype").lower(), ("int32"), param_name="mesh")

    para_check.check_dtype(davg.get("dtype").lower(), ("float32"), param_name="davg")

    para_check.check_dtype(dstd.get("dtype").lower(), ("float32"), param_name="dstd")

    para_check.check_dtype(descrpt.get("dtype").lower(), ("float32"), param_name="descrpt")

    para_check.check_dtype(descrpt_deriv.get("dtype").lower(), ("float32"), param_name="descrpt_deriv")

    distance_shape = distance.get("shape")
    para_check.check_shape(distance_shape, min_rank=2, max_rank=2, param_name="distance")

    rij_x_shape = rij_x.get("shape")
    para_check.check_shape(rij_x_shape, min_rank=2, max_rank=2, param_name="rij_x")

    rij_y_shape = rij_y.get("shape")
    para_check.check_shape(rij_y_shape, min_rank=2, max_rank=2, param_name="rij_y")

    rij_z_shape = rij_z.get("shape")
    para_check.check_shape(rij_z_shape, min_rank=2, max_rank=2, param_name="rij_z")

    type_shape = types.get("shape")
    para_check.check_shape(type_shape, min_rank=2, max_rank=2, param_name="type")

    natoms_shape = natoms.get("shape")
    para_check.check_shape(natoms_shape, min_rank=1, max_rank=1, min_size=3, param_name="natoms")

    davg_shape = davg.get("shape")
    para_check.check_shape(davg_shape, min_rank=2, max_rank=2, param_name="davg")

    dstd_shape = dstd.get("shape")
    para_check.check_shape(dstd_shape, min_rank=2, max_rank=2, param_name="dstd")

    nall = type_shape[1]
    max_nall_size = 32000
    if nall > max_nall_size:
        rule = "The nall value only support less than 32000."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "nall", nall)

    if any((rcut_r < 0, rcut_r_smth < 0)):
        rule = "The attributes {rcut_a, rcut_r, rcut_r_smth} can not be minus value or all 0."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "rcut_a", rcut_a)

    if rcut_r_smth > rcut_r:
        rule = "rcut_r_smth should be less than rcut_r."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "rcut_r", rcut_r)

    type_num = len(sel_a)
    if type_num == 0:
        rule = "sel_a list cant be empty."
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "type_num", type_num)


# 'pylint: disable=redefined-builtin
@register_operator("ProdEnvMatACalcDescrpt")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def prod_env_mat_a_calc_descrpt(distance, rij_x, rij_y, rij_z, type, natoms, mesh, davg, dstd, descrpt, descrpt_deriv,
                                rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r, kernel_name="prod_env_mat_a_calc_descrpt"):
    """
    algorithm: prod_env_mat_a_calc_descrpt

    Parameters
    ----------
    distance : dict. shape and dtype of input, only support float32.
    rij_x : dict. shape and dtype of input, only support float32.
    rij_y: dict. shape and dtype of input, only support float32.
    rij_z: dict. shape and dtype of input, only support float32.
    type : dict. shape and dtype of input, means the all neighbour types, only support int32
    natoms: dict. shape and dtype of input, contains the nloc value, only support int32.
    mesh: dict. shape and dtype of input, the input data contains the neighbour coords, only support int32.
    davg: dict. shape and dtype of input, only support float32.
    dstd: dict. shape and dtype of input, only support float32.
    descrpt: dict. shape and dtype of output, only support float32.
    descrpt_deriv: dict. shape and dtype of output, only support float32.
    kernel_name : str cce kernel name

    Returns
    -------
    None
    """
    _check_params(distance, rij_x, rij_y, rij_z, type, natoms, mesh, davg, dstd, descrpt, descrpt_deriv,
                  rcut_a, rcut_r, rcut_r_smth, sel_a, kernel_name)
    obj = ProdEnvMatACalcDescrpt(distance, rij_x, rij_y, rij_z, type, natoms, mesh, davg, dstd,
                                 sel_a, rcut_r, rcut_r_smth, kernel_name)
    obj.compute_process()
