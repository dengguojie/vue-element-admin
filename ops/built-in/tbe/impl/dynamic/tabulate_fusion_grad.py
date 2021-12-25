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

tabulate_fusion_grad
"""

from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl import constant_util as constant


# 'pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-arguments
# 'pylint: disable=too-many-instance-attributes,unused-argument,too-few-public-methods
class TabulateFusionGrad:
    """Function: use to calc tabulate fusion grad for all loc
    """

    def __init__(self, table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem,
                 split_count, split_index, kernel_name):
        """
        init TabulateFusionGrad.

        Parameters
        ----------
        table : dict. shape and dtype of input data table
        table_info : dict. shape and dtype of input data table_info
        em_x : dict. shape and dtype of input data em_x
        em : dict. shape and dtype of input data em
        dy : dict. shape and dtype of input data dy
        descriptor : dict. shape and dtype of input data descriptor
        dy_dem_x : dict. shape and dtype of input data dy_dem_x
        dy_dem : dict. shape and dtype of input data dy_dem
        kernel_name : str. cce kernel name

        Returns
        -------
        None
        """
        self.kernel_name = kernel_name
        self.tik_inst = tik.Tik(tik.Dprofile)

        self.op_dtype = "float32"

        self.split_count = split_count
        self.split_index = split_index

        self.nei_tile = 16
        self.size_tile = 64
        self.tile_size = self.nei_tile * self.size_tile
        self.em_tile_size = self.nei_tile * 4
        self.dy_tile_size = self.size_tile * 4
        self.table_tile_size = self.size_tile * 6

        self.loc = self.tik_inst.Scalar(dtype="int64", name="loc")
        self.nnei = self.tik_inst.Scalar(dtype="int64", name="nnei")
        self.size = self.tik_inst.Scalar(dtype="int64", name="last_layer_size")
        self.loc_offset = self.tik_inst.Scalar(dtype="int64", name="loc_offset")
        self.loc_split = self.tik_inst.Scalar(dtype="int64", name="loc_split")
        self.high_core_num = self.tik_inst.Scalar("int64", name="high_core_num")
        self.low_core_num = self.tik_inst.Scalar("int64", name="low_core_num")
        self.loc_per_high_core = self.tik_inst.Scalar("int64", name="loc_per_high_core")
        self.loc_per_low_core = self.tik_inst.Scalar("int64", name="loc_per_low_core")

        self.em_row_size = self.tik_inst.Scalar("int64", name="em_row_size")
        self.dy_row_size = self.tik_inst.Scalar("int64", name="dy_row_size")
        self.table_row_size = self.tik_inst.Scalar("int64", name="table_row_size")

        self.lower = self.tik_inst.Scalar(self.op_dtype, name="lower")
        self.upper = self.tik_inst.Scalar(self.op_dtype, name="upper")
        self._max = self.tik_inst.Scalar(self.op_dtype, name="_max")
        self.stride0 = self.tik_inst.Scalar(self.op_dtype, name="stride0")
        self.rec_stride0 = self.tik_inst.Scalar(self.op_dtype, name="rec_stride0")
        self.stride1 = self.tik_inst.Scalar(self.op_dtype, name="stride1")
        self.rec_stride1 = self.tik_inst.Scalar(self.op_dtype, name="rec_stride1")

        self.first_stride = self.tik_inst.Scalar(self.op_dtype, name="first_stride")
        self.tmp_scalar_int32 = self.tik_inst.Scalar("int32", name="tmp_scalar_int32")
        self.max_tbl_idx = self.tik_inst.Scalar(self.op_dtype, name="max_tbl_idx")

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.nei_max_size = 256  # max nei per loc
        self.layer_size_max_size = 256  # max nei per loc

        self.nei_tile_floor = self.tik_inst.Scalar("int64", name="nei_tile_floor")
        self.nei_tile_ceil = self.tik_inst.Scalar("int64", name="nei_tile_ceil")
        self.nei_batch = self.tik_inst.Scalar("int64", name="nei_batch")
        self.nei_repeats_align64 = self.tik_inst.Scalar("int64", name="nei_repeats_align64")
        self.nei_repeats = self.tik_inst.Scalar("int64", name="nei_repeats")
        self.size_tile_ceil = self.tik_inst.Scalar("int64", name="size_tile_ceil")
        self.size_align8 = self.tik_inst.Scalar("int64", name="size_align8")
        self.size_offset = self.tik_inst.Scalar("int64", name="size_offset")

        # init gm tensor
        self.tiling_gm = self.tik_inst.Tensor("int64", (constant.SIZE_SIXTEEN,), name="tiling_gm",
                                              scope=tik.scope_gm)
        self.table_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                             name="table_gm", scope=tik.scope_gm)
        self.table_info_gm = self.tik_inst.Tensor(self.op_dtype, (8,),
                                                  name="table_info_gm", scope=tik.scope_gm)
        self.em_x_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                            name="em_x_gm", scope=tik.scope_gm)
        self.em_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                          name="em_gm", scope=tik.scope_gm)
        self.dy_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                          name="dy_gm", scope=tik.scope_gm)
        self.descriptor_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                                  name="descriptor_gm", scope=tik.scope_gm)
        self.dy_dem_x_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                                name="dy_dem_x_gm", scope=tik.scope_gm)
        self.dy_dem_gm = self.tik_inst.Tensor(self.op_dtype, (constant.SHAPE_SIZE_LIMIT,),
                                              name="dy_dem_gm", scope=tik.scope_gm)

    def _init_scalar_var(self):
        tiling_ub = self.tik_inst.Tensor("int64", (constant.SIZE_SIXTEEN,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(tiling_ub, self.tiling_gm, 0, 1,
                                constant.SIZE_SIXTEEN * constant.DATA_SIZE_EIGHT // constant.BLOCK_SIZE, 0, 0)

        self.loc.set_as(tiling_ub[0])
        self.nnei.set_as(tiling_ub[1])
        self.size.set_as(tiling_ub[2])
        self.loc_offset.set_as(tiling_ub[3])
        self.loc_split.set_as(tiling_ub[4])
        self.high_core_num.set_as(tiling_ub[5])
        self.low_core_num.set_as(tiling_ub[6])
        self.loc_per_high_core.set_as(tiling_ub[7])
        self.loc_per_low_core.set_as(tiling_ub[8])

        self.em_row_size.set_as(self.nnei * 4)
        self.dy_row_size.set_as(self.size * 4)
        self.table_row_size.set_as(self.size * 6)

        table_info_ub = self.tik_inst.Tensor(self.op_dtype, (8,), name="table_info_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(table_info_ub, self.table_info_gm, 0, 1, 1, 0, 0)

        self.lower.set_as(table_info_ub[0])
        self.upper.set_as(table_info_ub[1])
        self._max.set_as(table_info_ub[2])
        self.stride0.set_as(table_info_ub[3])
        self.rec_stride0.set_as(1 / self.stride0)
        self.stride1.set_as(table_info_ub[4])
        self.rec_stride1.set_as(1 / self.stride1)

        self.first_stride.set_as((self.upper - self.lower) * self.rec_stride0)
        self.tik_inst.scalar_conv('floor', self.tmp_scalar_int32, self.first_stride)
        self.tik_inst.scalar_conv('none', self.first_stride, self.tmp_scalar_int32)

        self.max_tbl_idx.set_as((self._max - self.upper) * self.rec_stride1)
        self.tik_inst.scalar_conv('floor', self.tmp_scalar_int32, self.max_tbl_idx)
        self.tik_inst.scalar_conv('none', self.max_tbl_idx, self.tmp_scalar_int32)
        self.max_tbl_idx.set_as(self.first_stride + self.max_tbl_idx - 1)

        self.nei_tile_floor.set_as(self.nnei // self.nei_tile * self.nei_tile)
        self.nei_tile_ceil.set_as((self.nnei + self.nei_tile - 1) // self.nei_tile * self.nei_tile)
        self.nei_batch.set_as((self.nnei + self.nei_tile - 1) // self.nei_tile)
        self.nei_repeats_align64.set_as((self.nnei + constant.MASK64 - 1) // constant.MASK64)
        self.nei_repeats.set_as(self.nei_tile_ceil * 4 // self.nei_tile)
        self.size_tile_ceil.set_as((self.size + self.size_tile - 1) // self.size_tile * self.size_tile)
        self.size_align8.set_as((self.size + constant.REPEAT_STRIDE_EIGHT - 1) // constant.REPEAT_STRIDE_EIGHT)
        self.size_offset.set_as(self.size // self.size_tile)

    def _load_em_x_one_loc(self, loc):
        em_x_one_loc = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size,), name="em_x_one_loc",
                                            scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(constant.MASK64, em_x_one_loc, 0, self.nei_repeats_align64, 8)
        table_offset_one_loc = self.tik_inst.Tensor("int32", (self.nei_max_size,), name="table_offset_one_loc",
                                                    scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(constant.MASK64, table_offset_one_loc, 0, self.nei_repeats_align64, 8)
        tmp_cmp_mask = self.tik_inst.Tensor("uint64", (self.nei_max_size // constant.MASK64,),
                                            name="tmp_cmp_mask", scope=tik.scope_ubuf)

        em_x = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size,), name="em_x", scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(constant.MASK64, em_x, -1, self.nei_repeats_align64, 8)

        self.tik_inst.data_move(em_x, self.em_x_gm[loc * self.nnei], 0, 1,
                                self.nei_tile_floor // constant.REPEAT_STRIDE_EIGHT, 0, 0)

        with self.tik_inst.if_scope(self.nnei % self.nei_tile != 0):
            self.tik_inst.data_move(em_x[self.nei_tile_floor],
                                    self.em_x_gm[loc * self.nnei + self.nnei - self.nei_tile],
                                    0, 1, self.nei_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)

        em_x_tmp = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size,), name="em_x_tmp", scope=tik.scope_ubuf)
        table_offset_fp32 = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size,), name="table_offset_fp32",
                                                 scope=tik.scope_ubuf)
        table_idx_ub = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size,), name="table_idx_ub",
                                            scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(constant.MASK64, table_idx_ub, 0, self.nei_repeats_align64, 8)

        # condition 1: x >= lower
        # table_offset = (x - lower) // s0
        self.tik_inst.vadds(constant.MASK64, em_x_tmp, em_x, -self.lower,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vmuls(constant.MASK64, em_x_tmp, em_x_tmp, self.rec_stride0,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vconv(constant.MASK64, "floor", table_offset_one_loc, em_x_tmp,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vconv(constant.MASK64, "none", table_offset_fp32, table_offset_one_loc,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        # x -= (table_offset * s0) + lower
        self.tik_inst.vmuls(constant.MASK64, em_x_tmp, table_offset_fp32, self.stride0,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vadds(constant.MASK64, em_x_tmp, em_x_tmp, self.lower,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vsub(constant.MASK64, em_x_tmp, em_x, em_x_tmp,
                           self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)
        # mask and selection: x >= lower
        self.tik_inst.vcmpvs_ge(tmp_cmp_mask, em_x, self.lower, self.nei_repeats_align64, 1, 8)
        self.tik_inst.vsel(constant.MASK64, 2, table_idx_ub, tmp_cmp_mask, table_offset_fp32, table_idx_ub,
                           self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsel(constant.MASK64, 2, em_x_one_loc, tmp_cmp_mask, em_x_tmp, em_x_one_loc,
                           self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)

        # condition 2: x >= upper
        # table_offset = (x - upper) // s1
        self.tik_inst.vadds(constant.MASK64, em_x_tmp, em_x, -self.upper, self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vmuls(constant.MASK64, em_x_tmp, em_x_tmp, self.rec_stride1,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vconv(constant.MASK64, "floor", table_offset_one_loc, em_x_tmp,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vconv(constant.MASK64, "none", table_offset_fp32, table_offset_one_loc,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        # x -= (table_offset * s1) + upper
        self.tik_inst.vmuls(constant.MASK64, em_x_tmp, table_offset_fp32, self.stride1,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vadds(constant.MASK64, em_x_tmp, em_x_tmp, self.upper, self.nei_repeats_align64, 1, 1, 8, 8)
        self.tik_inst.vsub(constant.MASK64, em_x_tmp, em_x, em_x_tmp, self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)
        # table_offset = table_offset + first_stride
        self.tik_inst.vadds(constant.MASK64, table_offset_fp32, table_offset_fp32, self.first_stride,
                            self.nei_repeats_align64, 1, 1, 8, 8)
        # mask and selection: x >= upper
        self.tik_inst.vcmpvs_ge(tmp_cmp_mask, em_x, self.upper, self.nei_repeats_align64, 1, 8)
        self.tik_inst.vsel(constant.MASK64, 2, table_idx_ub, tmp_cmp_mask, table_offset_fp32, table_idx_ub,
                           self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsel(constant.MASK64, 2, em_x_one_loc, tmp_cmp_mask, em_x_tmp, em_x_one_loc,
                           self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)

        # condition 3: x >= max
        # table_offset = max_tbl_idx
        self.tik_inst.vec_dup(constant.MASK64, table_offset_fp32, self.max_tbl_idx, self.nei_repeats_align64, 8)
        # x = 0
        self.tik_inst.vec_dup(constant.MASK64, em_x_tmp, 0, self.nei_repeats_align64, 8)
        # mask and selection: x >= max
        self.tik_inst.vcmpvs_ge(tmp_cmp_mask, em_x, self._max, self.nei_repeats_align64, 1, 8)
        self.tik_inst.vsel(constant.MASK64, 2, table_idx_ub, tmp_cmp_mask, table_offset_fp32, table_idx_ub,
                           self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsel(constant.MASK64, 2, em_x_one_loc, tmp_cmp_mask, em_x_tmp, em_x_one_loc,
                           self.nei_repeats_align64, 1, 1, 1, 8, 8, 8)

        self.tik_inst.vconv(constant.MASK64, "floor", table_offset_one_loc, table_idx_ub,
                            self.nei_repeats_align64, 1, 1, 8, 8)

        return em_x_one_loc, table_offset_one_loc

    def _load_em_one_loc(self, loc):
        em_one_loc = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size * 4,), name="em_one_loc",
                                          scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(self.nei_tile, em_one_loc, 0, self.nei_repeats, self.nei_tile // 8)

        self.tik_inst.data_move(em_one_loc, self.em_gm[loc * self.em_row_size],
                                0, 1, (self.nei_tile_floor * 4) // constant.REPEAT_STRIDE_EIGHT, 0, 0)
        with self.tik_inst.if_scope(self.nnei % self.nei_tile != 0):
            self.tik_inst.data_move(em_one_loc[self.nei_tile_floor * 4],
                                    self.em_gm[loc * self.em_row_size + self.em_row_size - self.em_tile_size],
                                    0, 1, self.em_tile_size // constant.REPEAT_STRIDE_EIGHT, 0, 0)

        return em_one_loc

    def _load_dy_one_loc(self, loc):
        repeats = (self.layer_size_max_size * 4) // self.size_tile
        dy_one_loc = self.tik_inst.Tensor(self.op_dtype, (self.layer_size_max_size * 4,), name="dy_one_loc",
                                          scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(self.size_tile, dy_one_loc, 0, repeats, self.size_tile // 8)

        self.tik_inst.data_move(dy_one_loc, self.dy_gm[loc * self.dy_row_size],
                                0, 1, self.size_align8, 0, 0)
        self.tik_inst.data_move(dy_one_loc[self.size_tile_ceil], self.dy_gm[loc * self.dy_row_size + self.size],
                                0, 1, self.size_align8, 0, 0)
        self.tik_inst.data_move(dy_one_loc[self.size_tile_ceil * 2], self.dy_gm[loc * self.dy_row_size + self.size * 2],
                                0, 1, self.size_align8, 0, 0)
        self.tik_inst.data_move(dy_one_loc[self.size_tile_ceil * 3], self.dy_gm[loc * self.dy_row_size + self.size * 3],
                                0, 1, self.size_align8, 0, 0)

        return dy_one_loc

    def _process_layer_size(self, input_dict, nei_offset):
        vem_x_tile = input_dict["vem_x_tile"]
        vem_dot_tile = input_dict["vem_dot_tile"]
        vdy_dem = input_dict["vdy_dem"]
        dy_one_loc = input_dict["dy_one_loc"]
        table_offset_one_loc = input_dict["table_offset_one_loc"]

        vdy_out = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * 5,), name="vdy_out", scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(self.nei_tile, vdy_out, 0, 5, self.nei_tile // 8)

        vdy_dot_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 4,), name="vdy_dot_tile",
                                            scope=tik.scope_ubuf)
        vrr_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 4,), name="vrr_tile",
                                        scope=tik.scope_ubuf)
        va_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 6,), name="va_tile",
                                       scope=tik.scope_ubuf)

        with self.tik_inst.for_range(0, self.size // self.size_tile) as size_offset:
            # vrr ready : dy -> vrr
            with self.tik_inst.new_stmt_scope(disable_sync=False):
                self.tik_inst.vec_dup(constant.MASK64, vrr_tile, 0, (self.tile_size * 4) // constant.MASK64, 8)
                # vrr0 : dy0 broadcast
                self.tik_inst.vadd(self.size_tile, vrr_tile, vrr_tile, dy_one_loc[size_offset * self.size_tile],
                                   self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                                   self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
                # vrr1 : dy1 broadcast
                self.tik_inst.vadd(self.size_tile, vrr_tile[self.tile_size], vrr_tile[self.tile_size],
                                   dy_one_loc[self.size_tile_ceil + size_offset * self.size_tile],
                                   self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                                   self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
                # vrr2 : dy2 broadcast
                self.tik_inst.vadd(self.size_tile, vrr_tile[self.tile_size * 2], vrr_tile[self.tile_size * 2],
                                   dy_one_loc[self.size_tile_ceil * 2 + size_offset * self.size_tile],
                                   self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                                   self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
                # vrr3 : dy3 broadcast
                self.tik_inst.vadd(self.size_tile, vrr_tile[self.tile_size * 3], vrr_tile[self.tile_size * 3],
                                   dy_one_loc[self.size_tile_ceil * 3 + size_offset * self.size_tile],
                                   self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                                   self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
                self.tik_inst.v4dtrans(True, vdy_dot_tile, vrr_tile, self.tile_size, 4)

            # va0/va1/va2/va3/va4/va5 ready : table -> va
            with self.tik_inst.new_stmt_scope(disable_sync=False):
                table = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 6,), name="table", scope=tik.scope_ubuf)
                with self.tik_inst.for_range(0, self.nei_tile) as nei_idx:
                    offset = nei_offset * self.nei_tile + nei_idx
                    table_offset = self.tik_inst.Scalar(dtype="int32", init_value=table_offset_one_loc[offset])
                    offset = table_offset * self.table_row_size + size_offset * self.table_tile_size
                    self.tik_inst.data_move(table[nei_idx * self.table_tile_size], self.table_gm[offset],
                                            0, 1, self.table_tile_size // constant.REPEAT_STRIDE_EIGHT, 0, 0)
                self.tik_inst.v4dtrans(False, va_tile, table, self.tile_size, 6)

            # all data required ready, let's go!
            # res = a5 * x5 + a4 * x4 + a3 * x3 + a2 * x2 + a1 * x + a0
            # grad = res' = 5 * a5 * x4 + 4 * a4 * x3 + 3 * a3 * x2 + 2 * a2 * x + a1
            with self.tik_inst.new_stmt_scope(disable_sync=False):
                # res = a0
                vres = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vres", scope=tik.scope_ubuf)
                self.tik_inst.vec_dup(constant.MASK64, vres, 0, self.tile_size // constant.MASK64, 8)
                self.tik_inst.vadd(constant.MASK64, vres, vres, va_tile,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # grad = a1
                vgrad = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vgrad", scope=tik.scope_ubuf)
                self.tik_inst.vec_dup(constant.MASK64, vgrad, 0, self.tile_size // constant.MASK64, 8)
                self.tik_inst.vadd(constant.MASK64, vgrad, vgrad, va_tile[self.tile_size],
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # vpx = x
                vpx = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vpx", scope=tik.scope_ubuf)
                self.tik_inst.vec_dup(constant.MASK64, vpx, 0, self.tile_size // constant.MASK64, 8)
                self.tik_inst.vadd(constant.MASK64, vpx, vpx, vem_x_tile,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # res = res + a1 * x
                self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size], vpx,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # vxa = 2 * a2
                vxa = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vxa", scope=tik.scope_ubuf)
                self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 2], 2,
                                    self.tile_size // constant.MASK64, 1, 1, 8, 8)
                # grad = grad + 2 * a2 * x
                self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # vpx = x2
                self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # res = res + a2 * x2
                self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 2], vpx,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # vxa = 3 * a3
                self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 3], 3,
                                    self.tile_size // constant.MASK64, 1, 1, 8, 8)
                # grad = grad + 3 * a3 * x2
                self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # vpx = x3
                self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # res = res + a3 * x3
                self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 3], vpx,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # vxa = 4 * a4
                self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 4], 4,
                                    self.tile_size // constant.MASK64, 1, 1, 8, 8)
                # grad = grad + 4 * a4 * x3
                self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # vpx = x4
                self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # res = res + a4 * x4
                self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 4], vpx,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # vxa = 5 * a5
                self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 5], 5,
                                    self.tile_size // constant.MASK64, 1, 1, 8, 8)
                # grad = grad + 5 * a5 * x4
                self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)

                # vpx = x5
                self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # res = res + a5 * x5
                self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 5], vpx,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # dy_dem_0 = res * rr0
                self.tik_inst.vmla(constant.MASK64, vdy_dem, vres, vrr_tile, self.tile_size // constant.MASK64,
                                   1, 1, 1, 8, 8, 8)
                # dy_dem_1 = res * rr1
                self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size], vres, vrr_tile[self.tile_size],
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # dy_dem_2 = res * rr2
                self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size * 2], vres, vrr_tile[self.tile_size * 2],
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # dy_dem_3 = res * rr3
                self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size * 3], vres, vrr_tile[self.tile_size * 3],
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
                # dy_dot : dot(ll, rr)
                self.tik_inst.vmul(constant.MASK64, vdy_dot_tile, vem_dot_tile, vdy_dot_tile,
                                   (self.tile_size * 4) // constant.MASK64, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vcpadd(constant.MASK64, vdy_dot_tile, vdy_dot_tile,
                                     (self.tile_size * 4) // constant.MASK64, 1, 1, 8)
                self.tik_inst.vcpadd(constant.MASK64, vdy_dot_tile, vdy_dot_tile,
                                     (self.tile_size * 2) // constant.MASK64, 1, 1, 8)
                # dy_dem_4(grad) = grad * dy_dot
                self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size * 4], vgrad, vdy_dot_tile,
                                   self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)

        # sum of all value in same size_tile
        self.tik_inst.vcadd(self.size_tile, vdy_out, vdy_dem, (self.tile_size * 5) // self.size_tile,
                            1, 1, self.size_tile // 8)

        return vdy_out

    def _process_layer_size_tail(self, input_dict, nei_offset):
        vem_x_tile = input_dict["vem_x_tile"]
        vem_dot_tile = input_dict["vem_dot_tile"]
        vdy_dem = input_dict["vdy_dem"]
        dy_one_loc = input_dict["dy_one_loc"]
        table_offset_one_loc = input_dict["table_offset_one_loc"]

        vdy_out = self.tik_inst.Tensor(self.op_dtype, (self.nei_tile * 5,), name="vdy_out", scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(self.nei_tile, vdy_out, 0, 5, self.nei_tile // 8)

        vdy_dot_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 4,), name="vdy_dot_tile",
                                            scope=tik.scope_ubuf)
        vrr_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 4,), name="vrr_tile",
                                        scope=tik.scope_ubuf)
        va_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 6,), name="va_tile",
                                       scope=tik.scope_ubuf)

        # vrr ready : dy -> vrr
        with self.tik_inst.new_stmt_scope(disable_sync=False):
            self.tik_inst.vec_dup(constant.MASK64, vrr_tile, 0, (self.tile_size * 4) // constant.MASK64, 8)
            # vrr0 : dy0 broadcast
            self.tik_inst.vadd(self.size_tile, vrr_tile, vrr_tile, dy_one_loc[self.size_offset * self.size_tile],
                               self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                               self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
            # vrr1 : dy1 broadcast
            self.tik_inst.vadd(self.size_tile, vrr_tile[self.tile_size], vrr_tile[self.tile_size],
                               dy_one_loc[self.size_tile_ceil + self.size_offset * self.size_tile],
                               self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                               self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
            # vrr2 : dy2 broadcast
            self.tik_inst.vadd(self.size_tile, vrr_tile[self.tile_size * 2], vrr_tile[self.tile_size * 2],
                               dy_one_loc[self.size_tile_ceil * 2 + self.size_offset * self.size_tile],
                               self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                               self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
            # vrr3 : dy3 broadcast
            self.tik_inst.vadd(self.size_tile, vrr_tile[self.tile_size * 3], vrr_tile[self.tile_size * 3],
                               dy_one_loc[self.size_tile_ceil * 3 + self.size_offset * self.size_tile],
                               self.nei_tile, 1, 1, 1, self.size_tile // constant.REPEAT_STRIDE_EIGHT,
                               self.size_tile // constant.REPEAT_STRIDE_EIGHT, 0)
            self.tik_inst.v4dtrans(True, vdy_dot_tile, vrr_tile, self.tile_size, 4)

        # va0/va1/va2/va3/va4/va5 ready : table -> va
        with self.tik_inst.new_stmt_scope(disable_sync=False):
            table = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 6,), name="table", scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, self.nei_tile) as nei_idx:
                offset = nei_offset * self.nei_tile + nei_idx
                table_offset = self.tik_inst.Scalar(dtype="int32", init_value=table_offset_one_loc[offset])
                offset = table_offset * self.table_row_size + self.size_offset * self.table_tile_size
                self.tik_inst.data_move(table[nei_idx * self.table_tile_size], self.table_gm[offset],
                                        0, 1, self.table_tile_size // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.v4dtrans(False, va_tile, table, self.tile_size, 6)

        # all data required ready, let's go!
        # res = a5 * x5 + a4 * x4 + a3 * x3 + a2 * x2 + a1 * x + a0
        # grad = res' = 5 * a5 * x4 + 4 * a4 * x3 + 3 * a3 * x2 + 2 * a2 * x + a1
        with self.tik_inst.new_stmt_scope(disable_sync=False):
            # res = a0
            vres = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vres", scope=tik.scope_ubuf)
            self.tik_inst.vec_dup(constant.MASK64, vres, 0, self.tile_size // constant.MASK64, 8)
            self.tik_inst.vadd(constant.MASK64, vres, vres, va_tile,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # grad = a1
            vgrad = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vgrad", scope=tik.scope_ubuf)
            self.tik_inst.vec_dup(constant.MASK64, vgrad, 0, self.tile_size // constant.MASK64, 8)
            self.tik_inst.vadd(constant.MASK64, vgrad, vgrad, va_tile[self.tile_size],
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # vpx = x
            vpx = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vpx", scope=tik.scope_ubuf)
            self.tik_inst.vec_dup(constant.MASK64, vpx, 0, self.tile_size // constant.MASK64, 8)
            self.tik_inst.vadd(constant.MASK64, vpx, vpx, vem_x_tile,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # res = res + a1 * x
            self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size], vpx,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # vxa = 2 * a2
            vxa = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vxa", scope=tik.scope_ubuf)
            self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 2], 2,
                                self.tile_size // constant.MASK64, 1, 1, 8, 8)
            # grad = grad + 2 * a2 * x
            self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # vpx = x2
            self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # res = res + a2 * x2
            self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 2], vpx,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # vxa = 3 * a3
            self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 3], 3,
                                self.tile_size // constant.MASK64, 1, 1, 8, 8)
            # grad = grad + 3 * a3 * x2
            self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # vpx = x3
            self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # res = res + a3 * x3
            self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 3], vpx,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # vxa = 4 * a4
            self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 4], 4,
                                self.tile_size // constant.MASK64, 1, 1, 8, 8)
            # grad = grad + 4 * a4 * x3
            self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # vpx = x4
            self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # res = res + a4 * x4
            self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 4], vpx,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # vxa = 5 * a5
            self.tik_inst.vmuls(constant.MASK64, vxa, va_tile[self.tile_size * 5], 5,
                                self.tile_size // constant.MASK64, 1, 1, 8, 8)
            # grad = grad + 5 * a5 * x4
            self.tik_inst.vmla(constant.MASK64, vgrad, vxa, vpx, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)

            # vpx = x5
            self.tik_inst.vmul(constant.MASK64, vpx, vpx, vem_x_tile, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # res = res + a5 * x5
            self.tik_inst.vmla(constant.MASK64, vres, va_tile[self.tile_size * 5], vpx,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # dy_dem_0 = res * rr0
            self.tik_inst.vmla(constant.MASK64, vdy_dem, vres, vrr_tile, self.tile_size // constant.MASK64,
                               1, 1, 1, 8, 8, 8)
            # dy_dem_1 = res * rr1
            self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size], vres, vrr_tile[self.tile_size],
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # dy_dem_2 = res * rr2
            self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size * 2], vres, vrr_tile[self.tile_size * 2],
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # dy_dem_3 = res * rr3
            self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size * 3], vres, vrr_tile[self.tile_size * 3],
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)
            # dy_dot : dot(ll, rr)
            self.tik_inst.vmul(constant.MASK64, vdy_dot_tile, vem_dot_tile, vdy_dot_tile,
                               (self.tile_size * 4) // constant.MASK64, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vcpadd(constant.MASK64, vdy_dot_tile, vdy_dot_tile,
                                 (self.tile_size * 4) // constant.MASK64, 1, 1, 8)
            self.tik_inst.vcpadd(constant.MASK64, vdy_dot_tile, vdy_dot_tile,
                                 (self.tile_size * 2) // constant.MASK64, 1, 1, 8)
            # dy_dem_4(grad) = grad * dy_dot
            self.tik_inst.vmla(constant.MASK64, vdy_dem[self.tile_size * 4], vgrad, vdy_dot_tile,
                               self.tile_size // constant.MASK64, 1, 1, 1, 8, 8, 8)

        # just sum of tail of each last_layer_size by mask setting
        self.tik_inst.vcadd(self.size - self.size_offset * self.size_tile, vdy_out, vdy_dem,
                            (self.tile_size * 5) // self.size_tile, 1, 1, self.size_tile // 8)

        return vdy_out

    def _process_nei(self, loc, input_dict):
        em_x_one_loc = input_dict["em_x_one_loc"]
        table_offset_one_loc = input_dict["table_offset_one_loc"]
        em_one_loc = input_dict["em_one_loc"]
        dy_one_loc = input_dict["dy_one_loc"]

        dy_dem_x_ub = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size,), name="dy_dem_x_ub",
                                           scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(self.nei_tile, dy_dem_x_ub, 0, self.nei_max_size // self.nei_tile, self.nei_tile // 8)

        dy_dem_ub = self.tik_inst.Tensor(self.op_dtype, (self.nei_max_size * 4,), name="dy_dem_ub",
                                         scope=tik.scope_ubuf)
        self.tik_inst.vec_dup(constant.MASK64, dy_dem_ub, 0, (self.nei_max_size * 4) // constant.MASK64, 8)

        vem_x_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="vem_x_tile", scope=tik.scope_ubuf)
        vem_dot_tile = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 4,), name="vem_dot_tile",
                                            scope=tik.scope_ubuf)
        vdy_dem = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 5,), name="vdy_dem", scope=tik.scope_ubuf)

        with self.tik_inst.for_range(0, self.nei_batch) as nei_offset:
            # em_x_tile ready : broadcast x by nei_tile
            with self.tik_inst.new_stmt_scope(disable_sync=False):
                em_x_tmp = self.tik_inst.Tensor(self.op_dtype, (self.tile_size,), name="em_x_tmp", scope=tik.scope_ubuf)
                self.tik_inst.vec_dup(constant.MASK64, em_x_tmp, 0, self.tile_size // constant.MASK64, 8)
                self.tik_inst.vadd(self.nei_tile, em_x_tmp, em_x_tmp, em_x_one_loc[nei_offset * self.nei_tile],
                                   self.size_tile, 1, 1, 1, self.nei_tile // constant.REPEAT_STRIDE_EIGHT,
                                   self.nei_tile // constant.REPEAT_STRIDE_EIGHT, 0)
                self.tik_inst.v4dtrans(False, vem_x_tile, em_x_tmp, self.size_tile, self.nei_tile)

            # em_dot_tile ready : em -> em_dot_tile by op sequence trans -> broadcast -> trans
            with self.tik_inst.new_stmt_scope(disable_sync=False):
                em_bc = self.tik_inst.Tensor(self.op_dtype, (self.tile_size * 4,), name="em_bc", scope=tik.scope_ubuf)
                self.tik_inst.vec_dup(constant.MASK64, em_bc, 0, (self.tile_size * 4) // constant.MASK64, 8)
                em_tmp = self.tik_inst.Tensor(self.op_dtype, (self.em_tile_size,), name="em_tmp", scope=tik.scope_ubuf)
                self.tik_inst.v4dtrans(False, em_tmp, em_one_loc[nei_offset * self.em_tile_size], self.nei_tile, 4)
                # care!! Resolve the constrains that just copy max 64 in one repeating
                # constrains!! nei_tile must be 16 or 32
                self.tik_inst.vadd(constant.MASK64, em_bc, em_bc, em_tmp,
                                   self.size_tile, 1, 1, 1, self.em_tile_size // constant.REPEAT_STRIDE_EIGHT,
                                   self.em_tile_size // constant.REPEAT_STRIDE_EIGHT, 0)
                self.tik_inst.v4dtrans(False, vem_dot_tile, em_bc, self.size_tile * 4, self.nei_tile)

            self.tik_inst.vec_dup(constant.MASK64, vdy_dem, 0, (self.tile_size * 5) // constant.MASK64, 8)
            layer_size_input = {"vem_x_tile": vem_x_tile,
                                "vem_dot_tile": vem_dot_tile,
                                "vdy_dem": vdy_dem,
                                "dy_one_loc": dy_one_loc,
                                "table_offset_one_loc": table_offset_one_loc
                                }
            # calc part1 of last_layer_size data ( sum of tile by tile )
            vdy_out_part1 = self._process_layer_size(layer_size_input, nei_offset)

            # calc part2 of last_layer_size data ( tail )
            with self.tik_inst.if_scope(self.size % self.size_tile != 0):
                self.tik_inst.vec_dup(constant.MASK64, vdy_dem, 0, (self.tile_size * 5) // constant.MASK64, 8)
                vdy_out_part2 = self._process_layer_size_tail(layer_size_input, nei_offset)
                self.tik_inst.vadd(self.nei_tile, vdy_out_part1, vdy_out_part1, vdy_out_part2,
                                   5, 1, 1, 1, self.nei_tile // 8, self.nei_tile // 8, self.nei_tile // 8)

            # output dy_dem0/dy_dem1/dy_dem2/dy_dem3
            self.tik_inst.v4dtrans(True, dy_dem_ub[nei_offset * self.em_tile_size], vdy_out_part1, self.nei_tile, 4)
            # output dy_dem_x
            self.tik_inst.vadd(self.nei_tile, dy_dem_x_ub[nei_offset * self.nei_tile],
                               dy_dem_x_ub[nei_offset * self.nei_tile], vdy_out_part1[self.em_tile_size],
                               1, 1, 1, 1, 8, 8, 8)

        last_dem_x = self.tik_inst.Scalar(self.op_dtype, init_value=dy_dem_x_ub[self.nei_tile_ceil - 1])
        last_dem_4 = self.tik_inst.Scalar(self.op_dtype,
                                          init_value=dy_dem_ub[self.nei_tile_ceil * 4 - 1])
        last_dem_3 = self.tik_inst.Scalar(self.op_dtype,
                                          init_value=dy_dem_ub[self.nei_tile_ceil * 4 - 2])
        last_dem_2 = self.tik_inst.Scalar(self.op_dtype,
                                          init_value=dy_dem_ub[self.nei_tile_ceil * 4 - 3])
        last_dem_1 = self.tik_inst.Scalar(self.op_dtype,
                                          init_value=dy_dem_ub[self.nei_tile_ceil * 4 - 4])

        for_loop = self.tik_inst.Scalar("int32", init_value=self.nei_tile_ceil)
        with self.tik_inst.for_range(1, for_loop) as nei_idx:
            last_dem_x_pre = self.tik_inst.Scalar(self.op_dtype,
                                                  init_value=dy_dem_x_ub[self.nei_tile_ceil - 1 - nei_idx])
            count = self.tik_inst.Scalar("int32", init_value=nei_idx)
            with self.tik_inst.if_scope(last_dem_x_pre == last_dem_x):
                dy_dem_x_ub[self.nei_tile_ceil - nei_idx].set_as(0)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4 + 3].set_as(0)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4 + 2].set_as(0)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4 + 1].set_as(0)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4].set_as(0)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(tik.all(self.nnei % self.nei_tile != 0, nei_idx >= self.nei_tile)):
                    count.set_as(nei_idx - self.nei_tile + self.nnei % self.nei_tile)
                dy_dem_x_ub[self.nei_tile_ceil - nei_idx].set_as(last_dem_x * count)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4 + 3].set_as(last_dem_4 * count)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4 + 2].set_as(last_dem_3 * count)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4 + 1].set_as(last_dem_2 * count)
                dy_dem_ub[(self.nei_tile_ceil - nei_idx) * 4].set_as(last_dem_1 * count)
                for_loop.set_as(1)

        self.tik_inst.data_move(self.dy_dem_gm[(loc - self.loc_offset) * self.em_row_size], dy_dem_ub, 0, 1,
                                (self.nei_tile_floor * 4) // constant.REPEAT_STRIDE_EIGHT, 0, 0)
        self.tik_inst.data_move(self.dy_dem_x_gm[(loc - self.loc_offset) * self.nnei], dy_dem_x_ub, 0, 1,
                                self.nei_tile_floor // constant.REPEAT_STRIDE_EIGHT, 0, 0)

        with self.tik_inst.if_scope(self.nnei % self.nei_tile != 0):
            dy_dem_x_ub[self.nei_tile_floor].set_as(dy_dem_x_ub[self.nei_tile_floor - self.nei_tile_ceil + self.nnei])

            dy_dem_ub[self.nei_tile_floor * 4].set_as(dy_dem_ub[(self.nei_tile_floor -
                                                                 self.nei_tile_ceil + self.nnei) * 4])
            dy_dem_ub[self.nei_tile_floor * 4 + 1].set_as(dy_dem_ub[(self.nei_tile_floor -
                                                                     self.nei_tile_ceil + self.nnei) * 4 + 1])
            dy_dem_ub[self.nei_tile_floor * 4 + 2].set_as(dy_dem_ub[(self.nei_tile_floor -
                                                                     self.nei_tile_ceil + self.nnei) * 4 + 2])
            dy_dem_ub[self.nei_tile_floor * 4 + 3].set_as(dy_dem_ub[(self.nei_tile_floor -
                                                                     self.nei_tile_ceil + self.nnei) * 4 + 3])

            self.tik_inst.data_move(self.dy_dem_gm[(loc - self.loc_offset + 1) * self.em_row_size - self.em_tile_size],
                                    dy_dem_ub[self.nei_tile_floor * 4], 0, 1,
                                    self.em_tile_size // constant.REPEAT_STRIDE_EIGHT, 0, 0)
            self.tik_inst.data_move(self.dy_dem_x_gm[(loc - self.loc_offset + 1) * self.nnei - self.nei_tile],
                                    dy_dem_x_ub[self.nei_tile_floor], 0, 1,
                                    self.nei_tile // constant.REPEAT_STRIDE_EIGHT, 0, 0)

    def _compute_loc_grad(self, loc):
        """
        compute grad loc by loc
        """
        em_x_one_loc, table_offset_one_loc = self._load_em_x_one_loc(loc)

        dy_one_loc = self._load_dy_one_loc(loc)

        em_one_loc = self._load_em_one_loc(loc)

        data_load = {"em_x_one_loc": em_x_one_loc,
                     "table_offset_one_loc": table_offset_one_loc,
                     "em_one_loc": em_one_loc,
                     "dy_one_loc": dy_one_loc
                     }
        self._process_nei(loc, data_load)

    def compute(self):
        """
        compute
        """
        loc_start = self.tik_inst.Scalar(init_value=0, dtype="int32")
        loc_end = self.tik_inst.Scalar(init_value=0, dtype="int32")

        with self.tik_inst.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_i:
            self._init_scalar_var()

            with self.tik_inst.if_scope(core_i < self.high_core_num):
                loc_start.set_as(self.loc_offset + core_i * self.loc_per_high_core)
                loc_end.set_as(loc_start + self.loc_per_high_core)
            with self.tik_inst.elif_scope(core_i < self.loc):
                loc_start.set_as(self.loc_offset + self.high_core_num + core_i * self.loc_per_low_core)
                loc_end.set_as(loc_start + self.loc_per_low_core)
            with self.tik_inst.else_scope():
                loc_start.set_as(0)
                loc_end.set_as(0)

            with self.tik_inst.for_range(loc_start, loc_end, name="loc") as loc_i:
                self._compute_loc_grad(loc_i)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.ai_core_num,
                                                            "split_count": self.split_count,
                                                            "split_index": self.split_index})

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.table_gm, self.table_info_gm, self.em_x_gm, self.em_gm,
                                       self.dy_gm, self.descriptor_gm],
                               outputs=[self.dy_dem_x_gm, self.dy_dem_gm],
                               flowtable=(self.tiling_gm,),
                               config=opt_config)

        return self.tik_inst


def _check_params(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem, split_count,
                  split_index, kernel_name):
    # input dtype check
    para_check.check_dtype(table.get("dtype").lower(), ("float32",), param_name="table")
    para_check.check_dtype(table_info.get("dtype").lower(), ("float32",), param_name="table_info")
    para_check.check_dtype(em_x.get("dtype").lower(), ("float32",), param_name="em_x")
    para_check.check_dtype(em_.get("dtype").lower(), ("float32",), param_name="em")
    para_check.check_dtype(dy_.get("dtype").lower(), ("float32",), param_name="dy")
    para_check.check_dtype(descriptor.get("dtype").lower(), ("float32",), param_name="descriptor")

    # output dtype check
    para_check.check_dtype(dy_dem_x.get("dtype").lower(), ("float32",), param_name="dy_dem_x")
    para_check.check_dtype(dy_dem.get("dtype").lower(), ("float32",), param_name="dy_dem")

    # input shape check
    para_check.check_shape(table.get("shape"), min_rank=2, max_rank=2, param_name="table")
    para_check.check_shape(table_info.get("shape"), min_rank=1, max_rank=1, min_size=6, max_size=6,
                           param_name="table_info")
    para_check.check_shape(em_x.get("shape"), min_rank=2, max_rank=2, param_name="em_x")
    para_check.check_shape(em_.get("shape"), min_rank=3, max_rank=3, param_name="em")
    para_check.check_shape(dy_.get("shape"), min_rank=3, max_rank=3, param_name="dy")
    para_check.check_shape(descriptor.get("shape"), min_rank=3, max_rank=3, param_name="descriptor")

    # output shape check
    para_check.check_shape(dy_dem_x.get("shape"), min_rank=2, max_rank=2, param_name="dy_dem_x")
    para_check.check_shape(dy_dem.get("shape"), min_rank=3, max_rank=3, param_name="dy_dem")

    if any((split_count < 1, split_index < 0, split_count <= split_index)):
        rule = "Failed to check split info"
        error_manager_vector.raise_err_specific_reson(kernel_name, rule)


@register_operator("TabulateFusionGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def tabulate_fusion_grad(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem, split_count=1, split_index=0,
                         kernel_name="tabulate_fusion_grad"):
    """
    Compute TabulateFusionGrad.

    Parameters
    ----------
    table : dict. shape and dtype of input data table
    table_info : dict. shape and dtype of input data table_info
    em_x : dict. shape and dtype of input data em_x
    em : dict. shape and dtype of input data em
    dy : dict. shape and dtype of output data dy
    descriptor : dict. shape and dtype of output data descriptor
    dy_dem_x : dict. shape and dtype of output data dy_dem_x
    dy_dem : dict. shape and dtype of output data dy_dem
    split_count : int. enable/disable vector core. 1-disable, 2-enable
    split_index : int. index of AI Core/Vector Core. 0-AI Core index, 1-Vector Core Index
    kernel_name : str. cce kernel name, default value is "tabulate_fusion_grad"

    Returns
    -------
    None
    """
    _check_params(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem,
                  split_count, split_index, kernel_name)

    obj = TabulateFusionGrad(table, table_info, em_x, em_, dy_, descriptor, dy_dem_x, dy_dem,
                             split_count, split_index, kernel_name)
    obj.compute()
