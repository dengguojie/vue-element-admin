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
non_zero
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

dtype_dict = {
    "float16":2,
    "float32":4,
    "int32":4,
    "uint32":4,
    "int64":8
}
UB_MINIMUM_SIZE = 32
SHAPE_DTYPE = "uint32"


def _ceil(x_1, x_2):
    return (x_1 + x_2 - 1) // x_2


class NonZero():
    """Function: use to store nonzero paramters
    """

    def __init__(self, x_shape, x_dtype, y_dtype, kernel_name):
        """Init NonZero base parameters
        """
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile("v200", "aic"))
        self.ub_minimum_num = UB_MINIMUM_SIZE // dtype_dict[x_dtype]
        self.size = x_shape[0] * x_shape[1]
        self.x_shape_one_dim = (self.size, )
        self.tiling = 8192
        self.multi_core_partition()
        self.init_tensor()

    def init_tensor(self):
        # Number of non_zero elements
        self.num = self.tik_instance.Scalar(SHAPE_DTYPE, "num", init_value=0)
        # Number of non_zero elements in a single core
        self.num_blk = self.tik_instance.Scalar(SHAPE_DTYPE, "num_blk")

        self.zero_scalar_uint32 = self.tik_instance.Scalar(init_value=0, dtype=SHAPE_DTYPE)
        self.zero_scalar_int32 = self.tik_instance.Scalar(init_value=0, dtype=self.y_dtype)
        self.zero_scalar_fp32 = self.tik_instance.Scalar(init_value=0, dtype=self.x_dtype)
        self.scalar_2 = self.tik_instance.Scalar("uint32", "scalar_2", init_value=2)

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, name="x", scope=tik.scope_gm)
        # Temporary storage of output data in workspace
        self.data_out = self.tik_instance.Tensor(self.y_dtype, (self.core_loop, 2, self.one_core_num),
                                                 name="data_out", scope=tik.scope_gm, is_workspace=True)
        # Temporary storage of output data in workspace
        self.shape_out = self.tik_instance.Tensor(SHAPE_DTYPE, (self.core_loop, self.ub_minimum_num),
                                                  name="shape_out", scope=tik.scope_gm, is_workspace=True)

        # Final output data
        self.res_gm = self.tik_instance.Tensor(self.y_dtype, (2, self.num), name="res_gm", scope=tik.scope_gm)
        # Final output shape
        self.shape_out_gm = self.tik_instance.Tensor(SHAPE_DTYPE, (9,), name="shape_out_gm", scope=tik.scope_gm)

        # The offset of the current core output
        self.offset_gm = self.tik_instance.Scalar(SHAPE_DTYPE, "offset_gm", init_value=0)
        # Multi-core synchronization Tensor
        self.sync_workspace = self.tik_instance.Tensor("int64", (self.core_loop*32//8,), name="barrier_workspace",
                                                       scope=tik.scope_gm, is_workspace=True, is_atomic_add=True)

        # The number of bytes required for the workspace
        workspace_data_out = dtype_dict[self.y_dtype] * self.core_loop * 2 * self.one_core_num
        workspace_shape_out = dtype_dict[SHAPE_DTYPE] * self.core_loop * self.ub_minimum_num
        workspace_sync_barrier = dtype_dict["int64"] * (self.core_loop*32//8)
        self.workspace = [workspace_data_out, workspace_shape_out, workspace_sync_barrier]

    def multi_core_partition(self):
        # Calculate the number of sub-cores, the amount of data calculated per core and
        # the amount of data calculated by the last core
        self.core_loop = 8
        self.one_core_num = _ceil(self.size, self.core_loop)
        self.core_loop = _ceil(self.size, self.one_core_num)
        self.last_core_num = self.size - (self.core_loop - 1) * self.one_core_num
    
    def non_zero_compute(self):
        with self.tik_instance.for_range(0, self.core_loop, block_num=self.core_loop) as blk_idx:
            with self.tik_instance.if_scope(blk_idx < self.core_loop - 1):
                cur_core_num = self.one_core_num
                self.compute_one_core(blk_idx, cur_core_num)
            with self.tik_instance.else_scope():
                cur_core_num = self.last_core_num
                self.compute_one_core(blk_idx, cur_core_num)

            # block_barrier needs to bind more than 1 core
            if self.core_loop > 1:
                self.tik_instance.block_barrier(self.sync_workspace)

            shape_out_ub = self.tik_instance.Tensor(SHAPE_DTYPE, (self.core_loop, self.ub_minimum_num),
                                                    name="shape_out_ub", scope=tik.scope_ubuf)
            shape_out_ub_2 = self.tik_instance.Tensor(SHAPE_DTYPE, (9,), name="shape_out_ub_2", scope=tik.scope_ubuf)

            self.tik_instance.data_move(shape_out_ub, self.shape_out,
                                        sid=0, nburst=1, burst=self.core_loop, src_stride=0, dst_stride=0)

            # Data handling after block_barrier
            self.multi_core_sync(blk_idx, shape_out_ub)

            # The shape_out_ub_2 is (2,2,n), The first number represents the dim number of the output shape
            shape_out_ub_2[0].set_as(self.scalar_2)
            shape_out_ub_2[1].set_as(self.scalar_2)
            shape_out_ub_2[2].set_as(self.num)
            self.tik_instance.data_move(self.shape_out_gm, shape_out_ub_2,
                                        sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)

        # todo init_value
        tbe_context.get_context().add_compile_info("block_dim", self.core_loop)
        tbe_context.get_context().add_compile_info("workspace", self.workspace)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x_gm],
                                   outputs=[self.res_gm, self.shape_out_gm], config={"save_temp_cce_file":True})

        return self.tik_instance

    def compute_one_core(self, blk_idx, cur_core_num):
        tiling_loop = _ceil(cur_core_num, self.tiling)
        tiling_tail = cur_core_num - (tiling_loop - 1) * self.tiling
        # The number of non-zero elements in the current core
        self.res_blk_num_tensor = self.tik_instance.Tensor(SHAPE_DTYPE, (self.ub_minimum_num,),
                                                           name="res_blk_num_tensor", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(8, self.res_blk_num_tensor, self.zero_scalar_uint32, 1, 1, 8)

        with self.tik_instance.for_range(0, tiling_loop) as t_idx:
            with self.tik_instance.if_scope(t_idx < tiling_loop - 1):
                self.compute_one_loop(blk_idx, t_idx, self.tiling)
            with self.tik_instance.else_scope():
                self.compute_one_loop(blk_idx, t_idx, tiling_tail)

        self.tik_instance.data_move(self.shape_out[blk_idx, 0], self.res_blk_num_tensor,
                                    sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)

    def compute_one_loop(self, blk_idx, t_idx, cur_loop_num):
        row, col = self.x_shape
        blk_size = cur_loop_num
        align_num = 64
        all_tail = blk_size % align_num
        # Due to the limitation of the vcmpvs_ne instruction
        # the input elements processed by ub at one time need to be 64 aligned
        blk_align_size = _ceil(blk_size, align_num) * align_num
        x_shape_one_loop = (blk_align_size, )

        x_ub = self.tik_instance.Tensor(self.x_dtype, x_shape_one_loop, name="x_ub", scope=tik.scope_ubuf)

        row_auxiliary_matrix = self.tik_instance.Tensor(self.y_dtype, x_shape_one_loop,
                                                        name="row_auxiliary_matrix", scope=tik.scope_ubuf)
        col_auxiliary_matrix = self.tik_instance.Tensor(self.y_dtype, x_shape_one_loop,
                                                        name="col_auxiliary_matrix", scope=tik.scope_ubuf)

        res_blk_num = self.tik_instance.Scalar(dtype=SHAPE_DTYPE, name="res_blk_num")
        res_blk_num_cur_core = self.tik_instance.Scalar(dtype=SHAPE_DTYPE, name="res_blk_num_cur_core")

        vreduce_mask = self.tik_instance.Tensor(SHAPE_DTYPE, (blk_align_size // 32,),
                                                name="vreduce_mask", scope=tik.scope_ubuf)

        self.v_dup(row_auxiliary_matrix, self.zero_scalar_int32, blk_align_size, [], self.y_dtype)
        self.v_dup(col_auxiliary_matrix, self.zero_scalar_int32, blk_align_size, [], self.y_dtype)

        # Initialize the auxiliary matrix of rows and columns
        with self.tik_instance.for_range(0, blk_size) as _idx:
            offset = blk_idx * self.one_core_num + t_idx * self.tiling + _idx
            row_index = offset // col
            col_index = offset % col
            col_auxiliary_matrix[_idx].set_as(col_index)
            row_auxiliary_matrix[_idx].set_as(row_index)
        
        if all_tail == 0:
            offset = blk_idx * self.one_core_num + t_idx * self.tiling
            self.tik_instance.data_move(x_ub, self.x_gm[offset//col, offset%col], sid=0, nburst=1,
                                        burst=blk_align_size//self.ub_minimum_num, src_stride=0, dst_stride=0)
        else:
            offset = blk_idx * self.one_core_num + t_idx * self.tiling
            dma_burst = blk_size // self.ub_minimum_num
            dma_tail = blk_size % self.ub_minimum_num
            self.v_dup(x_ub, self.zero_scalar_fp32, blk_align_size, [], self.x_dtype)
            if dma_burst > 0:
                self.tik_instance.data_move(x_ub, self.x_gm[offset//col, offset%col], sid=0, nburst=1,
                                            burst=dma_burst, src_stride=0, dst_stride=0)
            # move input elements that are less than ub_minimun
            if dma_tail > 0:
                gm_offset = dma_burst * self.ub_minimum_num + offset
                ub_offset = dma_burst * self.ub_minimum_num
                unit_tensor = self.tik_instance.Tensor(self.x_dtype, (self.ub_minimum_num,),
                                                       name="unit_tensor", scope=tik.scope_ubuf)
                self.tik_instance.data_move(unit_tensor, self.x_gm[gm_offset//col, gm_offset%col],
                                            sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
                with self.tik_instance.for_range(0, dma_tail) as _idx:
                    x_ub[ub_offset + _idx].set_as(unit_tensor[_idx])
        
        self.gen_mask(vreduce_mask, x_ub, self.zero_scalar_fp32, blk_align_size, self.x_dtype)

        dst_ub_row = self.tik_instance.Tensor(self.y_dtype, (blk_align_size,), name="dst_ub_row", scope=tik.scope_ubuf)
        dst_ub_col = self.tik_instance.Tensor(self.y_dtype, (blk_align_size,), name="dst_ub_col", scope=tik.scope_ubuf)

        # Calculate the row index of non-zero elements
        self.tik_instance.vreduce(blk_align_size, dst_ub_row, row_auxiliary_matrix,
                                  vreduce_mask, 1, 1, 8, 1, 0, res_blk_num, "counter")
        # Calculate the col index of non-zero elements
        self.tik_instance.vreduce(blk_align_size, dst_ub_col, col_auxiliary_matrix,
                                  vreduce_mask, 1, 1, 8, 1, 0, None, "counter")

        tail_n = res_blk_num % self.ub_minimum_num
        burst_ub = res_blk_num // self.ub_minimum_num
        res_blk_num_cur_core.set_as(self.res_blk_num_tensor[0])
        data_out_offset = res_blk_num_cur_core
        # move out to workspace
        with self.tik_instance.if_scope(burst_ub > 0):
            self.tik_instance.data_move(self.data_out[blk_idx, 0, data_out_offset], dst_ub_row,
                                        sid=0, nburst=1, burst=burst_ub, src_stride=0, dst_stride=0)
            self.tik_instance.data_move(self.data_out[blk_idx, 1, data_out_offset], dst_ub_col,
                                        sid=0, nburst=1, burst=burst_ub, src_stride=0, dst_stride=0)

        with self.tik_instance.if_scope(tail_n > 0):
            row_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                        name="row_align_tensor", scope=tik.scope_ubuf)
            col_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                        name="col_align_tensor", scope=tik.scope_ubuf)

            tail_offset_gm_scalar = self.tik_instance.Scalar(init_value=0, dtype="int32")
            tail_offset_ub_scalar = self.tik_instance.Scalar(init_value=0, dtype="int32")
            tail_offset_gm = burst_ub * self.ub_minimum_num + res_blk_num_cur_core
            tail_offset_ub = burst_ub * self.ub_minimum_num
            tail_offset_gm_scalar.set_as(tail_offset_gm)
            tail_offset_ub_scalar.set_as(tail_offset_ub)
            with self.tik_instance.if_scope(burst_ub == 0):
                tail_offset_gm_scalar.set_as(res_blk_num_cur_core)
                tail_offset_ub_scalar.set_as(0)
            
            with self.tik_instance.for_range(0, tail_n) as _idx:
                row_align_tensor[_idx].set_as(dst_ub_row[tail_offset_ub + _idx])
                col_align_tensor[_idx].set_as(dst_ub_col[tail_offset_ub + _idx])
            
            self.tik_instance.data_move(self.data_out[blk_idx, 0, tail_offset_gm], row_align_tensor,
                                        sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
            self.tik_instance.data_move(self.data_out[blk_idx, 1, tail_offset_gm], col_align_tensor,
                                        sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
            
        # Update the non-zero elements of the current core
        res_blk_num_cur_core.set_as(res_blk_num_cur_core + res_blk_num)
        self.res_blk_num_tensor[0].set_as(res_blk_num_cur_core)

    def v_dup(self, dst, scalar, size, dst_offset, x_dtype):
        unit = 256 // (dtype_dict[x_dtype])
        repeat = size // unit
        left = size % unit
        repeat_loop = repeat // 255
        repeat_left = repeat % 255

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                self.tik_instance.vector_dup(unit, dst[rpt_idx * 255 * unit], scalar, 255, 1, 8)
        if repeat_left > 0:
            self.tik_instance.vector_dup(unit, dst[repeat_loop * 255 * unit], scalar, repeat_left, 1, 8)
        if left > 0:
            self.tik_instance.vector_dup(left, dst[repeat * unit], scalar, 1, 1, 8)

    def gen_mask(self, dst, src, scalar, size, x_dtype):
        unit = 256 // (dtype_dict[x_dtype])
        repeat = size // unit
        left = size % unit
        repeat_loop = repeat // 255
        repeat_left = repeat % 255

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * 255 * unit
                self.tik_instance.vcmpvs_ne(dst[offset//32], src[offset], scalar, 255, 1, 8)
        if repeat_left > 0:
            offset = repeat_loop * 255 * unit
            self.tik_instance.vcmpvs_ne(dst[offset//32], src[offset], scalar, repeat_left, 1, 8)
        if left > 0:
            offset = (repeat - 1) * 64 + left
            self.tik_instance.vcmpvs_ne(dst[offset//32], src[offset], scalar, 1, 1, 8)

    def multi_core_sync(self, blk_idx, shape_out_ub):
        tmp_ub = self.tik_instance.Tensor(self.y_dtype, (2, self.tiling), name="tmp_ub", scope=tik.scope_ubuf)
        # Calculate the offset of the current core output
        with self.tik_instance.if_scope(blk_idx > 0):
            with self.tik_instance.for_range(0, blk_idx) as o_idx:
                self.num_blk.set_as(shape_out_ub[o_idx, 0])
                self.offset_gm.set_as(self.offset_gm + self.num_blk)
        # Calculate the number of non-zeor elements
        with self.tik_instance.for_range(0, self.core_loop) as _idx:
            self.num_blk.set_as(shape_out_ub[_idx, 0])
            self.num.set_as(self.num + self.num_blk)
        # The number of non-zeor elements in the current core
        self.num_blk.set_as(shape_out_ub[blk_idx, 0])

        mv_out_loop = self.num_blk // self.tiling
        mv_out_tail = self.num_blk % self.tiling

        with self.tik_instance.if_scope(mv_out_loop > 0):
            with self.tik_instance.for_range(0, mv_out_loop) as mvo_idx:
                mvo_offset = mvo_idx * self.tiling
                # workspace to UB
                self.tik_instance.data_move(tmp_ub[0, 0], self.data_out[blk_idx, 0, mvo_offset],
                                            sid=0, nburst=1, burst=self.tiling // self.ub_minimum_num,
                                            src_stride=0, dst_stride=0)
                self.tik_instance.data_move(tmp_ub[1, 0], self.data_out[blk_idx, 1, mvo_offset],
                                            sid=0, nburst=1, burst=self.tiling // self.ub_minimum_num,
                                            src_stride=0, dst_stride=0)
                # UB to GM
                self.tik_instance.data_move(self.res_gm[0, self.offset_gm + mvo_offset], tmp_ub[0, 0],
                                            sid=0, nburst=1, burst=self.tiling // self.ub_minimum_num,
                                            src_stride=0, dst_stride=0)
                self.tik_instance.data_move(self.res_gm[1, self.offset_gm + mvo_offset], tmp_ub[1, 0],
                                            sid=0, nburst=1, burst=self.tiling // self.ub_minimum_num,
                                            src_stride=0, dst_stride=0)

        tail_n = mv_out_tail % self.ub_minimum_num
        burst_ub = mv_out_tail // self.ub_minimum_num

        with self.tik_instance.if_scope(mv_out_tail > 0):
            mvo_offset = mv_out_loop * self.tiling
            with self.tik_instance.if_scope(burst_ub > 0):
                # workspace to UB
                self.tik_instance.data_move(tmp_ub[0, 0], self.data_out[blk_idx, 0, mvo_offset],
                                            sid=0, nburst=1, burst=burst_ub, src_stride=0, dst_stride=0)
                self.tik_instance.data_move(tmp_ub[1, 0], self.data_out[blk_idx, 1, mvo_offset],
                                            sid=0, nburst=1, burst=burst_ub, src_stride=0, dst_stride=0)
                # UB to GM
                self.tik_instance.data_move(self.res_gm[0, self.offset_gm + mvo_offset], tmp_ub[0, 0],
                                            sid=0, nburst=1, burst=burst_ub, src_stride=0, dst_stride=0)
                self.tik_instance.data_move(self.res_gm[1, self.offset_gm + mvo_offset], tmp_ub[1, 0],
                                            sid=0, nburst=1, burst=burst_ub, src_stride=0, dst_stride=0)

            with self.tik_instance.if_scope(tail_n > 0):
                # Case 1, borrow data from the back to prevent tramplling between multiple cores
                with self.tik_instance.if_scope(self.num_blk < 8):
                    row_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                                name="row_align_tensor", scope=tik.scope_ubuf)
                    col_align_tensor = self.tik_instance.Tensor(self.y_dtype, (self.ub_minimum_num,),
                                                                name="col_align_tensor", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, tail_n) as _idx:
                        row_align_tensor[_idx].set_as(self.data_out[blk_idx, 0, _idx])
                        col_align_tensor[_idx].set_as(self.data_out[blk_idx, 1, _idx])

                    next_num_blk = self.tik_instance.Scalar(SHAPE_DTYPE, "next_num_blk")
                    remain = self.tik_instance.Scalar(SHAPE_DTYPE, "remain")
                    loop_size = self.tik_instance.Scalar(SHAPE_DTYPE, "loop_size")
                    remain.set_as(self.ub_minimum_num - tail_n)

                    with self.tik_instance.for_range(blk_idx + 1, self.core_loop) as b_idx:
                        next_num_blk.set_as(self.shape_out[b_idx, 0])

                        with self.tik_instance.if_scope(next_num_blk < remain):
                            loop_size.set_as(next_num_blk)
                        with self.tik_instance.else_scope():
                            loop_size.set_as(remain)
                        with self.tik_instance.for_range(0, loop_size) as n_idx:
                            with self.tik_instance.if_scope(remain > 0):
                                row_align_tensor[8 - remain].set_as(self.data_out[b_idx, 0, n_idx])
                                col_align_tensor[8 - remain].set_as(self.data_out[b_idx, 1, n_idx])
                                remain.set_as(remain - 1)

                    out_gm_offset = self.offset_gm
                    self.tik_instance.data_move(self.res_gm[0, out_gm_offset], row_align_tensor,
                                                sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
                    self.tik_instance.data_move(self.res_gm[1, out_gm_offset], col_align_tensor,
                                                sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)

                # Case 2, use gm_address_back to prevent tramplling between multiple cores
                with self.tik_instance.else_scope():
                    ub_offset = burst_ub * self.ub_minimum_num
                    gm_offset = (burst_ub - 1) * self.ub_minimum_num + tail_n + mvo_offset

                    self.tik_instance.data_move(tmp_ub[0, ub_offset], self.data_out[blk_idx, 0, gm_offset],
                                                sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
                    self.tik_instance.data_move(tmp_ub[1, ub_offset], self.data_out[blk_idx, 1, gm_offset],
                                                sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)

                    out_gm_offset = self.offset_gm + gm_offset
                    self.tik_instance.data_move(self.res_gm[0, out_gm_offset], tmp_ub[0, ub_offset],
                                                sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
                    self.tik_instance.data_move(self.res_gm[1, out_gm_offset], tmp_ub[1, ub_offset],
                                                sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator("NonZero")
def non_zero(x, y, transpose, kernel_name="non_zero"):
    """
    return a 2-D tensor where each row is the index for a nonzero value

    Paramters
    ---------
    x: dict
        data of input, support "float32"
    y: dict
        index of output
    kernel_name: str
        kernel_name, default value is "non_zero"

    Returns
    ---------
    tik_instance
    """
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    y_dtype = "int32"
    obj = NonZero(x_shape, x_dtype, y_dtype, kernel_name)
    tik_instance = obj.non_zero_compute()
    return tik_instance