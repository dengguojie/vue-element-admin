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
dynamic batch_to_space_nd
"""
# pylint: disable=unused-import
import te.lang.dynamic
from te import platform as tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

# max int32
MAX_INT32 = 2**31 - 1
# tiling param num
TILING_ARG_NUM = 32
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32


# pylint: disable=invalid-name,unused-argument,too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init,too-many-locals,too-many-arguments,too-many-statements
class BatchToSpaceND:
    """Performs batch_to_space_nd on input tensor
    5HD:
        input:          input_b  c1  input_h  input_w  c0
        reshape:        block_h  block_w  output_b  c1  input_h  input_w  c0
        permute(deal):  output_b  c1  input_h  block_h  input_w  block_w  c0
        crops+output:   output_b  c1  output_h  output_w  c0
    6HD:
        input:          input_b  input_d  c1  input_h  input_w  c0
        reshape:        block_d  block_h  block_w  output_b  input_d  c1  input_h  input_w  c0
        permute(deal):  output_b  input_d  block_d  c1  input_h  block_h  input_w  block_w  c0
        crops+output:   output_b  output_d  c1  output_h  output_w  c0
    """

    def __init__(self, dtype, block_size, kernel_name):
        """Init batch_to_space_nd parameters
        """
        self.dtype = dtype
        # zero means batch_to_space_nd; not zeros means batch_to_space
        self.block_size = block_size
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // EIGHT_BIT
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE
        self.ub_ele = self.ub_size // self.dtype_size
        self.blk_ele = BLOCK_BYTES // self.dtype_size
        self.init_gm_tensor()

    def tiling_args(self):
        """Get runtime params from tiling
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.act_core_num = self.tik_instance.Scalar("int32", name="act_core_num")
        self.act_core_num.set_as(self.tiling_ub[1])
        self.one_core_ele = self.tik_instance.Scalar("int32", name="one_core_ele")
        self.one_core_ele.set_as(self.tiling_ub[2])
        self.last_core_ele = self.tik_instance.Scalar("int32", name="last_core_ele")
        self.last_core_ele.set_as(self.tiling_ub[3])
        self.input_b = self.tik_instance.Scalar("int32", name="input_b")
        self.input_b.set_as(self.tiling_ub[4])
        self.block_d = self.tik_instance.Scalar("int32", name="block_d")
        self.block_d.set_as(self.tiling_ub[5])
        self.block_h = self.tik_instance.Scalar("int32", name="block_h")
        self.block_h.set_as(self.tiling_ub[6])
        self.block_w = self.tik_instance.Scalar("int32", name="block_w")
        self.block_w.set_as(self.tiling_ub[7])
        self.crops_f = self.tik_instance.Scalar("int32", name="crops_f")
        self.crops_f.set_as(self.tiling_ub[8])
        self.crops_a = self.tik_instance.Scalar("int32", name="crops_a")
        self.crops_a.set_as(self.tiling_ub[9])
        self.crops_t = self.tik_instance.Scalar("int32", name="crops_t")
        self.crops_t.set_as(self.tiling_ub[10])
        self.crops_b = self.tik_instance.Scalar("int32", name="crops_b")
        self.crops_b.set_as(self.tiling_ub[11])
        self.crops_l = self.tik_instance.Scalar("int32", name="crops_l")
        self.crops_l.set_as(self.tiling_ub[12])
        self.crops_r = self.tik_instance.Scalar("int32", name="crops_r")
        self.crops_r.set_as(self.tiling_ub[13])
        self.input_d = self.tik_instance.Scalar("int32", name="input_d")
        self.input_d.set_as(self.tiling_ub[14])
        self.channel_one = self.tik_instance.Scalar("int32", name="channel_one")
        self.channel_one.set_as(self.tiling_ub[15])
        self.input_h = self.tik_instance.Scalar("int32", name="input_h")
        self.input_h.set_as(self.tiling_ub[16])
        self.input_w = self.tik_instance.Scalar("int32", name="input_w")
        self.input_w.set_as(self.tiling_ub[17])
        self.channel_zero = self.tik_instance.Scalar("int32", name="channel_zero")
        self.channel_zero.set_as(self.tiling_ub[18])
        self.output_b = self.tik_instance.Scalar("int32", name="output_b")
        self.output_b.set_as(self.tiling_ub[19])
        self.output_d = self.tik_instance.Scalar("int32", name="output_d")
        self.output_d.set_as(self.tiling_ub[20])
        self.output_h = self.tik_instance.Scalar("int32", name="output_h")
        self.output_h.set_as(self.tiling_ub[21])
        self.output_w = self.tik_instance.Scalar("int32", name="output_w")
        self.output_w.set_as(self.tiling_ub[22])

    def init_gm_tensor(self):
        """Init gm tensor
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.input_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="input_gm", scope=tik.scope_gm)
        self.block_gm = self.tik_instance.Tensor("int32", (MAX_INT32,), name="block_gm", scope=tik.scope_gm)
        self.crops_gm = self.tik_instance.Tensor("int32", (MAX_INT32,), name="crops_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype, (MAX_INT32,), name="output_gm", scope=tik.scope_gm)

    def run_block_h(self, ub_a, ub_b, core_idx, idx_bh, ele_idx):
        """run block height function.
        """
        # move in and permute
        dst_stride_pt = (self.block_w - 1) * self.channel_zero // self.blk_ele
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # move in
                offset_gm_in = (idx_bh * self.block_w + idx_bw) * self.output_b * self.channel_one * self.input_h * \
                               self.input_w * self.channel_zero + (core_idx * self.one_core_ele + ele_idx) * \
                               self.input_h * self.input_w * self.channel_zero
                offset_ub_in = idx_bw * self.input_h * self.input_w * self.channel_zero
                self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0, 1,
                                            self.input_h * self.input_w * self.channel_zero // self.blk_ele, 0, 0)
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                # permute
                offset_ub_in = idx_bw * self.input_h * self.input_w * self.channel_zero
                offset_ub_pt = idx_bw * self.channel_zero
                self.tik_instance.data_move(ub_b[offset_ub_pt], ub_a[offset_ub_in], 0, self.input_h * self.input_w,
                                            self.channel_zero // self.blk_ele, 0, dst_stride_pt)

        # move out
        start = (self.crops_t - idx_bh + self.block_h - 1) // self.block_h
        end = (self.crops_t + self.output_h - idx_bh + self.block_h - 1) // self.block_h
        offset_gm_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * self.channel_zero + \
                        (idx_bh + start * self.block_h - self.crops_t) * self.output_w * self.channel_zero
        offset_ub_out = start * self.input_w * self.block_w * self.channel_zero + self.crops_l * self.channel_zero
        src_stride_out = (self.crops_l + self.crops_r) * self.channel_zero // self.blk_ele
        dst_stride_out = (self.block_h - 1) * self.output_w * self.channel_zero // self.blk_ele
        with self.tik_instance.if_scope(end > start):
            self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_b[offset_ub_out], 0, end - start,
                                        self.output_w * self.channel_zero // self.blk_ele, src_stride_out,
                                        dst_stride_out)

    def run_block_h_open_db_5hd(self, core_idx, core_ele):
        """run block height for 5hd function, open double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_a", scope=tik.scope_ubuf)
                ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_b", scope=tik.scope_ubuf)
                ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_c", scope=tik.scope_ubuf)
                ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_d", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                    self.run_block_h(ub_a, ub_b, core_idx, idx_bh, ele_idx * 2)
                    self.run_block_h(ub_c, ub_d, core_idx, idx_bh, ele_idx * 2 + 1)
                with self.tik_instance.if_scope(core_ele % 2 == 1):
                    self.run_block_h(ub_a, ub_b, core_idx, idx_bh, core_ele - 1)

    def run_block_h_close_db_5hd(self, core_idx, core_ele):
        """run block height for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                with self.tik_instance.for_range(0, core_ele) as ele_idx:
                    ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,), name="ub_a", scope=tik.scope_ubuf)
                    ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,), name="ub_b", scope=tik.scope_ubuf)
                    self.run_block_h(ub_a, ub_b, core_idx, idx_bh, ele_idx)

    def run_input_h(self, ub_a, ub_b, core_idx, idx_ih, idx_bh, ele_idx):
        """run input height function.
        """
        flag_h = idx_ih * self.block_h + idx_bh
        with self.tik_instance.if_scope(tik.all(flag_h >= self.crops_t, flag_h < self.crops_t + self.output_h)):
            # move in and permute
            dst_stride_pt = (self.block_w - 1) * self.channel_zero // self.blk_ele
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                    # move in
                    offset_gm_in = (idx_bh * self.block_w + idx_bw) * self.output_b * self.channel_one * \
                                   self.input_h * self.input_w * self.channel_zero + \
                                   (core_idx * self.one_core_ele + ele_idx) * self.input_h * self.input_w * \
                                   self.channel_zero + idx_ih * self.input_w * self.channel_zero
                    offset_ub_in = idx_bw * self.input_w * self.channel_zero
                    self.tik_instance.data_move(ub_a[offset_ub_in], self.input_gm[offset_gm_in], 0, 1,
                                                self.input_w * self.channel_zero // self.blk_ele, 0, 0)
            with self.tik_instance.new_stmt_scope(disable_sync=True):
                with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                    # permute
                    offset_ub_in = idx_bw * self.input_w * self.channel_zero
                    offset_ub_pt = idx_bw * self.channel_zero
                    self.tik_instance.data_move(ub_b[offset_ub_pt], ub_a[offset_ub_in], 0, self.input_w,
                                                self.channel_zero // self.blk_ele, 0, dst_stride_pt)

            # move out
            offset_gm_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * \
                            self.channel_zero + (idx_ih * self.block_h + idx_bh - self.crops_t) * self.output_w * \
                            self.channel_zero
            offset_ub_out = self.crops_l * self.channel_zero
            self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_b[offset_ub_out], 0, 1,
                                        self.output_w * self.channel_zero // self.blk_ele, 0, 0)

    def run_input_h_open_db_5hd(self, core_idx, core_ele):
        """run input height for 5hd function, open double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.input_h) as idx_ih:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_a", scope=tik.scope_ubuf)
                    ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_b", scope=tik.scope_ubuf)
                    ub_c = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_c", scope=tik.scope_ubuf)
                    ub_d = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 4,), name="ub_d", scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, core_ele // 2) as ele_idx:
                        self.run_input_h(ub_a, ub_b, core_idx, idx_ih, idx_bh, ele_idx * 2)
                        self.run_input_h(ub_c, ub_d, core_idx, idx_ih, idx_bh, ele_idx * 2 + 1)
                    with self.tik_instance.if_scope(core_ele % 2 == 1):
                        self.run_input_h(ub_a, ub_b, core_idx, idx_ih, idx_bh, core_ele - 1)

    def run_input_h_close_db_5hd(self, core_idx, core_ele):
        """run input height for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.input_h) as idx_ih:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    with self.tik_instance.for_range(0, core_ele) as ele_idx:
                        ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                        name="ub_a",
                                                        scope=tik.scope_ubuf)
                        ub_b = self.tik_instance.Tensor(self.dtype, (self.ub_ele // 2,),
                                                        name="ub_b",
                                                        scope=tik.scope_ubuf)
                        self.run_input_h(ub_a, ub_b, core_idx, idx_ih, idx_bh, ele_idx)

    def run_block_w(self, ub_a, core_idx, idx_ih, idx_bh, idx_bw, ele_idx):
        """run block width function.
        """
        flag_h = idx_ih * self.block_h + idx_bh
        with self.tik_instance.if_scope(tik.all(flag_h >= self.crops_t, flag_h < self.crops_t + self.output_h)):
            # move in
            offset_gm_in = (idx_bh * self.block_w + idx_bw) * self.output_b * self.channel_one * self.input_h * \
                           self.input_w * self.channel_zero + (core_idx * self.one_core_ele + ele_idx) * \
                           self.input_h * self.input_w * self.channel_zero + idx_ih * self.input_w * self.channel_zero
            self.tik_instance.data_move(ub_a, self.input_gm[offset_gm_in], 0, 1,
                                        self.input_w * self.channel_zero // self.blk_ele, 0, 0)
            # move out
            start = (self.crops_l - idx_bw + self.block_w - 1) // self.block_w
            end = (self.crops_l + self.output_w - idx_bw + self.block_w - 1) // self.block_w
            offset_gm_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * \
                            self.channel_zero + (idx_ih * self.block_h + idx_bh - self.crops_t) * self.output_w * \
                            self.channel_zero + (idx_bw + start * self.block_w - self.crops_l) * self.channel_zero
            offset_ub_out = start * self.channel_zero
            dst_stride_out = (self.block_w - 1) * self.channel_zero // self.blk_ele
            with self.tik_instance.if_scope(end > start):
                self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_a[offset_ub_out], 0, end - start,
                                            self.channel_zero // self.blk_ele, 0, dst_stride_out)

    def run_block_w_5hd(self, core_idx, core_ele):
        """run block width for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.input_h) as idx_ih:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                        with self.tik_instance.for_range(0, core_ele) as ele_idx:
                            ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele,),
                                                            name="ub_a",
                                                            scope=tik.scope_ubuf)
                            self.run_block_w(ub_a, core_idx, idx_ih, idx_bh, idx_bw, ele_idx)

    def run_input_w(self, ub_a, core_idx, idx_ih, idx_bh, idx_iw, idx_bw, ele_idx):
        """run input height function.
        """
        flag_h = idx_ih * self.block_h + idx_bh
        flag_w = idx_iw * self.block_w + idx_bw
        with self.tik_instance.if_scope(
                tik.all(flag_h >= self.crops_t, flag_h < self.crops_t + self.output_h, flag_w >= self.crops_l,
                        flag_w < self.crops_l + self.output_w)):
            # move in
            offset_gm_in = (idx_bh * self.block_w + idx_bw) * self.output_b * self.channel_one * self.input_h * \
                           self.input_w * self.channel_zero + (core_idx * self.one_core_ele + ele_idx) * \
                           self.input_h * self.input_w * self.channel_zero + (idx_ih * self.input_w + idx_iw) * \
                           self.channel_zero
            self.tik_instance.data_move(ub_a, self.input_gm[offset_gm_in], 0, 1, self.channel_zero // self.blk_ele, 0,
                                        0)

            # move out
            offset_gm_out = (core_idx * self.one_core_ele + ele_idx) * self.output_h * self.output_w * \
                            self.channel_zero + (idx_ih * self.block_h + idx_bh - self.crops_t) * self.output_w * \
                            self.channel_zero + (idx_iw * self.block_w + idx_bw - self.crops_l) * self.channel_zero
            self.tik_instance.data_move(self.output_gm[offset_gm_out], ub_a, 0, 1, self.channel_zero // self.blk_ele, 0,
                                        0)

    def run_input_w_5hd(self, core_idx, core_ele):
        """run input height for 5hd function, close double buffer.
        """
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.input_h) as idx_ih:
                with self.tik_instance.for_range(0, self.block_h) as idx_bh:
                    with self.tik_instance.for_range(0, self.input_w) as idx_iw:
                        with self.tik_instance.for_range(0, self.block_w) as idx_bw:
                            with self.tik_instance.for_range(0, core_ele) as ele_idx:
                                ub_a = self.tik_instance.Tensor(self.dtype, (self.ub_ele,),
                                                                name="ub_a",
                                                                scope=tik.scope_ubuf)
                                self.run_input_w(ub_a, core_idx, idx_ih, idx_bh, idx_iw, idx_bw, ele_idx)

    def batch_to_space_nd_compute_tiling(self):
        """BatchToSpaceND compute tiling
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_idx:
            # define tiling ub and move tiling gm to tiling ub,then get tiling args
            self.tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,),
                                                      name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
            self.tiling_args()

            # call select tiling mode function
            core_ele = self.tik_instance.Scalar("int32", name="core_ele")
            with self.tik_instance.if_scope(core_idx <= self.act_core_num - 1):
                with self.tik_instance.if_scope(core_idx < self.act_core_num - 1):
                    core_ele.set_as(self.one_core_ele)
                with self.tik_instance.else_scope():
                    core_ele.set_as(self.last_core_ele)
                # when format is NC1HWC0, can copy input_h * input_w * block_w * c0, open double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 0):
                    self.run_block_h_open_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy input_h * input_w * block_w * c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 1):
                    self.run_block_h_close_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy input_w * block_w * c0, open double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 2):
                    self.run_input_h_open_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy input_w * block_w * c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    self.run_input_h_close_db_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy input_w * c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 4):
                    self.run_block_w_5hd(core_idx, core_ele)
                # when format is NC1HWC0, can copy c0, close double buffer
                with self.tik_instance.if_scope(self.tiling_mode == 5):
                    self.run_input_w_5hd(core_idx, core_ele)

    def batch_to_space_nd_operator(self):
        """BatchToSpaceND operator
        """
        self.batch_to_space_nd_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.block_gm, self.crops_gm],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)

        tbe_context.get_context().add_compile_info("vars", {
            "ub_ele": self.ub_ele,
            "core_num": self.core_num,
            "block_size": self.block_size,
        })

        return self.tik_instance


@register_operator("BatchToSpaceND")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def batch_to_space_nd(x, block_shape, crops, y, kernel_name="batch_to_space_nd"):
    """BatchToSpaceND for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    block_shape: dict
        the dict of block_shape tensor.
    crops: dict
        the dict of crops tensor.
    y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "batch_to_space_nd".

    Returns
    -------
    None.
    """
    # get input shape, format and dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")

    # check input shape, format and dtype
    para_check.check_shape(input_shape, param_name="x")
    para_check.check_dtype(input_dtype, ("float16", "float32"), param_name="x")
    if input_format not in ("NC1HWC0", "NDC1HWC0"):
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", "NC1HWC0,NDC1HWC0", input_format)

    # run tik
    obj = BatchToSpaceND(input_dtype, 0, kernel_name)
    obj.batch_to_space_nd_operator()
