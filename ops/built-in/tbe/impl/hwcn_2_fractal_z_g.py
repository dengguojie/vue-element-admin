# Copyright 2020 Huawei Technologies Co., Ltd
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
hwcn_2_fractal_z_g
"""
import functools
import math
from te import tik
from te import platform as tbe_platform
from te.utils import para_check

EPB = 16
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
OFFSET_1 = 256 * 256
OFFSET_2 = 0
MAX_CORE_NUM = 32


class Hwcn2Fractalzg(object):
    """
    Hwcn2Fractalzg
    """
    class TilingParam(object):
        """
        TilingParam
        """
        def __init__(self, shape_in, shape_out, groups):
            self.cout_orig = shape_in[3] // groups
            self.cin_orig = shape_in[2]
            self.cin_orig_ms = EPB if self.cin_orig > EPB else  self.cin_orig # mjaor section
            self.cin_orig_ts = self.cin_orig % EPB  if self.cin_orig > EPB else 0# tail section
            self.kh = shape_in[0]
            self.kw = shape_in[1]
            self.groups = groups
            self.loop_on_chw = 1
            self.loop_on_groups = self.groups
            self.loop_on_cn = 1
            self.loop_on_src_c = 1
            self.src_c = shape_in[2]
            self.src_n = shape_in[3]
            self.src_n_ms = self.src_n
            self.dst_n = shape_out[1] * shape_out[2]
            self.src_n_blocks = self.src_n_ms // EPB
            self.vol_cn = self.src_c * self.src_n
            self.c0 = shape_out[3]
            self.khw = self.kh * self.kw
            self.vol_gc1 = shape_out[0] // self.khw
            self.g = 1
            self.c1 = 1
            self.summary_t = []
            self.summary = []
            self.partial = []
            self.loops = []
            self._update_g()
            self._update_loop_on_src_c()

            self.src_c_ms = self.src_c // self.loop_on_src_c
            self.vol_cn_ms = self.vol_cn // self.loop_on_src_c
            self.vol_hwnc0 = self.khw * self.dst_n * EPB
            self.vol_c1hwnc0 = self.c1 * self.khw * self.dst_n * EPB
            self.loop_on_c0 = EPB // self.cin_orig_ms
            self.height_per_c1 = self.loop_on_c0 * self.cout_orig
            self.interweave_num = math.ceil(self.src_c_ms / EPB)

            print("\n----------------------tiling_param----------------------")
            print("ub_size =", UB_SIZE)
            print("core_num =", CORE_NUM)
            print("groups =", self.groups)
            print("cout_orig =", self.cout_orig, ",cin_orig =", self.cin_orig)
            print("cin_orig_ms =", self.cin_orig_ms, ",cin_orig_ts =", self.cin_orig_ts)
            print("kh =", self.kh, ", kw = ", self.kw)
            print("src_c =", self.src_c, ", src_n =", self.src_n, ",dst_n =", self.dst_n)
            print("src_n_blocks =", self.src_n_blocks)
            print("vol_cn =", self.vol_cn)
            print("vol_cn_ms =", self.vol_cn_ms)
            print("vol_gc1 =", self.vol_gc1)
            print("g =", self.g)
            print("c1 =", self.c1)
            print("vol_hwnc0 =", self.vol_hwnc0)
            print("vol_c1hwnc0 =", self.vol_c1hwnc0)
            print("loop_on_c0=", self.loop_on_c0)
            print("height_per_c1 =", self.height_per_c1)
            print("loop_on_src_c=", self.loop_on_src_c)
            print("interweave_num=", self.interweave_num)
            print("--------------------------------------------------------")

        def _ceil(self, m, n):
            return (m + n - 1) // n

        def _lcm(self, m, n):
            return (m * n) // math.gcd(m, n)

        def _update_g(self):
            E = min(self._lcm(self._lcm(self.cin_orig, EPB) // self.cin_orig,
                              self._lcm(self.cout_orig, EPB) // self.cout_orig),
                    self.groups)
            self.g = self._ceil(self.groups, E)
            self.c1 = self.vol_gc1 // self.g

        def _update_loop_on_src_c(self):
            if self.src_c > 32:
                self.loop_on_src_c = 4

    def __init__(self, tik_inst, data_in, data_out):
        self.tik_inst = tik_inst
        self.ub_input = tik_inst.Tensor("float16", (UB_SIZE // 2, ), tik.scope_ubuf, "ub_input")
        self.data_in = data_in
        self.data_out = data_out

    def _tiling(self, shape_in, shape_out, groups):
        tp = self.TilingParam(shape_in, shape_out, groups)
        return tp

    def _check_params(self, shape_in, shape_out, in_dtyppe, out_dtype):
        return

    def _calc_src_addr(self, tp, lk, lsc, lcn, src_addr):
        src_addr.set_as(lk * tp.vol_cn + lsc * tp.vol_cn_ms)

    def _calc_dst_addr(self, tp, lk, lsc, lc1, lc0, dst_addr):
        dis_top = lc0 * tp.cout_orig * EPB
        left_zero =  lc0 * tp.cin_orig_ms
        dst_addr.set_as(lk * tp.dst_n * EPB +\
                        lsc * tp.khw * tp.dst_n * EPB +\
                        lc1 * tp.vol_hwnc0 +\
                        (lc1 * tp.height_per_c1 % tp.dst_n) * EPB +\
                        lc0 * tp.cout_orig * EPB +\
                        lc0 * tp.cin_orig_ms)

    def _reorder(self, tp, ub_input):
        tik_inst = self.tik_inst
        ele_num_per_line = tp.src_n_blocks * EPB

        # mask,dst,scalar, repeat_times, dst_blk_stride, dst_rep_stride
        tik_inst.vector_dup(128, ub_input[OFFSET_1], 0, 248, 1, 8)

        with tik_inst.for_range(0, tp.interweave_num) as li:
            with tik_inst.for_range(0, tp.src_n_ms // EPB) as ln:
                src_addr_list = [ub_input[li * tp.src_n_ms * EPB + ele_num_per_line * i] for i in range(EPB)]
                dst_addr_list = [ub_input[OFFSET_1 + li * tp.src_n_ms * EPB + EPB * i] for i in range(EPB)]
                repeat_cnt = tp.src_n_blocks
                if(repeat_cnt == 1):
                    src_stride = 0
                    dst_stride = 0
                else:
                    src_stride = 1
                    dst_stride = 16
                tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

    def _copy_in(self, tp, tik_inst, ub_offset, lk, lsc):
        src_addr = tik_inst.Scalar("int64", init_value=0)
        ub_offset.set_as(0)
        with tik_inst.for_range(0, tp.loop_on_cn) as lcn:
            burst_len = math.ceil(tp.src_c_ms * tp.src_n / EPB)
            self._calc_src_addr(tp, lk, lsc, lcn, src_addr)
            tik_inst.data_move(self.ub_input[ub_offset * EPB], self.data_in[src_addr], 0, 1, burst_len, 0, 0)
            ub_offset.set_as(ub_offset + burst_len)

    def _is_overlap(self, tp, lc1, lc0):
        if (tp.cin_orig % EPB == 0):
            return False
        return (lc1 * tp.height_per_c1 + (lc0 + 1) * tp.cout_orig) % tp.dst_n == 0

    def _copy_out_gt_16(self, tp, tik_inst, ub_offset, lk, lsc):
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        seq  = tik_inst.Scalar("int64", init_value=0)

        with tik_inst.for_range(0, tp.c1 // tp.loop_on_src_c) as lc1:
            with tik_inst.for_range(0, tp.loop_on_c0) as lc0:
                self._calc_dst_addr(tp, lk, lsc, lc1, lc0, dst_addr)
                tik_inst.data_move(self.data_out[dst_addr],
                                   self.ub_input[OFFSET_1 + seq * tp.src_n_ms * EPB],
                                   0,
                                   tp.g,
                                   tp.dst_n,
                                   0,
                                   tp.khw * tp.dst_n * tp.c1 - tp.cout_orig)
                seq.set_as(seq + 1)

    def _copy_out_lt_16(self, tp, tik_inst, ub_offset, lk, lsc):
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        tail_block_addr = tik_inst.Scalar("int64", init_value=0)
        tail_start_addr = tik_inst.Scalar("int64", init_value=0)
        tail_dst_addr = tik_inst.Scalar("int64", init_value=0)
        store_addr = tik_inst.Scalar("int64", init_value=0)
        seq  = tik_inst.Scalar("int64", init_value=0)

        with tik_inst.for_range(0, tp.c1) as lc1:
            with tik_inst.for_range(0, tp.loop_on_c0) as lc0:
                self._calc_dst_addr(tp, lk, lsc, lc1, lc0, dst_addr)
                with tik_inst.if_scope(self._is_overlap(tp, lc1, lc0) == False):
                    tik_inst.data_move(self.data_out[dst_addr],
                                       self.ub_input[OFFSET_1 + seq * tp.cout_orig * EPB],
                                       0,
                                       tp.g,
                                       tp.cout_orig,
                                       tp.dst_n - tp.cout_orig,
                                       tp.khw * tp.dst_n * tp.c1 - tp.cout_orig)

                with tik_inst.else_scope():
                    tik_inst.data_move(self.data_out[dst_addr],
                                       self.ub_input[OFFSET_1 + seq * tp.cout_orig * EPB],
                                       0,
                                       tp.g,
                                       tp.cout_orig - 1,
                                       tp.dst_n - tp.cout_orig + 1,
                                       tp.khw * tp.dst_n * tp.c1 - tp.cout_orig + 1)

                    tik_inst.vector_dup(128, self.ub_input[OFFSET_2], 0, math.ceil(tp.khw / 8), 1, 8)

                    tail_dst_addr.set_as(dst_addr + (tp.cout_orig - 1) * EPB - (EPB - tp.cin_orig))

                    with tik_inst.for_range(0, tp.g) as lg:

                        tail_block_addr.set_as(OFFSET_1 + (tp.dst_n - tp.cout_orig) * EPB + lg * tp.dst_n * EPB)
                        store_addr.set_as(OFFSET_2 + EPB - tp.cin_orig + lg * EPB)

                        with tik_inst.for_range(0, tp.cin_orig) as k:
                            tail_start_addr.set_as(tail_block_addr + (tp.cout_orig - 1) * EPB)
                            self.ub_input[store_addr + k] = self.ub_input[tail_start_addr + k]

                    tik_inst.data_move(self.data_out[tail_dst_addr], self.ub_input[OFFSET_2],
                                       0, tp.g, 1, 0, tp.khw * tp.dst_n * tp.c1 - 1)

                seq.set_as(seq + 1)

    def _copy_out(self, tp, tik_inst, ub_offset, lk, lsc):
        with tik_inst.if_scope(tp.src_c <= EPB):
            self._copy_out_lt_16(tp, tik_inst, ub_offset, lk, lsc)
        with tik_inst.else_scope():
            self._copy_out_gt_16(tp, tik_inst, ub_offset, lk, lsc)

    def _update_param_per_batch(self, tp, tik_inst, summary, c1_pos, lgs, left_zero, partial, cur_height, active):
        pass

    def _get_param_by_block_idx(self, block_idx, tp, summary, loops):
        pass

    def _get_param_by_block_idx(self, block_idx, tp, summary, loops):
        for i in range(MAX_CORE_NUM):
            with self.tik_inst.if_scope(block_idx == i):
                summary.set_as(tp.summary[i])
                loops.set_as(tp.loops[i])

    def _calc_loop_param(self, block_idx, tp, summary, partial, loops, left_zero, c1_pos, cur_height, lgs):
        pass

    def _compute(self, tp, data_in, data_out):
        tik_inst = self.tik_inst
        ub_offset = tik_inst.Scalar("int64", init_value=0)
        summary = tik_inst.Scalar("int64", init_value=0)
        loops = tik_inst.Scalar("int64", init_value=tp.khw)
        loop_on_kernel = tik_inst.Scalar("int64", init_value=tp.khw)

        with tik_inst.for_range(0, CORE_NUM, block_num = CORE_NUM) as block_idx:
            with tik_inst.if_scope(block_idx == 0):
                with tik_inst.for_range(0, loop_on_kernel) as lk:
                    with tik_inst.for_range(0, tp.loop_on_src_c) as lsc:
                        self._copy_in(tp, tik_inst, ub_offset, lk, lsc)
                        self._reorder(tp, self.ub_input)
                        self._copy_out(tp, tik_inst, ub_offset, lk, lsc)
        return


@para_check.check_input_type(dict, dict, str, str, int, str)
def hwcn_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name="hwcn_2_fractal_z_g"):
    """
    algorithm: hwcn_2_fractal_z_g

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src
    dst_format: str
        data format of dst
    kernel_name: str
        kernel name, default value is "hwcn_2_fractal_z_g"

    Returns
    -------
    tik_instance: tik_instance
    """
    shape_in = src.get("shape")
    shape_out = dst.get("shape")
    in_dtype = src.get("dtype").lower()
    out_dtype = dst.get("dtype").lower()
    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(in_dtype, shape_in, tik.scope_gm, name="data_in")
    data_out = tik_inst.Tensor(out_dtype, shape_out, tik.scope_gm, name="data_out", is_atomic_add=True)
    instance = Hwcn2Fractalzg(tik_inst, data_in, data_out)
    instance._check_params(shape_in, shape_out, in_dtype, out_dtype)
    tp = instance._tiling(shape_in, shape_out, groups)
    instance._compute(tp, data_in, data_out)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[data_in], outputs=[data_out])

