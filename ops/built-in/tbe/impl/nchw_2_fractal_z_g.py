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
nchw_2_fractal_z_g
"""
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


# pylint: disable=superfluous-parens,useless-object-inheritance,invalid-name
# pylint: disable=too-many-instance-attributes,too-few-public-methods,unused-argument,no-self-use
# pylint: disable=too-many-arguments,too-many-locals,useless-return,protected-access
class Nchw2Fractalzg(object):
    """
    Nchw2Fractalzg
    """
    class TilingParam(object):
        """
        TilingParam
        """
        def __init__(self, shape_in, shape_out, groups):
            self.cout_orig = shape_in[0] // groups
            self.cin_orig = shape_in[1]
            self.cin_orig_ms = EPB if self.cin_orig > EPB else  self.cin_orig # mjaor section
            self.cin_orig_ts = self.cin_orig % EPB  if self.cin_orig > EPB else 0# tail section
            self.kh = shape_in[2]
            self.kw = shape_in[3]
            self.groups = groups
            self.loop_on_chw = 1
            self.loop_on_groups = self.groups
            self.n = shape_out[1] * shape_out[2]
            self.n16 = self.n // EPB
            self.loop_on_n = 1
            self.c0 = shape_out[3]
            self.khw = self.kh * self.kw
            self.ele_per_line = self.cin_orig_ms * self.kh * self.kw
            self.line_blocks = (self.ele_per_line + EPB - 1) // EPB
            self.ele_per_line_with_tail = self.line_blocks * EPB
            self.tail_ele_num = self.ele_per_line_with_tail - self.ele_per_line
            self.vol_per_c1 = self.khw * self.n * self.c0
            self.unit_height = self.cout_orig * (EPB // self.cin_orig_ms)
            self.group_height = self.cout_orig
            self.summary_t = []
            self.summary = []
            self.partial = []
            self.loops = []

            self._update_loop_on_chw()
            self._update_loop_on_groups()
            self._update_loop_on_n()

            print("\n----------------------tiling_param----------------------")
            print("ub_size = ", UB_SIZE)
            print("core_num = ", CORE_NUM)
            print("cout_orig = ", self.cout_orig, ",cin_orig = ", self.cin_orig)
            print("cin_orig_ms = ", self.cin_orig_ms, ",cin_orig_ts = ", self.cin_orig_ts)
            print("kh = ", self.kh, ", kw = ", self.kw)
            print("groups = ", self.groups, ", n = ", self.n)
            print("loop_on_n = ", self.loop_on_n, ",loop_on_chw = ", self.loop_on_chw,
                  ",loop_on_groups = ", self.loop_on_groups)
            print("ele_per_line =", self.ele_per_line, ",ele_per_line_with_tail =", self.ele_per_line_with_tail,
                  ",tail_ele_num =", self.tail_ele_num)
            print("line_blocks =", self.line_blocks)
            print("summary_t:", self.summary_t)
            print("summary:", self.summary)
            print("loops:", self.loops)

        def _update_loop_on_chw(self):
            if self.cin_orig > EPB:
                self.loop_on_chw = (self.cin_orig + EPB - 1) // EPB

        def _sum_all_loops(self):
            res = 0
            for i in range(CORE_NUM):
                res = res + self.loops[i]
            return res

        def _dispatch_loop(self):
            if self.loop_on_groups < CORE_NUM:
                self.summary = self.summary_t
                while len(self.loops) < self.loop_on_groups:
                    self.loops.append(1)
                while len(self.loops) < MAX_CORE_NUM:
                    self.loops.append(0)
                    self.summary.append(0)
            else:
                each = math.ceil(self.loop_on_groups / CORE_NUM)
                for i in range(CORE_NUM):
                    self.loops.append(each)
                for i in range(CORE_NUM):
                    if self._sum_all_loops() > self.loop_on_groups:
                        self.loops[i] = self.loops[i] - 1
                index = 0
                for i in range(CORE_NUM):
                    self.summary.append(self.summary_t[index])
                    index = index + self.loops[i]
                while len(self.loops) < MAX_CORE_NUM:
                    self.loops.append(0)
                    self.summary.append(0)

        def _update_loop_on_groups(self):
            self.loop_on_groups = 0
            tp = 0
            start = 0
            left = self.cin_orig * self.groups
            self.summary_t.append(0)
            while left > 0:
                self.loop_on_groups = self.loop_on_groups + 1
                if tp > 0:
                    self.summary_t.append(tp + self.summary_t[-1])
                    left = left - tp
                    start = tp
                    tp = 0
                    continue
                ts = start + self.cin_orig_ms
                if ts == EPB:
                    self.summary_t.append(self.cin_orig_ms + self.summary_t[-1])
                    left -= self.cin_orig_ms
                    start = 0
                elif ts > EPB:
                    tp = EPB - start
                    left = left - tp
                    self.summary_t.append(tp + self.summary_t[-1])
                    tp = self.cin_orig_ms - tp
                    start = 0
                else:
                    left = left - self.cin_orig_ms
                    self.summary_t.append(self.cin_orig_ms + self.summary_t[-1])
                    start = ts

            if self.cin_orig > EPB:
                self.loop_on_groups = (self.cin_orig // EPB) * self.groups

            self._dispatch_loop()


        def _update_loop_on_n(self):
            if self.cin_orig > EPB:
                if self.n16 > 1:
                    self.loop_on_n = (self.cin_orig + EPB - 1)// EPB


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

    def _clear_tail_memory(self, tp, head_offset):
        if tp.tail_ele_num != 0:
            effective_ele_offset = tp.cin_orig_ms * tp.khw * EPB
            #performance could be optimized
            self.tik_inst.vector_dup(16, self.ub_input[OFFSET_1 + head_offset + effective_ele_offset],
                                     0, tp.tail_ele_num, 1, 1)

    def _calc_head_offset(self, tp, left_zero, head_offset):
        head_offset.set_as(left_zero * tp.khw * EPB)

    def _calc_src_addr(self, tp, lgs, lcorg, src_addr):
        with self.tik_inst.if_scope(tp.cin_orig <= EPB):
            src_addr.set_as(lgs * tp.cout_orig * tp.cin_orig_ms * tp.khw +\
                            lcorg * tp.cin_orig * tp.khw)
        with self.tik_inst.else_scope():
            src_addr.set_as((lgs // tp.loop_on_chw) * tp.cout_orig * tp.cin_orig * tp.khw +\
                            (lgs % tp.loop_on_chw) * tp.cin_orig_ms * tp.khw +\
                            lcorg * tp.cin_orig * tp.khw)

    def _calc_dst_addr(self, tp, lhw, c1_pos, cur_height, dst_addr):
        dst_addr.set_as(c1_pos * tp.vol_per_c1 + lhw * tp.n * EPB + cur_height * EPB)

    def _calc_cur_height(self, tp, summary, cur_height):
        cur_height.set_as((summary // tp.cin_orig_ms) * tp.cout_orig % tp.n)

    def _reorder(self, tp, ub_input, left_zero, partial):
        tik_inst = self.tik_inst
        ele_num_per_line = tp.line_blocks * EPB

        head_offset = tik_inst.Scalar("int64", init_value=0)
        self._calc_head_offset(tp, left_zero, head_offset)

        active_ele_num = tik_inst.Scalar("int64", init_value=0)
        skip_ele_num = tik_inst.Scalar("int64", init_value=0)

        with tik_inst.if_scope(partial == 0):
            active_ele_num.set_as(tp.cin_orig_ms)
            skip_ele_num.set_as(0)
        with tik_inst.else_scope():
            active_ele_num.set_as(partial)
            skip_ele_num.set_as(tp.cin_orig_ms - active_ele_num)

        # mask,dst,scalar, repeat_times, dst_blk_stride, dst_rep_stride
        tik_inst.vector_dup(128, ub_input[OFFSET_1], 0, 248, 1, 8)

        # step 1 : first vnchwconv
        with tik_inst.for_range(0, tp.loop_on_n) as ln:
            src_addr_list = [ub_input[ln * EPB * tp.ele_per_line_with_tail + ele_num_per_line * i] for i in range(EPB)]
            dst_addr_list = [ub_input[OFFSET_1 + head_offset + ln * EPB + tp.loop_on_n * EPB * i] for i in range(EPB)]
            repeat_cnt_first = tp.line_blocks
            if(repeat_cnt_first == 1):
                src_stride = 0
                dst_stride = 0
            else:
                src_stride = 1
                dst_stride = 16 * tp.loop_on_n
            tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt_first, dst_stride, src_stride)
            self._clear_tail_memory(tp, head_offset)

        tik_inst.vector_dup(128, ub_input[OFFSET_2], 0, 248, 1, 8)

        # step 2 : second vnchwconv
        src_addr_list = [ub_input[OFFSET_1 + skip_ele_num * tp.khw * EPB + tp.loop_on_n * tp.khw * EPB * i]\
                for i in range(EPB)]
        dst_addr_list = [ub_input[OFFSET_2 + EPB * i] for i in range(EPB)]
        repeat_cnt_second = tp.khw * tp.loop_on_n
        if(repeat_cnt_second == 1):
            src_stride = 0
            dst_stride = 0
        else:
            src_stride = 1
            dst_stride = 16
        tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt_second, dst_stride, src_stride)

    def _copy_in(self, tp, tik_inst, ub_offset, lgs, partial):
        src_addr = tik_inst.Scalar("int64", init_value=0)
        ub_offset.set_as(0)
        with tik_inst.for_range(0, tp.cout_orig) as lcorg:
            ele_num_per_burst = tp.cin_orig_ms * tp.khw
            burst_len = math.ceil(ele_num_per_burst / EPB)
            self._calc_src_addr(tp, lgs, lcorg, src_addr)
            tik_inst.data_move(self.ub_input[ub_offset * EPB], self.data_in[src_addr], 0, 1, burst_len, 0, 0)
            ub_offset.set_as(ub_offset + burst_len)

    def _copy_out(self, tp, tik_inst, ub_offset, c1_pos, cur_height):
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        with tik_inst.for_range(0, tp.khw) as lhw:
            self._calc_dst_addr(tp, lhw, c1_pos, cur_height, dst_addr)
            tik_inst.data_move(self.data_out[dst_addr], self.ub_input[OFFSET_2 + tp.loop_on_n * EPB * EPB * lhw],
                               0, 1, tp.cout_orig, 0, 0)

    def _update_param_per_batch(self, tp, tik_inst, summary, c1_pos, lgs, left_zero, partial, cur_height, active):
        line_ele = tik_inst.Scalar("int64", init_value=0)
        summary.set_as(summary + active)
        with tik_inst.if_scope(tp.cin_orig < EPB):
            with tik_inst.if_scope(left_zero >= EPB):
                c1_pos.set_as(c1_pos + 1)
                left_zero.set_as(0)
                with tik_inst.if_scope(active != tp.cin_orig_ms):
                    partial.set_as(tp.cin_orig_ms - active)
                    active.set_as(partial)
            with tik_inst.else_scope():
                partial.set_as(0)
                left_zero.set_as(left_zero + active)
                with tik_inst.if_scope(left_zero == EPB):
                    left_zero.set_as(0)
                    c1_pos.set_as(c1_pos + 1)
                    with tik_inst.if_scope(active != tp.cin_orig_ms):
                        partial.set_as(tp.cin_orig_ms - active)
                        active.set_as(partial)
                    with tik_inst.else_scope():
                        active.set_as(tp.cin_orig_ms)
                with tik_inst.else_scope():
                    line_ele.set_as(left_zero + tp.cin_orig_ms)
                    with tik_inst.if_scope(line_ele > EPB):
                        active.set_as(tp.cin_orig_ms - (line_ele - EPB))
                    with tik_inst.else_scope():
                        active.set_as(tp.cin_orig_ms)
        with tik_inst.else_scope():
            c1_pos.set_as(c1_pos + 1)
            left_zero.set_as(0)
            partial.set_as(0)
            active.set_as(EPB)

        cur_height.set_as((summary // tp.cin_orig_ms) * tp.cout_orig % tp.n)
        lgs.set_as(summary // tp.cin_orig_ms)

    def _get_param_by_block_idx(self, block_idx, tp, summary, loops):
        for i in range(MAX_CORE_NUM):
            with self.tik_inst.if_scope(block_idx == i):
                summary.set_as(tp.summary[i])
                loops.set_as(tp.loops[i])

    def _calc_loop_param(self, block_idx, tp, summary, partial, loops, left_zero, c1_pos, cur_height, lgs):
        self._get_param_by_block_idx(block_idx, tp, summary, loops)
        cur_height.set_as((summary // tp.cin_orig) * tp.cout_orig % tp.n)
        left_zero.set_as(summary % EPB)
        c1_pos.set_as(summary // EPB)
        lgs.set_as(summary // tp.cin_orig_ms)
        #partial.set_as((tp.cin_orig_ms - summary % tp.cin_orig_ms) % tp.cin_orig_ms)
        partial.set_as(tp.cin_orig_ms - summary % tp.cin_orig_ms)

    def _compute(self, tp, data_in, data_out):
        tik_inst = self.tik_inst
        ub_offset = tik_inst.Scalar("int64", init_value=0)
        summary = tik_inst.Scalar("int64", init_value=0)
        partial = tik_inst.Scalar("int64", init_value=0)
        loops = tik_inst.Scalar("int64", init_value=0)
        left_zero = tik_inst.Scalar("int64", init_value=0)
        c1_pos = tik_inst.Scalar("int64", init_value=0)
        cur_height = tik_inst.Scalar("int64", init_value=0)
        lgs = tik_inst.Scalar("int64", init_value=0)
        active = tik_inst.Scalar("int64", init_value=0)

        with tik_inst.for_range(0, CORE_NUM, block_num = CORE_NUM) as block_idx:
            self._calc_loop_param(block_idx, tp, summary, partial, loops, left_zero, c1_pos, cur_height, lgs)
            active.set_as(summary % tp.cin_orig_ms)
            with tik_inst.if_scope(active == 0):
                active.set_as(tp.cin_orig_ms)
            with tik_inst.else_scope():
                active.set_as(tp.cin_orig_ms - active)

            with tik_inst.for_range(0, loops):
                self._copy_in(tp, tik_inst, ub_offset, lgs, partial)
                self._reorder(tp, self.ub_input, left_zero, partial)
                self._copy_out(tp, tik_inst, ub_offset, c1_pos, cur_height)
                self._update_param_per_batch(tp, tik_inst, summary, c1_pos, lgs, left_zero, partial, cur_height, active)
        return


@para_check.check_input_type(dict, dict, str, str, int, str)
def nchw_2_fractal_z_g(src, dst, src_format, dst_format, groups, kernel_name="nchw_2_fractal_z_g"):
    """
    algorithm: nchw_2_fractal_z_g

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
        kernel name, default value is "nchw_2_fractal_z_g"

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
    instance = Nchw2Fractalzg(tik_inst, data_in, data_out)
    instance._check_params(shape_in, shape_out, in_dtype, out_dtype)
    tp = instance._tiling(shape_in, shape_out, groups)
    instance._compute(tp, data_in, data_out)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[data_in], outputs=[data_out])
