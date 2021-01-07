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


# pylint: disable=superfluous-parens,useless-object-inheritance
# pylint: invalid-name,too-many-instance-attributes,too-many-arguments
# pylint: unused-variable,too-few-public-methods,too-many-statements
# pylint: no-self-use,consider-using-enumerate,unused-argument,too-many-locals
# pylint: singleton-comparison,useless-return,protected-access
class Hwcn2Fractalzg(object):
    """
    Hwcn2Fractalzg
    """
    class TilingParam(object):
        """
        TilingParam
        """
        class PerCoreParam(object):
            """
            PerCoreParam
            """
            def __init__(self):
                self.loop_kernel_base = []
                self.loop_kernel_repeat = []
                self.loop_src_c_base = []
                self.loop_src_c_repeat = []
                self.loop_src_n_base = []
                self.loop_src_n_repeat = []

            def _composite(self, pf_kernel_base, pf_kernel_repeat,
                           pf_src_c_base, pf_src_c_repeat,
                           pf_src_n_base, pf_src_n_repeat):
                len_k = len(pf_kernel_base)
                len_c = len(pf_src_c_base)
                len_n = len(pf_src_n_base)
                for i in range(len_k):
                    for j in range(len_c):
                        for k in range(len_n):
                            m = i * j * k + j * k + k
                            self.loop_kernel_base.append(pf_kernel_base[i])
                            self.loop_kernel_repeat.append(pf_kernel_repeat[i])
                            self.loop_src_c_base.append(pf_src_c_base[j])
                            self.loop_src_c_repeat.append(pf_src_c_repeat[j])
                            self.loop_src_n_base.append(pf_src_n_base[k])
                            self.loop_src_n_repeat.append(pf_src_n_repeat[k])

                self._pad_zero(len_k * len_c * len_n)

            def _pad_zero(self, cur_len):
                for i in range(MAX_CORE_NUM - cur_len):
                    self.loop_kernel_base.append(0)
                    self.loop_kernel_repeat.append(0)
                    self.loop_src_c_base.append(0)
                    self.loop_src_c_repeat.append(0)
                    self.loop_src_n_base.append(0)
                    self.loop_src_n_repeat.append(0)

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
            self.pf_kernel_base = []
            self.pf_kernel_repeat = []
            self.pf_src_c_base = []
            self.pf_src_c_repeat = []
            self.pf_src_n_base = []
            self.pf_src_n_repeat = []
            self._update_g()
            self._update_loop_on_src_c()

            self.src_c_ms = self.src_c // self.loop_on_src_c
            self.vol_cn_ms = self.vol_cn // self.loop_on_src_c
            self.vol_hwnc0 = self.khw * self.dst_n * EPB
            self.vol_c1hwnc0 = self.c1 * self.khw * self.dst_n * EPB
            self.loop_on_c0 = EPB // self.cin_orig_ms
            self.height_per_c1 = self.loop_on_c0 * self.cout_orig

            self._dispatch_loop()

            self.pcp = self.PerCoreParam()
            self.pcp._composite(self.pf_kernel_base, self.pf_kernel_repeat,
                                self.pf_src_c_base, self.pf_src_c_repeat,
                                self.pf_src_n_base, self.pf_src_c_repeat)

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
            print("loop_on_c0 =", self.loop_on_c0)
            print("height_per_c1 =", self.height_per_c1)
            print("loop_on_src_c =", self.loop_on_src_c)
            print("loop_kernel_base   =", self.pcp.loop_kernel_base)
            print("loop_kernel_repeat =", self.pcp.loop_kernel_repeat)
            print("loop_src_c_base    =", self.pcp.loop_src_c_base)
            print("loop_src_c_repeat  =", self.pcp.loop_src_c_repeat)
            print("loop_src_n_base    =", self.pcp.loop_src_n_base)
            print("loop_src_n_repeat  =", self.pcp.loop_src_n_repeat)
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
            if self.src_c > EPB:
                self.loop_on_src_c = self.src_c // EPB

        def _sum_all(self, loop):
            res = 0
            for i in range(len(loop)):
                res = res + loop[i]
            return res

        def _remove_zero_data(self, tup):
            tup = [x for x in tup if x != 0]

        def _each_factor_process_num(self, total, factor, unit, base, repeat):
            if total <= unit:
                base.append(0)
                repeat.append(1)
            else:
                each_factor = []
                share = total // unit
                item = math.ceil(share / factor)
                for i in range(factor):
                    each_factor.append(item)
                for i in reversed(range(len(each_factor))):
                    if self._sum_all(each_factor) > share:
                        each_factor[i] = each_factor[i] - 1
                self._remove_zero_data(each_factor)
                r = 0
                for i in range(len(each_factor)):
                    base.append(r)
                    #r = r + each_factor[i] * unit
                    r = r + each_factor[i]
                    repeat.append(each_factor[i])

        def _dispatch_loop(self):
            factor_k = self.khw
            factor_c = 1
            factor_n = 1
            if factor_k >= CORE_NUM:
                factor_k = CORE_NUM

            # step 1: kernel factor
            unit = 1
            self._each_factor_process_num(self.khw, factor_k, unit, self.pf_kernel_base, self.pf_kernel_repeat)

            # step 2: src_c factor
            if self.src_c > EPB:
                factor_c = self.src_c // EPB if CORE_NUM // factor_k > self.src_c // EPB else CORE_NUM // factor_k
            unit = EPB
            self._each_factor_process_num(self.src_c, factor_c, unit, self.pf_src_c_base, self.pf_src_c_repeat)

            # step 3: src_n factor
            factor_n = CORE_NUM // (factor_k * factor_c)
            unit = 2048
            if factor_n == 0:
                factor_n = 1
            self._each_factor_process_num(self.src_n, factor_n, unit, self.pf_src_n_base, self.pf_src_n_repeat)

            #while len(self.pf_kernel_repeat) < MAX_CORE_NUM:
            #    self.pf_kernel_base.append(0)
            #    self.pf_kernel_repeat.append(0)

        def _composite_loop(self):
            pass


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
        left_zero = lc0 * tp.cin_orig_ms
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

        with tik_inst.for_range(0, tp.src_n_ms // EPB) as ln:
            src_addr_list = [ub_input[ele_num_per_line * i] for i in range(EPB)]
            dst_addr_list = [ub_input[OFFSET_1 + EPB * i] for i in range(EPB)]
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
        lcn = 0
        burst_len = math.ceil(tp.src_c_ms * tp.src_n / EPB)
        self._calc_src_addr(tp, lk, lsc, lcn, src_addr)
        tik_inst.data_move(self.ub_input[ub_offset * EPB], self.data_in[src_addr], 0, 1, burst_len, 0, 0)
        ub_offset.set_as(ub_offset + burst_len)

    def _is_overlap(self, tp, lc1, lc0):
        if tp.cin_orig % EPB == 0:
            return False
        return (lc1 * tp.height_per_c1 + (lc0 + 1) * tp.cout_orig) % tp.dst_n == 0

    def _copy_out_gt_16(self, tp, tik_inst, ub_offset, lk, lsc):
        dst_addr = tik_inst.Scalar("int64", init_value=0)
        seq = tik_inst.Scalar("int64", init_value=0)

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
        seq = tik_inst.Scalar("int64", init_value=0)

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

    def _get_param_by_block_idx(self, block_idx, tp, pc_kernel_base, pc_kernel_repeat,
                                pc_src_c_base, pc_src_c_repeat, pc_src_n_base, pc_src_n_repeat):
        for i in range(MAX_CORE_NUM):
            with self.tik_inst.if_scope(block_idx == i):
                pc_kernel_base.set_as(tp.pcp.loop_kernel_base[i])
                pc_kernel_repeat.set_as(tp.pcp.loop_kernel_repeat[i])
                pc_src_c_base.set_as(tp.pcp.loop_src_c_base[i])
                pc_src_c_repeat.set_as(tp.pcp.loop_src_c_repeat[i])
                pc_src_n_base.set_as(tp.pcp.loop_src_n_base[i])
                pc_src_n_repeat.set_as(tp.pcp.loop_src_n_repeat[i])

    def compute(self, tp, data_in, data_out):
        """
        hwcn_2_fractal_z_g entrance function
        """
        tik_inst = self.tik_inst
        ub_offset = tik_inst.Scalar("int64", init_value=0)
        pc_kernel_base = tik_inst.Scalar("int64", init_value=0)
        pc_kernel_repeat = tik_inst.Scalar("int64", init_value=tp.khw)
        pc_src_c_base = tik_inst.Scalar("int64", init_value=0)
        pc_src_c_repeat = tik_inst.Scalar("int64", init_value=tp.khw)
        pc_src_n_base = tik_inst.Scalar("int64", init_value=0)
        pc_src_n_repeat = tik_inst.Scalar("int64", init_value=tp.khw)

        with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
            self._get_param_by_block_idx(block_idx, tp, pc_kernel_base, pc_kernel_repeat,
                                         pc_src_c_base, pc_src_c_repeat, pc_src_n_base, pc_src_n_repeat)
            with tik_inst.for_range(pc_kernel_base, pc_kernel_base + pc_kernel_repeat) as lk:
                with tik_inst.for_range(pc_src_c_base, pc_src_c_base + pc_src_c_repeat) as lsc:
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
    instance.compute(tp, data_in, data_out)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[data_in], outputs=[data_out])
