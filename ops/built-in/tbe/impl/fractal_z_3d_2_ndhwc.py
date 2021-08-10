# Copyright 2019 Huawei Technologies Co., Ltd
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
fractal_z_3d_2_ndhwc
"""
from functools import reduce as functools_reduce
from te import platform as cce
import te.platform.cce_params as cce_params
from te import tik
from te.utils import para_check
from impl import trans_data_negative_target_tc

# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
# available number of cores
AICORE_NUM = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)


# pylint: disable=locally-disabled,too-many-lines,too-many-locals
# pylint: disable=bad-option-value,no-else-return
def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block


def _ceil_fill(value, block):
    """
    fill the input value by block

    """
    return _ceil_div(value, block)*block


def _gm_to_ub_one(args):
    """
    move data from GM to UB one scene

    """
    tik_instance, data, data_ub, data_offset, ub_offset, ori_nburst, \
    burst_len, src_stride, dst_stride, cp_align_len = args

    if src_stride <= 65535:
        if ori_nburst <= 4095:
            tik_instance.data_move(data_ub[ub_offset],
                                   data[data_offset],
                                   0, ori_nburst,
                                   burst_len,
                                   src_stride, dst_stride)

        else:
            n_burst = 4095
            c_cycle = ori_nburst // n_burst
            c_mod = ori_nburst % n_burst
            for num_cy in range(c_cycle):
                data_cur = data_offset + (burst_len + src_stride)\
                           * cp_align_len * n_burst * num_cy
                ub_cur = ub_offset + (burst_len + dst_stride)\
                         * cp_align_len * n_burst * num_cy
                tik_instance.data_move(
                    data_ub[ub_cur],
                    data[data_cur],
                    0, n_burst,
                    burst_len,
                    src_stride, dst_stride)

            if c_mod > 0:
                data_cur = data_offset + (burst_len + src_stride)\
                           * cp_align_len * n_burst * c_cycle
                ub_cur = ub_offset + (burst_len + dst_stride)\
                         * cp_align_len * n_burst * c_cycle
                tik_instance.data_move(
                    data_ub[ub_cur],
                    data[data_cur],
                    0, c_mod,
                    burst_len,
                    src_stride, dst_stride)

    else:
        for num_nb in range(ori_nburst):
            data_cur = data_offset + (burst_len + src_stride)\
                       * cp_align_len * num_nb
            ub_cur = ub_offset + (burst_len + dst_stride)\
                     * cp_align_len * num_nb
            tik_instance.data_move(
                data_ub[ub_cur],
                data[data_cur],
                0, 1,
                burst_len,
                0, 0)


def _ub_to_gm_one(args):
    """
    function of moving data from ub to gm

    """
    tik_instance, dst, data_res, dst_offset, res_offset, ori_nburst, \
    burst_len, src_stride, dst_stride, cp_align_len = args

    if dst_stride <= 65535:
        if ori_nburst <= 4095:
            tik_instance.data_move(
                dst[dst_offset],
                data_res[res_offset],
                0, ori_nburst, burst_len,
                src_stride, dst_stride)

        else:
            n_burst = 4095
            c_cycle = ori_nburst // n_burst
            c_mod = ori_nburst % n_burst

            for num_cy in range(c_cycle):
                dst_cur = dst_offset + (burst_len + dst_stride)\
                          * cp_align_len * n_burst * num_cy
                res_cur = res_offset + (burst_len + src_stride)\
                          * cp_align_len * n_burst * num_cy

                tik_instance.data_move(
                    dst[dst_cur],
                    data_res[res_cur],
                    0, n_burst, burst_len,
                    src_stride, dst_stride)

            if c_mod > 0:
                dst_cur = dst_offset + (burst_len + dst_stride)\
                          * cp_align_len * n_burst * c_cycle
                res_cur = res_offset + (burst_len + src_stride)\
                          * cp_align_len * n_burst * c_cycle

                tik_instance.data_move(
                    dst[dst_cur],
                    data_res[res_cur],
                    0, c_mod, burst_len,
                    src_stride, dst_stride)

    else:
        for num_nb in range(ori_nburst):
            dst_cur = dst_offset + (burst_len + dst_stride)\
                      * cp_align_len * num_nb
            res_cur = res_offset + (burst_len + src_stride)\
                      * cp_align_len * num_nb

            tik_instance.data_move(
                dst[dst_cur],
                data_res[res_cur],
                0, 1, burst_len,
                0, 0)


def _set_core_num(origin_num):
    """
    function of set core num
    """
    if origin_num < AICORE_NUM:
        return origin_num
    return AICORE_NUM


def _set_loop(tik_instance, num_core, max_core, total_dim):
    """
    function of set loop
    """
    core_loop = tik_instance.Scalar("uint64")

    with tik_instance.if_scope(num_core < total_dim % AICORE_NUM):
        core_loop.set_as(_ceil_div(total_dim, max_core))
    with tik_instance.else_scope():
        core_loop.set_as(total_dim // max_core)

    return core_loop


# pylint: disable=locally-disabled,too-many-instance-attributes
# pylint: disable=locally-disabled,old-style-class,too-many-statements
class Fz3d2NdhwcCompute:
    """
    Rearranges data from FRACTAL_Z_3D format to NDHWC format

    Returns
    -------
    None
    """
    def __init__(self, src_shape, dst_shape, dtype, kernel_name):
        """
        initialize some properties
        """
        self.src_shape = list(src_shape)
        self.dst_shape = list(dst_shape)
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.float_size = cce.cce_intrin.get_bit_len(dtype) // 8
        self.cp_align_len = cce_params.BLOCK_REDUCE_INT8 // self.float_size
        self.ub_ele = ((UB_SIZE_B - 64) // self.float_size // 2
                       // self.cp_align_len) * self.cp_align_len
        self.n_o = self.src_shape[1]
        self.n_i = self.src_shape[2]
        self.c_0 = self.src_shape[3]
        self.c_1 = self.calc_c1()
        self.src_gm = None
        self.dst_gm = None

    def calc_c1(self):
        """
        function of calculating c_1
        """
        dc1hw_s = self.src_shape[0]
        _, d_d, h_d, w_d, _ = self.dst_shape
        c_1 = dc1hw_s // (d_d * h_d * w_d)
        return c_1

    def c_align_small(self, tik_instance):
        """
        n % 16 == 0
        c % 16 == 0
        DC1HW * No * Ni * C0 <= ub_ele
        """
        n_d, d_d, h_d, w_d, c_d = self.dst_shape
        hw_d = h_d * w_d
        hwnoni = hw_d * self.n_o * self.n_i
        dhw_d = d_d * h_d * w_d

        ub_ori = tik_instance.Tensor(self.dtype,
                                     (self.ub_ele,),
                                     name="ub_ori",
                                     scope=tik.scope_ubuf)
        ub_trans = tik_instance.Tensor(self.dtype,
                                       (self.ub_ele,),
                                       name="ub_trans",
                                       scope=tik.scope_ubuf)

        burst_len = d_d * self.c_1 * hwnoni * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm,
                               0, 1, burst_len, 0, 0)

        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, self.c_1) as num_c1:
                ori_cur = num_d * self.c_1 * hwnoni * self.c_0\
                          + num_c1 * hwnoni * self.c_0
                trans_cur = num_d * self.c_1 * hwnoni * self.c_0\
                            + num_c1 * self.c_0
                nburst = hwnoni
                burst_len = self.c_0 // self.cp_align_len
                src_stride = 0
                dst_stride = (self.c_1 - 1) * self.c_0 // self.cp_align_len
                tik_instance.data_move(
                    ub_trans[trans_cur],
                    ub_ori[ori_cur],
                    0, nburst, burst_len, src_stride, dst_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_cur = num_dhw * self.n_o * self.n_i * c_d
            dst_cur = num_dhw * c_d
            nburst = n_d
            burst_len = c_d // self.cp_align_len
            src_stride = 0
            dst_stride = (dhw_d - 1) * c_d // self.cp_align_len
            tik_instance.data_move(
                ub_ori[dst_cur],
                ub_trans[src_cur],
                0, nburst, burst_len, src_stride, dst_stride)

        burst_len = n_d * dhw_d * c_d // self.cp_align_len
        tik_instance.data_move(self.dst_gm,
                               ub_ori,
                               0, 1, burst_len, 0, 0)

        return tik_instance

    def func_c_align_split_n(self, args):
        """
        function of moving data for c_align_split_n scene
        """
        tik_instance, ub_ori, ub_trans, n_before, n_len = args

        n_d, d_d, h_d, w_d, c_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        hw_d = h_d * w_d

        data_offset = n_before * self.c_0
        ub_offset = 0
        ori_nburst = dhw_d * self.c_1
        burst_len = n_len * self.c_0 // self.cp_align_len
        src_stride = (n_d - n_len) * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset, \
               ori_nburst, burst_len, src_stride, dst_stride, self.cp_align_len
        _gm_to_ub_one(args)

        hwnoni = hw_d * n_len
        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, self.c_1) as num_c1:
                ori_cur = num_d * self.c_1 * hwnoni * self.c_0 \
                          + num_c1 * hwnoni * self.c_0
                trans_cur = num_d * self.c_1 * hwnoni * self.c_0 \
                            + num_c1 * self.c_0
                nburst = hwnoni
                burst_len = self.c_0 // self.cp_align_len
                src_stride = 0
                dst_stride = (self.c_1 - 1) * self.c_0 // self.cp_align_len
                tik_instance.data_move(
                    ub_trans[trans_cur],
                    ub_ori[ori_cur],
                    0, nburst, burst_len, src_stride, dst_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_cur = num_dhw * n_len * c_d
            dst_cur = num_dhw * c_d
            nburst = n_len
            burst_len = c_d // self.cp_align_len
            src_stride = 0
            dst_stride = (dhw_d - 1) * c_d // self.cp_align_len
            tik_instance.data_move(
                ub_ori[dst_cur],
                ub_trans[src_cur],
                0, nburst, burst_len, src_stride, dst_stride)

        dst_offset = n_before * dhw_d * c_d
        burst_len = n_len * dhw_d * c_d // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_ori,
                               0, 1, burst_len, 0, 0)

    def c_align_split_n(self, tik_instance):
        """
        n % 16 == 0
        c % 16 == 0
        1 <= ub_ele // (dhw_d * c_align) < n_d
        """
        n_d, d_d, h_d, w_d, _ = self.dst_shape
        dhw_d = d_d * h_d * w_d
        nc_one = self.ub_ele // dhw_d
        c_align = self.c_1 * self.c_0
        n_ub = nc_one // c_align

        all_core = _ceil_div(n_d, n_ub)
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core

                with tik_instance.if_scope(core_index < all_core - 1):
                    n_len = n_ub
                    n_before = n_ub * core_index
                    args = tik_instance, ub_ori, ub_trans, n_before, n_len
                    self.func_c_align_split_n(args)

                with tik_instance.else_scope():
                    n_before = (all_core - 1) * n_ub
                    n_len = n_d - n_before
                    args = tik_instance, ub_ori, ub_trans, n_before, n_len
                    self.func_c_align_split_n(args)

        return tik_instance

    def c_not_align_small_fp16(self, tik_instance):
        """
        dtype == "float16"
        n % 16 == 0
        c % 16 > 0
        n*d*h*w*c*cp_align_len <= ub_ele
        """
        n_d, d_d, h_d, w_d, c_d = self.dst_shape
        hw_d = h_d * w_d
        hwnoni = hw_d * self.n_o * self.n_i
        dhw_d = d_d * h_d * w_d

        ub_ori = tik_instance.Tensor(self.dtype,
                                     (self.ub_ele,),
                                     name="ub_ori",
                                     scope=tik.scope_ubuf)
        ub_trans = tik_instance.Tensor(self.dtype,
                                       (self.ub_ele,),
                                       name="ub_trans",
                                       scope=tik.scope_ubuf)

        all_ele = d_d * self.c_1 * hwnoni * self.c_0
        burst_len = _ceil_div(all_ele, self.cp_align_len)
        tik_instance.data_move(ub_ori,
                               self.src_gm,
                               0, 1, burst_len, 0, 0)

        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, hw_d) as num_hw:
                with tik_instance.for_range(0, n_d) as num_n:
                    ori_begin = num_d * self.c_1 * hwnoni * self.c_0 \
                                + num_hw * self.n_o * self.n_i * self.c_0\
                                + num_n * self.c_0
                    trans_begin = num_d * hw_d * n_d \
                                  * c_d * self.cp_align_len \
                                  + num_hw * n_d * c_d * self.cp_align_len \
                                  + num_n * c_d * self.cp_align_len
                    src_list = [ub_ori[ori_begin + 16 * i]
                                for i in range(16)]
                    dst_list = [ub_trans[trans_begin + 16 * i]
                                for i in range(16)]
                    repeat = self.c_1

                    if repeat == 1:
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list, repeat, 0, 0)
                    else:
                        src_rep_stride = hwnoni * self.c_0 // self.cp_align_len
                        dst_rep_stride = 16
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list,
                                               repeat,
                                               dst_rep_stride,
                                               src_rep_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_offset = num_dhw * n_d * c_d * self.cp_align_len
            dst_offset = num_dhw * c_d * self.cp_align_len
            n_burst = n_d
            burst_len = c_d
            src_stride = 0
            dst_stride = (dhw_d - 1) * c_d
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        out_ele = n_d * d_d * h_d * w_d * c_d
        ele_zu = _ceil_div(out_ele, 16)
        repeat = ele_zu

        if repeat == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, repeat, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   repeat,
                                   dst_rep_stride,
                                   src_rep_stride)

        burst_len = _ceil_div(out_ele, self.cp_align_len)
        tik_instance.data_move(self.dst_gm,
                               ub_trans,
                               0, 1, burst_len, 0, 0)

        return tik_instance

    def func_c_not_align_split_n_fp32(self, args):
        """
        function of moving data for c_not_align_split_n_fp32 scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail, \
        n_before, n_len = args

        _, d_d, h_d, w_d, c_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        hw_d = h_d * w_d
        hwn_len = hw_d * n_len

        data_offset = n_before * self.c_0
        ub_offset = 0
        ori_nburst = dhw_d * self.c_1
        burst_len = n_len * self.c_0 // self.cp_align_len
        src_stride = (self.n_o * self.n_i - n_len)\
                     * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset, \
               ori_nburst, burst_len, src_stride, dst_stride, self.cp_align_len
        _gm_to_ub_one(args)

        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, hw_d) as num_hw:
                with tik_instance.for_range(0, n_len) as num_n:
                    for num_c0zu in range(2):
                        ori_begin = (num_d * self.c_1 * hwn_len * self.c_0
                                     + num_hw * n_len * self.c_0
                                     + num_n * self.c_0 + num_c0zu * 8) * 2
                        trans_begin = (num_d * hw_d * n_len
                                       * c_d * 2 * self.cp_align_len
                                       + num_hw * n_len * c_d
                                       * 2 * self.cp_align_len
                                       + num_n * c_d * 2 * self.cp_align_len
                                       + num_c0zu * 16 * self.cp_align_len) * 2
                        src_list = [ub_ori[ori_begin + 16 * i]
                                    for i in range(16)]
                        dst_list = [ub_trans[trans_begin + 16 * i]
                                    for i in range(16)]
                        repeat = self.c_1

                        if repeat == 1:
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list, repeat, 0, 0)
                        else:
                            src_rep_stride = hwn_len * self.c_0\
                                             // self.cp_align_len
                            dst_rep_stride = 32
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list,
                                                   repeat,
                                                   dst_rep_stride,
                                                   src_rep_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_offset = num_dhw * n_len * c_d * 2 * self.cp_align_len * 2
            dst_offset = num_dhw * c_d * 2 * self.cp_align_len * 2
            n_burst = n_len
            burst_len = c_d * 2
            src_stride = 0
            dst_stride = (dhw_d - 1) * c_d * 2
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        out_ele = n_len * d_d * h_d * w_d * c_d
        out_ele_2 = out_ele * 2
        ele_zu = _ceil_div(out_ele_2, 16)
        repeat = ele_zu

        if repeat == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, repeat, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   repeat,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = n_before * dhw_d * c_d
        if out_ele % self.cp_align_len > 0:
            sub_ele = out_ele - self.cp_align_len
            burst_len = _ceil_div(sub_ele, self.cp_align_len)
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)
            for k in range(16):
                ub_tail[k] = ub_trans[sub_ele * 2 + k]

            tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                   ub_tail,
                                   0, 1, 1, 0, 0)
        else:
            burst_len = out_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def c_not_align_split_n_fp32(self, tik_instance):
        """
        dtype == "float32"
        n % 16 == 0
        c % 16 > 0
        1 <= ub_ele // (dhw_d * c_d * 16) < n_d
        """
        n_d, d_d, h_d, w_d, c_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        nc_one = self.ub_ele // dhw_d
        n_ub = nc_one // 2 // self.cp_align_len // c_d

        all_core = _ceil_div(n_d, n_ub)
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor("float16",
                                         (self.ub_ele * 2,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (self.ub_ele * 2,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_tail = tik_instance.Tensor("float16",
                                          (16,),
                                          name="ub_tail",
                                          scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core

                with tik_instance.if_scope(core_index < all_core - 1):
                    n_len = n_ub
                    n_before = n_ub * core_index
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           n_before, n_len
                    self.func_c_not_align_split_n_fp32(args)

                with tik_instance.else_scope():
                    n_before = (all_core - 1) * n_ub
                    n_len = n_d - n_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           n_before, n_len
                    self.func_c_not_align_split_n_fp32(args)

        return tik_instance

    def check_branch(self):
        """
        check which branch of fz3d_2_ndhwc compute
        """
        n_d, d_d, h_d, w_d, c_d = self.dst_shape
        all_ele = d_d * self.c_1 * h_d * w_d * self.n_o * self.n_i * self.c_0
        out_ele = n_d * d_d * h_d * w_d * c_d
        dhw_d = d_d * h_d * w_d
        c_align = self.c_1 * self.c_0

        if n_d % 16 == 0:
            if c_d % self.c_0 == 0 and all_ele <= self.ub_ele:
                return "c_align_small"
            elif c_d % self.c_0 == 0\
                    and 1 <= self.ub_ele // (dhw_d * c_align) < n_d:
                return "c_align_split_n"
            elif c_d % self.c_0 > 0 and self.dtype == "float16"\
                    and out_ele * self.cp_align_len <= self.ub_ele:
                return "c_not_align_small_fp16"
            elif c_d % self.c_0 > 0 and self.dtype == "float32"\
                    and 1 <= self.ub_ele // (dhw_d * c_d * 16) < n_d:
                return "c_not_align_split_n_fp32"
            else:
                return "not_support"
        else:
            return "not_support"

    def fz3d_2_ndhwc_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        branch = self.check_branch()

        if branch == "c_align_small":
            tik_instance = self.c_align_small(tik_instance)
        elif branch == "c_align_split_n":
            tik_instance = self.c_align_split_n(tik_instance)
        elif branch == "c_not_align_small_fp16":
            tik_instance = self.c_not_align_small_fp16(tik_instance)
        elif branch == "c_not_align_split_n_fp32":
            tik_instance = self.c_not_align_split_n_fp32(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        src_element_number = functools_reduce(lambda x, y: x * y,
                                              self.src_shape[:])
        dst_element_number = functools_reduce(lambda x, y: x * y,
                                              self.dst_shape[:])
        self.src_gm = tik_instance.Tensor(self.dtype,
                                          (src_element_number,),
                                          name="src_gm",
                                          scope=tik.scope_gm)
        self.dst_gm = tik_instance.Tensor(self.dtype,
                                          (dst_element_number,),
                                          name="dst_gm",
                                          scope=tik.scope_gm)

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.fz3d_2_ndhwc_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


def _check_parameters(src, dst, src_format, dst_format):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "fractal_z_3d":
        raise RuntimeError("src_format must be FRACTAL_Z_3D !")

    if dst_format.lower() != "ndhwc":
        raise RuntimeError("dst_format must be NDHWC!")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    para_check.check_shape(src_shape, min_rank=4, max_rank=4)
    para_check.check_shape(dst_shape, min_rank=5, max_rank=5)

    if src_shape[2] != 16:
        raise RuntimeError(
            "the 3rd dimension of src_shape is not 16, Ni must be 16 !")

    if src_shape[3] != 16:
        raise RuntimeError(
            "the 4th dimension of src_shape is not 16, C0 must be 16 !")

    n_d, d_d, h_d, w_d, c_d = dst_shape

    n_i = 16
    n_s = n_i - 1
    n_o = (n_d + n_s) // n_i

    if src_shape[1] != n_o:
        raise RuntimeError(
            "the 2nd dimension of src_shape is wrong, "
            "No must be (N + 15)//16 !")

    c_0 = 16
    c_s = c_0 - 1
    c_1 = (c_d + c_s) // c_0
    one_dim = d_d * c_1 * h_d * w_d

    if src_shape[0] != one_dim:
        raise RuntimeError(
            "the 1st dimension of src_shape is wrong, "
            "it must be D*C1*H*W !")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def fractal_z_3d_2_ndhwc(src, dst, src_format, dst_format,
                         kernel_name="fractal_z_3d_2_ndhwc"):
    """
    algorithm: fractal_z_3d_2_ndhwc
    calculating: change data format from FRACTAL_Z_3D to NDHWC

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src, only support "FRACTAL_Z_3D"
    dst_format: str
        data format of dst, only support "NDHWC"
    kernel_name: str
        kernel name, default value is "fractal_z_3d_2_ndhwc"

    Returns
    -------
    tik_instance: tik_instance
    """
    _check_parameters(src, dst, src_format, dst_format)
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype").lower()

    template_fp16 = Fz3d2NdhwcCompute(src_shape, dst_shape, dtype, kernel_name)
    if template_fp16.check_branch() != "not_support":
        return template_fp16.get_tik_instance()
    else:
        trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
