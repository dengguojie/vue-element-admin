# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
conv2d backprop filter tiling case
"""

from collections import OrderedDict
from functools import reduce

from te import tvm
from te.lang.cce.te_compute.conv2d_backprop_filter_compute import \
    DynamicConv2dBpFilterParams as DynamicParams

from tbe.common.tiling.get_tiling import get_tiling
from te.lang.base.operation_impl import register_tiling_case
from te.lang.base.operation_impl import get_te_var

from .cube_tilingcase import TilingSelection
from .cube_tilingcase import CubeTilingOp
from .cube_tilingcase import TilingUtils as utils
from . import Pattern

H_RANGE = 4096
W_RANGE = 4096
N_RANGE = 1000000
W_DELTA = 1
H_LEN = 400
W_LEN = 400


@register_tiling_case(pattern=Pattern.CONV2D_BACKPROP_FILTER)
def calc_conv2dbp_filter(outs, option=None):
    mode = DynamicParams.dynamic_mode
    var_names = {'dynamic_batch': ('batch', ),
                 'dynamic_hw': ('fmap_h', 'fmap_w')}
    tgt_area = [tuple(get_te_var(v).get_bound()) for v in var_names[mode]]
    info = DynamicParams.tiling_info_dict

    tiling_op = Conv2dBpFilterTiling(info, mode)

    tiling_cases = TilingSelection(tiling_op).calc_tiling(tgt_area)
    return tiling_cases


class Conv2dBpFilterTiling(CubeTilingOp):
    def __init__(self, tiling_info, dynamic_mode):
        super().__init__(tiling_info, dynamic_mode)
        self.a_info = self.tiling_info['A_shape']
        self.b_info = self.tiling_info['B_shape']
        self.c_info = self.tiling_info['C_shape']
        self._get_calc_info()
        self.key = 'B_shape'
        self.op_type = 'conv2d_bp_filter'

    def get_repo_tiling(self):
        tiling_list = get_tiling(self.tiling_info)
        res_list = []
        for tiling in tiling_list:
            self._set_padding_list(tiling['B_shape'][2],
                                   tiling['B_shape'][3])
            if tiling['pad'] == self.cur_pads:
                self._get_attach_flag(tiling)
                res_list.append(tiling)
        return res_list

    def get_costmodel_tiling(self, shape):
        """
        get tiling using cost model

        Parameters
        ----------
        shape: specified shape to get tiling

        Returns
        -------
        tiling: tiling retrieved by cost model
        """

        if self.dynamic_mode == "dynamic_batch":
            self.a_info[0] = shape
            self.b_info[0] = shape
        elif self.dynamic_mode == "dynamic_hw":
            self.b_info[2], self.b_info[3] = shape[0], shape[1]
            self._set_padding_list(self.b_info[2], self.b_info[3])
            self.tiling_info['padl'] = self.cur_pads[0]
            self.tiling_info['padr'] = self.cur_pads[1]
            self.tiling_info['padu'] = self.cur_pads[2]
            self.tiling_info['padd'] = self.cur_pads[3]
            self.a_info[2] = self._get_output_h(self.b_info[2])
            self.a_info[3] = self._get_output_w(self.b_info[3])
        self.tiling_info["tiling_type"] = "cost_model_tiling"

        cost_tiling = get_tiling(self.tiling_info)
        if cost_tiling:
            tiling = cost_tiling[0]
            self._get_attach_flag(tiling)
        else:
            tiling = self._get_default_tiling()
            self._get_attach_flag(tiling)
        return tiling

    def get_tiling_range(self, tiling, fmap_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        fmap_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """

        _, _, h_i, w_i, _ = fmap_shape

        if not tiling["BL1_shape"]:
            # fully load in BL1, covering lower region
            return [1, h_i, 1, w_i]

        self._set_padding_list(h_i, w_i)

        # get min value
        hi_min = max(self.k_h - self.cur_pads[2] - self.cur_pads[3], 1)
        wi_min = max(self.k_w - self.cur_pads[0] - self.cur_pads[1], 1)

        hi_max = h_i
        cur_w_size = w_i

        # check tiling covering itself situation
        if not self._check_tiling_match(tiling, cur_w_size, h_i) \
            or h_i > H_RANGE or w_i > W_RANGE:
            return [0, 0, 0, 0]

        # searching down-ward for w_min
        while self._check_tiling_match(tiling, cur_w_size, h_i) and \
            cur_w_size > max(self.k_w - self.cur_pads[0] - self.cur_pads[1], 1):
            wi_min = cur_w_size
            cur_w_size = cur_w_size - W_DELTA

        # searching up-ward for w_max
        cur_w_size = w_i
        while self._check_tiling_match(tiling, cur_w_size, h_i) \
            and cur_w_size <= W_RANGE:
            wi_max = cur_w_size
            cur_w_size = cur_w_size + W_DELTA

        perf_wi_min = max(wi_min, w_i - W_LEN)
        perf_wi_max = min(wi_max, w_i + W_LEN)

        # searching down-ward for h_min based on w_min
        perf_hi_min = max(hi_min, h_i - H_LEN)
        cur_h_size = h_i
        while self._check_tiling_match(tiling, perf_wi_min, cur_h_size) \
            and cur_h_size > \
            max(self.k_h - self.cur_pads[2] - self.cur_pads[3], 1):
            hi_min = cur_h_size
            cur_h_size = cur_h_size - W_DELTA
        perf_hi_min = max(hi_min, h_i - H_LEN)

        # searching up-ward for h_max based on w_max
        cur_h_size = h_i
        while self._check_tiling_match(tiling, perf_wi_max, cur_h_size) \
                and cur_h_size <= H_RANGE:
            hi_max = cur_h_size
            cur_h_size = cur_h_size + W_DELTA
        perf_hi_max = min(hi_max, h_i + H_LEN)

        if perf_wi_min > perf_wi_max:
            return [0, 0, 0, 0]

        perf_range = [perf_hi_min, perf_hi_max, perf_wi_min, perf_wi_max]
        perf_range = [int(v) for v in perf_range]
        return perf_range

    def get_batch_range(self, tiling, fmap_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        fmap_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """

        n_i, _, h_i, w_i, _ = fmap_shape
        self._set_padding_list(h_i, w_i)
        h_o = self._get_output_h(h_i)
        w_o = self._get_output_w(w_i)
        dy_shape = n_i, self.a_info[1], h_o, w_o, utils.CUBE_SIZE
        block_dim_batch = tiling.get("block_dim")[0]

        ni_min = 1
        ni_max = N_RANGE

        full_k_in_l0a, full_k_in_l0b, grads_l1_tiling_nparts, \
            fmap_l1_tiling_nparts = self._check_full_k(tiling, dy_shape)
        batch_num_sc = utils.icd(n_i, block_dim_batch)

        # based on l0_attach and l1_attach
        if (full_k_in_l0a <= 0 and full_k_in_l0b <= 0) \
            and (grads_l1_tiling_nparts[0] != 1 \
            and fmap_l1_tiling_nparts[0] != 1):
            # batch wont influence attach flag
            return [ni_min, ni_max]
        else:
            # attach flag with different batch situation
            if batch_num_sc == 1:
                return [ni_min, block_dim_batch]
            else:
                return [block_dim_batch+1, ni_max]

    def assembly_case(self, tiling, coverage, cnt):
        var_range = OrderedDict()
        if self.dynamic_mode == "dynamic_hw":
            x_h_low, x_h_high = int(coverage[0]), int(coverage[1])
            x_w_low, x_w_high = int(coverage[2]), int(coverage[3])
            self._set_padding_list(x_h_low, x_w_low)
            dedy_h_low = self._get_output_h(x_h_low)
            dedy_w_low = self._get_output_w(x_w_low)
            self._set_padding_list(x_h_high, x_w_high)
            dedy_h_high = self._get_output_h(x_h_high)
            dedy_w_high = self._get_output_w(x_w_high)

            var_range['fmap_h'] = (x_h_low, x_h_high)
            var_range['fmap_w'] = (x_w_low, x_w_high)
            var_range['dedy_h'] = (dedy_h_low, dedy_h_high)
            var_range['dedy_w'] = (dedy_w_low, dedy_w_high)
        elif self.dynamic_mode == "dynamic_batch":
            var_range['batch'] = (int(coverage[0]), int(coverage[1]))

        block_dim_multi = tiling["AUB_shape"][0] \
            if tiling["AUB_shape"] else 1
        block_dims = block_dim_multi * reduce(
            lambda x, y: x * y, tiling['block_dim'])

        return {"key": cnt, "tiling_strategy": tiling,
                "var_range": var_range, "block_dim": block_dims}

    def _get_default_tiling(self):
        return {
            'AUB_shape': None, 'BUB_shape': None,
            'AL1_shape': [utils.CUBE_SIZE, 1, 1],
            'BL1_shape': [utils.CUBE_SIZE, 1, 1],
            'AL0_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'BL0_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'CL0_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'CUB_matrix': [1, 1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1],
            'block_dim': [1, 1, 1],
            'cout_bef_batch_flag': 0,
            'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0,
            'manual_pingpong_buffer': {
                'AUB_pbuffer': 1, 'BUB_pbuffer': 1,
                'AL1_pbuffer': 1, 'BL1_pbuffer': 1,
                'AL0_pbuffer': 1, 'BL0_pbuffer': 1,
                'CL0_pbuffer': 1, 'CUB_pbuffer': 1,
                'UBG_pbuffer': 1}
        }

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.c_info[2:4]
        self.k_cin = self.c_info[1] * self.c_info[4]
        self.k_cout = self.c_info[0]
        self.stride_h, self.stride_w = self.tiling_info["strideH"], \
            self.tiling_info["strideW"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilationH"], \
            self.tiling_info["dilationW"]

        if isinstance(self.tiling_info["padl"], tvm.expr.Expr):
            self.pad_mode = "SAME"
            self.cur_pads = [-1, -1, -1, -1]
            for pad in ("padl", "padr", "padu", "padd"):
                self.tiling_info[pad] = -1
        else:
            self.pad_mode = "FIX"
            self.cur_pads = [
                self.tiling_info["padl"], self.tiling_info["padr"],
                self.tiling_info["padu"], self.tiling_info["padd"]
            ]

        self.k_h_dilation = (self.k_h - 1) * self.dilate_h + 1
        self.k_w_dilation = (self.k_w - 1) * self.dilate_w + 1

    def _set_padding_list(self, cur_h, cur_w):
        """
        get padding list in cur dx shape
        """

        if self.pad_mode == "SAME":
            pad_h = max(utils.align(cur_h, self.stride_h) -
                        self.stride_h + self.k_h_dilation - cur_h, 0)
            pad_up = pad_h // 2
            pad_down = pad_h - pad_up
            pad_w = max(utils.align(cur_w, self.stride_w) -
                        self.stride_w + self.k_w_dilation - cur_w, 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            self.cur_pads = [pad_left, pad_right, pad_up, pad_down]

    def _get_bound_fmap(self, tiling,
                        width_grads, width_fmap,
                        local_tiling_flag, height_grads):
        """
        get bound info for _get_bound_fmap

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        width_grads, width_fmap : int, size of w

        Returns
        -------
        int, load_length

        """

        # bl1 set storage bound
        # actual load length in k_reduce_axis
        bl1_k = tiling.get("BL1_shape")[0]
        block_dim_hw = tiling.get("AUB_shape")[0] \
            if tiling.get("AUB_shape") else 1
        hw_pad_1 = utils.icd(width_grads * height_grads, utils.CUBE_SIZE)
        flag_fmap_load2d = local_tiling_flag[-2]
        flag_conv1d_case = local_tiling_flag[-3]

        # load2d instructions refer to data_mov with raw lens
        if flag_fmap_load2d:
            return bl1_k

        if tiling["AL0_matrix"]:
            # dw_k equals to ka if L0A needs tiling
            dw_k = tiling["AL0_matrix"][1]
        elif tiling["BL0_matrix"]:
            dw_k = tiling["BL0_matrix"][0]
        else:
            # both fully loaded
            dw_k = hw_pad_1 // block_dim_hw

        hw_single_core_factor = utils.icd(hw_pad_1, block_dim_hw) * \
                                utils.CUBE_SIZE
        hw_single_core_factor = utils.align(hw_single_core_factor,
                                            dw_k * utils.CUBE_SIZE)

        if bl1_k < width_grads:
            # tiling load lens less then width_grads, need to load a full line
            if flag_conv1d_case:
                return tiling["BL1_shape"][0]
            else:
                # if res_data exists then need to load 2 lines
                ho_len = 1 if (width_grads % bl1_k == 0 and \
                               hw_single_core_factor % width_grads == 0) else 2
        else:
            # load3d instructions refer to load extra lines with pad/stride/filter
            if bl1_k % width_grads == 0 and \
                    hw_single_core_factor % width_grads == 0:
                # full line could load without extra lines
                additional_rows = 0
            elif bl1_k * 2 % width_grads == 0 or \
                bl1_k % width_grads == 1:
                # every 2 load3d covered only 1 extra line
                additional_rows = 1
            else:
                # other situations need 2 extra lines in case
                additional_rows = 2
            ho_len = bl1_k // width_grads + additional_rows

        if flag_conv1d_case:
            bl1_hi = 1
            kbl1_data = utils.icd(hw_pad_1 * utils.CUBE_SIZE, block_dim_hw)    
            if tiling.get("BL1_shape"):
                kbl1_data = tiling["BL1_shape"][0]
            bl1_wi = (kbl1_data - 1) * self.stride_w + self.k_w
            bl1_k_full = bl1_hi * bl1_wi
        else:
            hi_max = self.k_h + (ho_len - 1) * self.stride_h
            bl1_k_full = width_fmap * hi_max

        return bl1_k_full

    def _check_tiling_match(self, tiling, current_w, current_h):
        """

        check whether this tiling matches the shape

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of h,w

        Returns
        -------
        bool, True: match
            False: do not match

        """

        # shape info
        block_dim_hw = tiling.get("AUB_shape")[0] \
            if tiling.get("AUB_shape") else 1

        w_i, h_i = current_w, current_h
        self._set_padding_list(h_i, w_i)
        h_o = self._get_output_h(h_i)
        w_o = self._get_output_w(w_i)
        howo_align = utils.align(h_o * w_o, utils.FP16_K)

        dy_shape = self.a_info[0], self.a_info[1], h_o, w_o, utils.CUBE_SIZE
        fmap_shape = self.b_info[0], self.b_info[1], h_i, w_i, utils.CUBE_SIZE

        # flag check
        local_tiling_flag = self._get_attach_flag_detail(tiling,
                                                         dy_shape, fmap_shape)
        seed_tiling_flag = (tiling["dynamic_l0a_attach"],
                            tiling["dynamic_l0b_attach"],
                            tiling["dynamic_al1_attach"],
                            tiling["dynamic_bl1_attach"],
                            tiling["bl1_hw_allin_flag"],
                            tiling["batch_num_sc"],
                            tiling["flag_conv1d_case"],
                            tiling["flag_fmap_load2d"],
                            tiling["k_atomic_add_len"],)

        for index, flag in enumerate(seed_tiling_flag[:-2]):
            if flag == "dw_cc":
                continue
            elif flag != local_tiling_flag[index]:
                return False

        # align tiling["BL1_shape"] for k_h * k_w
        tiling["BL1_shape"][1] = utils.align(
            tiling["BL1_shape"][1] * tiling["BL0_matrix"][1],
            self.k_h * self.k_w) // tiling["BL0_matrix"][1]

        # get K axis length in al1
        bl1_bound = self._get_bound_fmap(tiling,
                                         w_o, w_i, local_tiling_flag, h_o)

        # fmap size in L1 ( K * N * db * 2byte)
        fmap_l1_size = utils.FP16_SIZE * bl1_bound * \
                       tiling['BL1_shape'][1] * \
                       tiling["BL0_matrix"][1] // (self.k_h * self.k_w) * \
                       utils.FP16_N * \
                       tiling['manual_pingpong_buffer']['BL1_pbuffer']

        # grad size
        if tiling["AL1_shape"]:
            # tiling size in L1 ( M * K * db * 2byte)
            al1_m = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * \
                    utils.FP16_M
            grad_l1_size = utils.FP16_SIZE * tiling["AL1_shape"][0] * \
                           al1_m * \
                           tiling['manual_pingpong_buffer']['AL1_pbuffer']
        else:
            # fully load in AL1
            al1_m = self.k_cout
            grad_l1_size = utils.FP16_SIZE * howo_align * al1_m

        return int(fmap_l1_size + grad_l1_size) <= utils.L1BUFFER


    def _check_batch_flag(self, tiling, current_n, seed_fmap_shape):
        """

        check whether this tiling matches the shape

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of n

        Returns
        -------
        bool, True: match
            False: do not match

        """

        # shape info
        _, _, w_i, h_i, _ = seed_fmap_shape
        self._set_padding_list(h_i, w_i)
        h_o = self._get_output_h(h_i)
        w_o = self._get_output_w(w_i)

        cur_dy_shape = current_n, self.a_info[1], h_o, w_o, utils.CUBE_SIZE
        cur_fmap_shape = current_n, self.b_info[1], h_i, w_i, utils.CUBE_SIZE

        # flag check
        local_tiling_flag = \
            self._get_attach_flag_detail(tiling, cur_dy_shape, cur_fmap_shape)
        seed_tiling_flag = (tiling["dynamic_l0a_attach"],
                            tiling["dynamic_l0b_attach"],
                            tiling["dynamic_al1_attach"],
                            tiling["dynamic_bl1_attach"],
                            tiling["bl1_hw_allin_flag"],
                            tiling["batch_num_sc"],
                            tiling["flag_conv1d_case"],
                            tiling["flag_fmap_load2d"],
                            tiling["k_atomic_add_len"],)

        return seed_tiling_flag[:-3] == local_tiling_flag[:-3]

    def _get_output_h(self, h_i):
        return (h_i + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h *
                (self.k_h - 1) - 1) // self.stride_h + 1

    def _get_output_w(self, w_i):
        return (w_i + self.cur_pads[0] + self.cur_pads[1] - self.dilate_w *
                (self.k_w - 1) - 1) // self.stride_w + 1

    def _check_full_k(self, tiling, dy_shape):
        """
        set flag whether axis K is fully loaded in L0A and L0B
        return:
        -------
        full_k_l0a: 1 or 0, 1 means K is fully loaded in L0A
        full_k_l0b: 1 or 0, 1 means K is fully loaded in L0B
        """

        # if k is fully load in BL1 and
        # there is multi load in N1 and N1 in BL1
        # isn't aligned to kernel_height*kernel_width, then align to it
        _, _, grads_height, grads_width, _ = dy_shape
        hw_mad_1 = utils.icd(grads_height * grads_width, utils.FP16_K)

        fmap_channel_1 = utils.icd(self.k_cin, utils.CUBE_SIZE)
        fkk = fmap_channel_1 * self.k_h * self.k_w
        c1_grads = utils.icd(self.k_cout, utils.CUBE_SIZE)
        block_dim_hw = tiling.get("AUB_shape")[0] \
            if tiling.get("AUB_shape") else 1

        block_dim_cout = tiling.get("block_dim")[2]
        block_dim_cin = tiling.get("block_dim")[1]

        if tiling.get("BL1_shape"):
            tiling["BL1_shape"][1] = utils.align(
                tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1],
                self.k_h * self.k_w) // tiling.get("BL0_matrix")[1]

        # whether axis K is fully loaded in L0A and L0B
        # excluding axis batch
        full_k_l0a = 1 \
            if not tiling["AL0_matrix"] \
            else tiling["AL0_matrix"][1] // utils.icd(hw_mad_1, block_dim_hw)
        full_k_l0b = 1 \
            if not tiling["BL0_matrix"] \
            else tiling["BL0_matrix"][0] // utils.icd(hw_mad_1, block_dim_hw)

        dw_tiling_factor = [tiling["CL0_matrix"][0], tiling["CL0_matrix"][1]]
        dw_tiling_nparts = \
            [utils.icd(fkk // block_dim_cin, dw_tiling_factor[0]),
             utils.icd(utils.icd(c1_grads, dw_tiling_factor[1]), block_dim_cout)]

        if tiling["AL1_shape"]:  # if grads needs tiling in L1
            if len(tiling["AL1_shape"]) == 1:  # but no C_1 tiling info
                tiling["AL1_shape"] = tiling["AL1_shape"] + [1]
            # nparts K1 in L1, nparts M1 in L1
            grads_l1_tiling_nparts = [
                utils.icd(hw_mad_1,
                          (block_dim_hw *
                           (tiling["AL1_shape"][0] // utils.CUBE_SIZE))),
                dw_tiling_nparts[1] // tiling["AL1_shape"][1]]
        else:
            grads_l1_tiling_nparts = [1, 1]

        if tiling["BL1_shape"]:  # if fmap needs tiling in L1
            if len(tiling["BL1_shape"]) == 1:  # but no fkk tiling info
                tiling["BL1_shape"] = \
                    tiling["BL1_shape"] + [1]  # tiling fkk=1
            # DDR to L1 [nparts K1, nparts N1]
            fmap_l1_tiling_nparts = [
                utils.icd(hw_mad_1,
                          (block_dim_hw *
                           (tiling["BL1_shape"][0] // utils.CUBE_SIZE))),
                dw_tiling_nparts[0] // tiling["BL1_shape"][1]]
        else:
            fmap_l1_tiling_nparts = [1, 1]

        return full_k_l0a, full_k_l0b, \
               grads_l1_tiling_nparts, fmap_l1_tiling_nparts

    def _get_attach_flag(self, tiling_extend):
        """
        tiling_extend: tiling with "A_shape", "B_shape", "C_shape"
        """

        tiling = tiling_extend["tiling"]
        dy_shape = tiling_extend["A_shape"]
        fmap_shape = tiling_extend["B_shape"]

        l0a_attach, l0b_attach, al1_attach, bl1_attach, \
            bl1_hw_allin_flag, batch_num_sc, \
            flag_conv1d_case, flag_fmap_load2d, \
            k_atomic_add_len = \
            self._get_attach_flag_detail(tiling, dy_shape, fmap_shape)

        tiling.update({
            "dynamic_l0a_attach": l0a_attach,
            "dynamic_l0b_attach": l0b_attach,
            "dynamic_al1_attach": al1_attach,
            "dynamic_bl1_attach": bl1_attach,
            "bl1_hw_allin_flag": bl1_hw_allin_flag,
            "batch_num_sc": batch_num_sc,
            "flag_conv1d_case": flag_conv1d_case,
            "flag_fmap_load2d": flag_fmap_load2d,
            "k_atomic_add_len": k_atomic_add_len})

    def _get_attach_flag_detail(self, tiling, dy_shape, fmap_shape):
        l0a_attach = None
        l0b_attach = None
        al1_attach = None
        bl1_attach = None
        bl1_hw_allin_flag = False
        flag_conv1d_case = False
        flag_fmap_load2d = False
        k_atomic_add_len = -1

        batch = dy_shape[0]
        block_dim_batch = tiling.get("block_dim")[0]
        block_dim_hw = tiling.get("AUB_shape")[0] \
            if tiling.get("AUB_shape") else 1
        batch_num_sc = utils.icd(batch, block_dim_batch)

        height_all_one = self.stride_h == 1 \
            and dy_shape[2] == 1 and fmap_shape[2] == 1 \
            and self.k_h == 1
        width_all_one = self.stride_w == 1 \
            and dy_shape[3] == 1 and fmap_shape[3] == 1 \
            and self.k_w == 1

        # conv1d_split_w
        flag_conv1d_case = True \
            if height_all_one and not width_all_one else False

        # load2d check
        flag_fmap_load2d = True \
            if height_all_one and width_all_one else False

        full_k_in_l0a, full_k_in_l0b, grads_l1_tiling_nparts, \
            fmap_l1_tiling_nparts = self._check_full_k(tiling, dy_shape)
        l0a_attach, l0b_attach = self._get_l0_attach(
            tiling, batch_num_sc, full_k_in_l0a, full_k_in_l0b)
        al1_attach, bl1_attach = self._get_l1_attach(
            tiling, batch_num_sc, grads_l1_tiling_nparts, fmap_l1_tiling_nparts)

        bl1_hw_allin_flag = self._get_bl1_hw_allin_flag(
            tiling, fmap_l1_tiling_nparts)

        fmap_hw_align = utils.align(fmap_shape[2] * fmap_shape[3], 16)
        k_atomic_add_len = utils.align(
            utils.icd(fmap_hw_align, block_dim_hw), 16)

        return l0a_attach, l0b_attach, al1_attach, bl1_attach, \
                bl1_hw_allin_flag, batch_num_sc == 1,\
                flag_conv1d_case, flag_fmap_load2d, \
                k_atomic_add_len

    def _get_bl1_hw_allin_flag(self, tiling, fmap_l1_tiling_nparts):
        if tiling["BL1_shape"]:
            if fmap_l1_tiling_nparts[0] == 1:
                return True
        else:
            return True
        return False

    def _get_l0_attach(self, tiling, batch_num_sc, full_k_in_l0a, full_k_in_l0b):
        l0a_attach = None
        l0b_attach = None

        if tiling["AL0_matrix"]:
            l0a_attach = "dw_ddr" if batch_num_sc == 1 and full_k_in_l0a > 0 \
                else "dw_cc"

        if tiling["BL0_matrix"]:
            l0b_attach = "dw_ddr" if batch_num_sc == 1 and full_k_in_l0b > 0 \
                else "dw_cc"
        return l0a_attach, l0b_attach

    def _get_l1_attach(self, tiling, batch_num_sc, grads_l1_tiling_nparts,
                       fmap_l1_tiling_nparts):
        al1_attach = None
        bl1_attach = None

        if tiling["AL1_shape"]:
            # if axis K needs split, then attach to dw_cc, else attach to dw_ddr
            al1_attach = "dw_cc" if grads_l1_tiling_nparts[0] != 1 or \
                batch_num_sc != 1 else "dw_ddr"

        if tiling["BL1_shape"]:
            # if axis K needs split, then attach to dw_cc else attach to dw_ddr
            bl1_attach = "dw_cc" if fmap_l1_tiling_nparts[0] != 1 or \
                batch_num_sc != 1 else "dw_ddr"
        return al1_attach, bl1_attach
