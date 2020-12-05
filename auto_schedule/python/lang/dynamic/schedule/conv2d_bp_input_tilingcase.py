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
conv2d backprop input tiling case
"""
import copy
from collections import OrderedDict
from functools import reduce

from te.tvm.expr import Expr
from te.domain.tiling.get_tiling import get_tiling

from te.lang.base.operation_impl import register_tiling_case
from te.lang.base.operation_impl import get_te_var
from te.platform import get_soc_spec

from te.lang.cce.te_compute.conv2d_backprop_input_compute import DynamicConv2dBpInputParams
from te.lang.dynamic.schedule.cube_tilingcase import TilingSelection
from te.lang.dynamic.schedule.cube_tilingcase import CubeTilingOp
from te.lang.dynamic.schedule.cube_tilingcase import TilingUtils as utils
from te.lang.dynamic.schedule.constants import Pattern


H_RANGE = 4096
W_RANGE = 4096
W_DELTA = 1
H_LEN = 400
W_LEN = 400
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

@register_tiling_case(pattern=Pattern.CONV2D_BACKPROP_INPUT)
def calc_conv2dbp_input(outs, option=None):
    mode = DynamicConv2dBpInputParams.dynamic_mode
    var_names = {'dynamic_batch': ('batch_n', ), 'dynamic_hw': ('dx_h', 'dx_w')}
    tgt_area = [get_te_var(v).get_bound() for v in var_names[mode]]
    info = DynamicConv2dBpInputParams.tiling_info_dict

    tiling_op = Conv2dBpInputTiling(info, mode)

    tiling_cases = TilingSelection(tiling_op).calc_tiling(tgt_area)
    return tiling_cases


class Conv2dBpInputTiling(CubeTilingOp):
    def __init__(self, tiling_info, dynamic_mode):
        super().__init__(tiling_info, dynamic_mode)
        self.a_info = self.tiling_info['A_shape']
        self.b_info = self.tiling_info['B_shape']
        self.c_info = self.tiling_info['C_shape']
        self.a_type = self.tiling_info["A_dtype"]
        self.c_type = self.tiling_info["C_dtype"]

        self._get_calc_info()
        self.key = 'C_shape'
        self.op_type = "conv2d_bp_input"

    def _modify_repo_tiling(self, tiling_mess):
        tiling = tiling_mess.get("tiling")
        n_size = tiling_mess.get("B_shape")[1]
        block_dim = tiling.get("block_dim")
        nc = tiling.get("CL0_matrix")[0]
        n_factor = utils.icd(n_size // block_dim[1], nc)
        block_dim[1] = n_size // nc // n_factor

    def get_repo_tiling(self):
        """
        get tiling from repository

        Returns
        -------
        tiling: shape and tiling retrieved from repository
        """
        tiling_list = get_tiling(self.tiling_info)
        res_list = []
        for tiling_mess in tiling_list:
            self._modify_repo_tiling(tiling_mess)
            # in dx_opti, tiling's C_shape returned from repository is 0,
            # we calculate C_shape according to A_shape and stride
            if tiling_mess["C_shape"][2] == 0:
                tiling_mess["C_shape"][2] = tiling_mess["A_shape"][2] * self.stride_h
            if tiling_mess["C_shape"][3] == 0:
                tiling_mess["C_shape"][3] = tiling_mess["A_shape"][3] * self.stride_w
            # pad set -1 to get tilings from repository, so we need to
            # check A_shape&C_shape to filter tilings not matched with
            # current kernel_info out
            t_h, t_w = self._get_input_h(tiling_mess["C_shape"][2]), \
                self._get_input_w(tiling_mess["C_shape"][3])
            if t_h == tiling_mess["A_shape"][2] and t_w == tiling_mess["A_shape"][3]:
                res_list.append(tiling_mess)
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
            self.c_info[0] = shape
        elif self.dynamic_mode == "dynamic_hw":
            self.c_info[2], self.c_info[3] = shape[0], shape[1]
            self.a_info[2] = self._get_input_h(self.c_info[2])
            self.a_info[3] = self._get_input_w(self.c_info[3])
        self.tiling_info["tiling_type"] = "cost_model_tiling"
        for pad in ("padl", "padr", "padu", "padd"):
            self.tiling_info[pad] = 0

        cost_seeds = get_tiling(self.tiling_info)
        tiling = self._check_and_set_default_tiling(cost_seeds[0])

        return tiling

    def get_tiling_range(self, tiling_in, c_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        c_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """

        tiling = self._preprocess_tiling(tiling_in)
        _, _, h_o, w_o, _ = c_shape
        if not tiling["AL1_shape"]:
            # fully load in AL1, covering lower region
            return [1, h_o, 1, w_o]

        # get min value
        ho_min, wo_min = 1, 1
        if self.pad_mode != "SAME":
            ho_min = max(self.k_h - self.cur_pads[2] - self.cur_pads[3], 1)
            wo_min = max(self.k_w - self.cur_pads[0] - self.cur_pads[1], 1)
        support_w_min = wo_min

        ho_max = H_RANGE
        cur_w_size = w_o

        # check tiling covering itself situation
        if not self._check_tiling_match(tiling, cur_w_size, h_o) \
            or h_o > H_RANGE or w_o > W_RANGE:
            return [0, 0, 0, 0]

        # searching down-ward for w_min
        while self._check_tiling_match(tiling, cur_w_size, h_o) and \
                cur_w_size > support_w_min:
            wo_min = cur_w_size
            cur_w_size = cur_w_size - W_DELTA

        # searching up-ward for w_max
        cur_w_size = w_o
        while self._check_tiling_match(tiling, cur_w_size, h_o) \
                and cur_w_size <= W_RANGE:
            wo_max = cur_w_size
            cur_w_size = cur_w_size + W_DELTA

        perf_ho_min = max(ho_min, h_o - H_LEN)
        perf_wo_min = max(wo_min, w_o - W_LEN)
        perf_ho_max = min(ho_max, h_o + H_LEN)
        perf_wo_max = min(wo_max, w_o + W_LEN)

        if perf_wo_min > perf_wo_max:
            return [0, 0, 0, 0]

        def _modify_max_range(ho_range, wo_range):
            """
            modify h_max and w_max according to the limit of ub buffer,
            ensure that aub + cub < ub buffer
            aub = ma * ka * db_flag * bit_num
            cub = mc * nc * m0 * n0 * db_flag * bit_num
            """

            cub_buffer = (reduce(lambda x, y: x * y, tiling_in.get("CUB_matrix"))
                        * tiling_in.get("manual_pingpong_buffer").get("CUB_pbuffer")
                        * BIT_RATIO_DICT.get(self.c_type))
            tiling_k_aub = tiling_in.get("AUB_shape")[0] // (self.b_info[2] * self.b_info[3])
            m_aub_max = ((get_soc_spec("UB_SIZE") - cub_buffer)
                    // BIT_RATIO_DICT.get(self.a_type)
                    // tiling_in.get("manual_pingpong_buffer").get("AUB_pbuffer")
                    // tiling_k_aub / (1 + 1 / self.stride_w))

            if tiling_in.get("AUB_shape")[1] >= 1:
                wo_range = min(wo_range,
                               max(m_aub_max // tiling_in.get("AUB_shape")[1], c_shape[3]))

            return ho_range, wo_range

        def _get_perf_range():
            # modify range for curv performance line
            bool_check_case = utils.icd(
                utils.icd(utils.icd(h_o * w_o, tiling["block_dim"][2]), utils.FP16_M),
                tiling["AL0_matrix"][0]) <= tiling["AL1_shape"][1]
            if not bool_check_case:
                perf_range = [perf_ho_min, perf_ho_max, perf_wo_min, perf_wo_max]
            else:
                range_max = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * \
                            utils.FP16_M * tiling["block_dim"][2]
                if perf_ho_max * perf_wo_max <= range_max:
                    perf_range = [perf_ho_min, perf_ho_max, perf_wo_min, perf_wo_max]
                else:
                    perf_range = [perf_ho_min, range_max // w_o, perf_wo_min, w_o]

            perf_range = [int(v) for v in perf_range]
            return perf_range

        if tiling_in.get("AUB_shape"):
            perf_ho_max, perf_wo_max = _modify_max_range(perf_ho_max, perf_wo_max)
        perf_range = _get_perf_range()
        return perf_range

    def assembly_case(self, tiling, coverage, cnt):
        var_range = OrderedDict()
        if self.dynamic_mode == "dynamic_hw":
            dx_h_low, dx_h_high = int(coverage[0]), int(coverage[1])
            dx_w_low, dx_w_high = int(coverage[2]), int(coverage[3])
            dedy_h_low = self._get_input_h(dx_h_low)
            dedy_w_low = self._get_input_w(dx_w_low)
            dedy_h_high = self._get_input_h(dx_h_high)
            dedy_w_high = self._get_input_w(dx_w_high)

            var_range['dx_h'] = (dx_h_low, dx_h_high)
            var_range['dx_w'] = (dx_w_low, dx_w_high)
            var_range['dedy_h'] = (dedy_h_low, dedy_h_high)
            var_range['dedy_w'] = (dedy_w_low, dedy_w_high)
        elif self.dynamic_mode == "dynamic_batch":
            var_range['batch_n'] = (int(coverage[0]), int(coverage[1]))

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range}

    def _check_and_set_default_tiling(self, tiling_in):
        if tiling_in.get("tiling").get("AL0_matrix")[2] == 32:
            tiling = {}
            _, _, k_w, k_h, _ = self.b_info
            bit_dir = {
                "float32": 16,
                "int32": 16,
                "float16": 16,
                "int8": 32,
            }
            atype = self.tiling_info["A_dtype"]
            btype = self.tiling_info["B_dtype"]
            if atype in bit_dir.keys():
                k_al1 = k_w * k_h * 16
                k_al0 = bit_dir[atype]
            else:
                # default value 32
                k_al1 = 32
                k_al0 = 32

            if btype in bit_dir.keys():
                k_bl1 = bit_dir[atype]
                k_bl0 = bit_dir[atype]
            else:
                # default value 32
                k_bl1 = 32
                k_bl0 = 32

            if self.tiling_info["strideH_expand"] > 1 \
                    or self.tiling_info["strideW_expand"] > 1:
                tiling["AUB_shape"] = [k_w * k_h * 16, 1, 1, 1]
                tiling["BUB_shape"] = None
            else:
                tiling["AUB_shape"] = None
                tiling["BUB_shape"] = None

            tiling["AL1_shape"] = [k_al1, 1, 1, 1]
            tiling["BL1_shape"] = [k_bl1, 1, 1, 1]
            tiling["AL0_matrix"] = [1, 1, 16, k_al0, 1, 1]
            tiling["BL0_matrix"] = [1, 1, 16, k_bl0, 1, 1]
            tiling["CL0_matrix"] = [1, 1, 16, 16, 1, 1]
            tiling["CUB_matrix"] = [1, 1, 16, 16, 1, 1]
            tiling["block_dim"] = [1, 1, 1, 1]
            tiling["n_bef_batch_flag"] = 0
            tiling["n_bef_group_flag"] = 0
            tiling["batch_bef_group_fla"] = 0
            tiling["A_overhead_opt_flag"] = 0
            tiling["B_overhead_opt_flag"] = 0
            tiling["AUB_channel_wise_flag"] = None
            tiling["BUB_channel_wise_flag"] = None
            tiling["CUB_channel_wise_flag"] = None
            tiling["manual_pingpong_buffer"] = {
                'AUB_pbuffer': 1,
                'BUB_pbuffer': 1,
                'AL1_pbuffer': 1,
                'BL1_pbuffer': 1,
                'AL0_pbuffer': 1,
                'BL0_pbuffer': 1,
                'CL0_pbuffer': 1,
                'CUB_pbuffer': 1,
                'UBG_pbuffer': 1,
            }
            tiling = {"tiling": tiling, "A_shape": self.a_info,
                        "B_shape": self.b_info, "C_shape": self.c_info}
        else:
            return tiling_in
        return tiling

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.b_info[2:4]
        self.k_cout = self.b_info[1] * self.b_info[4]
        self.k_cin = self.b_info[0]
        self.stride_h, self.stride_w = self.tiling_info["strideH_expand"], \
                                       self.tiling_info["strideW_expand"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilationH"], \
                                       self.tiling_info["dilationW"],

        self.pad_mode = "SAME" if isinstance(self.tiling_info["padl"], Expr) else "FIX"
        self.cur_pads = []
        for pad in ("padl", "padr", "padu", "padd"):
            self.cur_pads.append(self.tiling_info[pad])
            self.tiling_info[pad] = -1

    def _preprocess_tiling(self, tiling_in):
        """
        preprocess tiling for get tiling range
        """

        tiling = copy.deepcopy(tiling_in)
        if tiling["AL1_shape"]:
            tiling["AL1_shape"][0] = tiling["AL1_shape"][0] // \
                (self.k_h * self.k_w * utils.CUBE_SIZE)
        if tiling["BL1_shape"]:
            tiling["BL1_shape"][0] = tiling["BL1_shape"][0] // \
                (self.k_h * self.k_w * utils.CUBE_SIZE)
        return tiling

    def _get_al1_bound(self, tiling, curent_size):
        """
        get al1 bound info

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of h,w

        Returns
        -------
        int, al1_load_length (al1_bound)

        """

        # shape info
        out_w = curent_size
        w_i = self._get_input_w(out_w, stride=1)

        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'].append(1)

        # M axis theorically loading length in al1
        al1_m_data = tiling['CL0_matrix'][1] * utils.FP16_M * tiling['AL1_shape'][1]

        # load2d instructions refer to data_mov with raw lens
        if (self.pad_mode == "SAME" or sum(self.cur_pads) == 0) and \
                self.k_h * self.k_w == 1:
            return al1_m_data

        # load3d instructions refer to load extra lines with pad/stride/filter
        if al1_m_data % out_w == 0:
            # full line could load without extra lines
            extend_h = 0
        elif (al1_m_data * 2) % out_w == 0:
            # every 2 load3d covered only 1 extra line
            extend_h = 1
        else:
            # other situations need 2 extra lines in case
            extend_h = 2
        l1_ho = al1_m_data // out_w + extend_h

        # calculate input lines (hi) from output lines (ho)
        li_hi = self.k_h + (l1_ho - 1)

        return li_hi * w_i

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
        w_o, h_o = current_w, current_h

        # get M axis length in al1
        al1_bound = self._get_al1_bound(tiling, w_o)

        # fmap size in L1 ( M * K * db * 2byte)
        fmap_l1_size = utils.FP16_SIZE * al1_bound * tiling['AL1_shape'][0] * \
            utils.FP16_K * tiling['manual_pingpong_buffer']['AL1_pbuffer']

        # filter size
        if tiling['BL1_shape'] is None:
            # not using BL1
            filter_l1_size = 0
        elif len(tiling['BL1_shape']) == 0:
            # fully load in BL1
            filter_l1_size = utils.FP16_SIZE * self.k_cout * self.k_cin * self.k_h * \
                self.k_w // tiling['block_dim'][1]
        else:
            # fmap size in L1 ( K * N * db * 2byte)
            filter_l1_size = utils.FP16_SIZE * tiling['BL1_shape'][1] * \
                tiling['CL0_matrix'][0] * utils.FP16_N * tiling['BL1_shape'][0] * \
                utils.FP16_K * self.k_h * self.k_w * \
                tiling['manual_pingpong_buffer']['BL1_pbuffer']

        return fmap_l1_size + filter_l1_size <= utils.L1BUFFER

    def _get_input_h(self, fmap_h, stride=None):
        if not stride:
            stride = self.stride_h
        if self.pad_mode == "SAME":
            return utils.icd(fmap_h, stride)
        return (fmap_h + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h *
                (self.k_h - 1) - 1) // stride + 1

    def _get_input_w(self, fmap_w, stride=None):
        if not stride:
            stride = self.stride_w
        if self.pad_mode == "SAME":
            return utils.icd(fmap_w, stride)
        return (fmap_w + self.cur_pads[0] + self.cur_pads[1] - self.dilate_w *
                (self.k_w - 1) - 1) // stride + 1
