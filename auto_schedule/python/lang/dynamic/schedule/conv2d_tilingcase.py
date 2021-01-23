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
conv2d tiling case
"""
import copy
import math
from collections import OrderedDict

import te
from te.tvm.expr import Expr
from te.domain.tiling.get_tiling import get_tiling

from te.lang.base.operation_impl import register_tiling_case
from te.lang.base.operation_impl import get_te_var

from te.lang.cce.te_compute.conv_compute import ConvParam
from te.lang.dynamic.schedule.cube_tilingcase import TilingSelection
from te.lang.dynamic.schedule.cube_tilingcase import CubeTilingOp
from te.lang.dynamic.schedule.cube_tilingcase import TilingUtils as utils
from te.lang.dynamic.schedule.constants import Pattern


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.CONV2D)
def calc_conv2d(outs, option=None):
    """
    tiling_case func for dynamic shape conv2d

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """

    var_names = ["batch_n", "fmap_h", "fmap_w"]
    conv_info = ConvParam.tiling_info_dict
    tgt_area = {}
    shape_dict = {"batch_n": conv_info.get("a_shape")[0],
                  "fmap_h": conv_info.get("a_shape")[2],
                  "fmap_w": conv_info.get("a_shape")[3]}
    for var_name in var_names:
        if get_te_var(var_name):
            tgt_area[var_name] = tuple(get_te_var(var_name).get_bound())
        else:
            tgt_area[var_name] = (int(shape_dict.get(var_name)), int(shape_dict.get(var_name)))

    # check fusion
    if outs[-1].op.tag == "elewise_single_relu":
        if outs[-1].op.input_tensors[0].op.input_tensors[0].op.tag == "conv_vector_bias_add":
            conv_info['fusion_type'] = 4
            conv_info['fused_coefficient'] = [0, 0, 3]
        else:
            conv_info['fusion_type'] = 1
            conv_info['fused_coefficient'] = [0, 0, 1]
    elif outs[-1].op.input_tensors[0].op.tag == "conv_vector_bias_add":
        conv_info['fusion_type'] = 1
        conv_info['fused_coefficient'] = [0, 0, 2]

    tiling_op = Conv2dTiling(conv_info, ConvParam.dynamic_para)
    tiling_cases = TilingSelection(tiling_op).calc_tiling(tgt_area, var_names)
    return tiling_cases


class Conv2dTiling(CubeTilingOp):
    def __init__(self, tiling_info, dynamic_para):
        super().__init__(tiling_info, None, dynamic_para.get("var_map"))
        self.a_info = tiling_info['a_shape']
        self.a_5hd_info = tiling_info['placeholder_fmap_5hd_shape']
        self.b_info = tiling_info['b_shape']
        self.c_info = tiling_info['c_shape']
        self._get_calc_info()
        self.key = 'A_shape'
        self.op_type = "conv2d"

    def get_repo_tiling(self):
        tiling_list = get_tiling(self.tiling_info)
        res_list = []
        for tiling in tiling_list:
            t_h, t_w = self.get_output_h(tiling["A_shape"][2]), \
                self.get_output_w(tiling["A_shape"][3])
            if t_h == tiling["C_shape"][2] and t_w == tiling["C_shape"][3]:
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

        if "batch_n" in self.var_map:
            self.a_info[0] = shape if isinstance(shape, int) else shape[0]
            self.c_info[0] = shape if isinstance(shape, int) else shape[0]
        if "fmap_h" in self.var_map:
            self.a_info[2] = shape[1]
            self.c_info[2] = self.get_output_h(self.a_info[2])
        if "fmap_w" in self.var_map:
            self.a_info[3] = shape[2]
            self.c_info[3] = self.get_output_w(self.a_info[3])
        if self.pad_mode == "VAR":
            self.tiling_info["pad"] = self._calc_pads(shape[0], shape[1])
        self.tiling_info["tiling_type"] = "cost_model_tiling"
        tiling = get_tiling(self.tiling_info)[0]
        return tiling

    def get_default_tiling(self):
        """
        get default tiling

        Parameters
        ----------
        None

        Returns
        -------
        default tiling
        """

        def _handle_block_dim():
            """
            avoid cyclomatic complexity, handle block_dim
            """
            tiling["block_dim"] = [1, 1, 1, 1]
            device_core_num = utils.CORE_NUM
            if (self.a_info[0] > 1) and (device_core_num > 1):
                if self.a_info[0] <= device_core_num:
                    tiling["block_dim"][0] = self.a_info[0]
                else:
                    for i in range(device_core_num, 0, -1):
                        if self.a_info[0] % i == 0:
                            break
                    tiling["block_dim"][0] = i
            else:
                tiling["block_dim"][0] = 1

        tiling = {}

        tiling_m = 1
        tiling_k = 1
        tiling_n = 1
        tiling["AL1_shape"] = [self.b_info[2] * self.b_info[3] * utils.FP16_K, 1, 1, 1]
        tiling["BL1_shape"] = None
        tiling["AL0_matrix"] = [tiling_m, tiling_k, utils.FP16_M, utils.FP16_K, 1, 1]
        tiling["BL0_matrix"] = [tiling_k, tiling_n, utils.FP16_N, utils.FP16_K, 1, 1]
        tiling["CL0_matrix"] = [tiling_n, tiling_m, utils.FP16_M, utils.FP16_N, 1, 1]
        tiling["CUB_matrix"] = [tiling_n, tiling_m, utils.FP16_M, utils.FP16_N, 1, 1]
        tiling["AUB_shape"] = tiling["AL1_shape"]
        tiling["manual_pingpong_buffer"] = {'AL1_pbuffer': 1,
                                            'BL1_pbuffer': 1,
                                            'AL0_pbuffer': 1,
                                            'BL0_pbuffer': 1,
                                            'CL0_pbuffer': 1,
                                            'CUB_pbuffer': 1,
                                            'UBG_pbuffer': 1,
                                            'AUB_pbuffer': 1}
        tiling["A_overhead_opt_flag"] = False
        tiling["B_overhead_opt_flag"] = False
        tiling["CUB_channel_wise_flag"] = True
        tiling["n_bef_batch_flag"] = False
        _handle_block_dim()

        return tiling

    def get_tiling_range(self, tiling_in, a_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        a_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """
        tiling = self._preprocess_tiling(tiling_in)
        _, _, fmap_h, fmap_w, _ = a_shape

        paras = {
            "var_map": self.var_map,
            "k_h": self.k_h,
            "k_w": self.k_w,
            "pad_mode": self.pad_mode,
            "pads": self.cur_pads
        }
        n_range_min, n_range_max = self.get_batch_range(a_shape[0], paras)
        tiling_range = [n_range_min, n_range_max]
        # check tiling covering itself situation
        if not self.check_tiling_match(tiling, fmap_w, fmap_h) or fmap_h > utils.NHW_MAX or fmap_w > utils.NHW_MAX:
            return tiling_range + [0, 0, 0, 0]
        h_range_min, h_range_max = self.get_h_range(fmap_h, tiling, paras)
        tiling_range += [h_range_min, h_range_max]
        w_range_min, w_range_max = self.get_w_range(fmap_h, fmap_w, tiling, paras)
        tiling_range += [w_range_min, w_range_max]

        if not tiling.get("AL1_shape"):
            return tiling_range

        h_o = self.get_output_h(fmap_h)
        w_o = self.get_output_w(fmap_w)

        # modify range for curv performance line
        if utils.icd(utils.icd(utils.icd(h_o * w_o, tiling["block_dim"][2]), utils.FP16_M),
                     tiling["AL0_matrix"][0]) <= tiling["AL1_shape"][1]:
            range_max = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * utils.FP16_M * tiling["block_dim"][2]
            perf_ho = self.get_output_h(h_range_max)
            perf_wo = self.get_output_w(w_range_max)
            if perf_ho * perf_wo > range_max:
                range_inc = int((math.sqrt((h_o + w_o) ** 2 - 4 * (h_o * w_o - range_max)) - (h_o + w_o)) / 2)
                perf_ho_max = h_o + range_inc
                perf_wo_max = w_o + range_inc
                perf_hi_max_rev = self._get_input_h(perf_ho_max)
                perf_wi_max_rev = self._get_input_w(perf_wo_max)
                perf_hi_max = min(h_range_max, perf_hi_max_rev)
                perf_wi_max = min(w_range_max, perf_wi_max_rev)
                tiling_range[3], tiling_range[5] = perf_hi_max, perf_wi_max

        return tiling_range

    def assembly_case(self, tiling, coverage, cnt):
        var_range = OrderedDict()
        if self.var_map.get("batch_n") is not None:
            var_range['batch_n'] = (coverage[0], coverage[1])

        if self.var_map.get("fmap_h") is not None:
            var_range['fmap_h'] = (coverage[2], coverage[3])
            var_range['ho'] = (self.get_output_h(var_range['fmap_h'][0]),
                               self.get_output_h(var_range['fmap_h'][1]))

        if self.var_map.get("fmap_w") is not None:
            var_range['fmap_w'] = (coverage[4], coverage[5])
            var_range['wo'] = (self.get_output_w(var_range['fmap_w'][0]),
                               self.get_output_w(var_range['fmap_w'][1]))

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range}

    def _get_al1_bound(self, tiling, curent_size):
        """
        get al1 bound info

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of h,w

        Returns
        -------
        int, al1_load_length (al1_bound) in M axis

        """
        # shape info
        h_i, w_i = curent_size, curent_size
        out_w = self.get_output_w(w_i)

        zero_padding = False
        if self.pad_mode == "VAR":
            pad_h = utils.align(h_i, self.stride_h) - self.stride_h + self.k_h_dilation - h_i
            pad_w = utils.align(w_i, self.stride_w) - self.stride_w + self.k_w_dilation - w_i
            zero_padding = pad_h <= 0 and pad_w <= 0
        else:
            zero_padding = sum(self.cur_pads) == 0
        strideh_opti_flag = self.k_h == 1 and self.stride_h > 1 and zero_padding

        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'].append(1)

        # M axis theorically loading length in al1
        al1_m_data = tiling['CL0_matrix'][1] * utils.FP16_M * tiling['AL1_shape'][1]

        # load2d instructions refer to data_mov with raw lens
        if (self.pad_mode == "VAR" or sum(self.cur_pads) == 0) \
            and (self.stride_h * self.stride_w == 1) \
                and (self.k_h * self.k_w == 1):
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
        if not strideh_opti_flag:
            li_hi = self.k_h + (l1_ho - 1) * self.stride_h
        else:
            li_hi = self.k_h + (l1_ho - 1)

        return li_hi * w_i

    def check_tiling_match(self, tiling, current_w, current_h):
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
        if not tiling.get("AL1_shape"):
            return True

        # get M axis length in al1
        al1_bound = self._get_al1_bound(tiling, current_w)

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
                self.k_w / tiling['block_dim'][1]
        else:
            # fmap size in L1 ( K * N * db * 2byte)
            filter_l1_size = utils.FP16_SIZE * tiling['BL1_shape'][1] * \
                tiling['CL0_matrix'][0] * utils.FP16_N * tiling['BL1_shape'][0] * \
                utils.FP16_K * self.k_h * self.k_w * \
                tiling['manual_pingpong_buffer']['BL1_pbuffer']

        return int(fmap_l1_size) + int(filter_l1_size) <= utils.L1BUFFER

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.a_5hd_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.b_info[2:4]
        self.k_cin = self.b_info[1] * self.b_info[4]
        self.k_cout = self.b_info[0]
        self.stride_h, self.stride_w = self.tiling_info["stride"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilation"]

        self.pad_mode = "FIX"
        # currently, in dynamic_hw, when padding is SAME, pad_mode is "VAR"
        if isinstance(self.tiling_info["pad"][0], Expr) or isinstance(self.tiling_info["pad"][2], Expr):
            self.pad_mode = "VAR"
            self.tiling_info["pad"] = [-1, -1, -1, -1]
        self.cur_pads = self.tiling_info["pad"]

        self.k_h_dilation = (self.k_h - 1) * self.dilate_h + 1
        self.k_w_dilation = (self.k_w - 1) * self.dilate_w + 1

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

    def get_output_h(self, h_in):
        """
        calculate output h
        """

        if not h_in:
            return h_in
        if self.pad_mode == "VAR":
            return utils.icd(h_in, self.stride_h)
        return (h_in + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h *
                (self.k_h - 1) - 1) // self.stride_h + 1

    def get_output_w(self, w_in):
        """
        calculate output w
        """

        if not w_in:
            return w_in
        if self.pad_mode == "VAR":
            return utils.icd(w_in, self.stride_w)
        return (w_in + self.cur_pads[0] + self.cur_pads[1] - self.dilate_w *
                (self.k_w - 1) - 1) // self.stride_w + 1

    def _get_input_h(self, h_out):
        """
        calculate max input h
        """

        if self.pad_mode == "VAR":
            return h_out * self.stride_h
        return h_out * self.stride_h - self.cur_pads[2] - self.cur_pads[3] \
            + self.dilate_h * (self.k_h - 1)

    def _get_input_w(self, w_out):
        """
        calculate max input w
        """

        if self.pad_mode == "VAR":
            return w_out * self.stride_w
        return w_out * self.stride_w - self.cur_pads[0] - self.cur_pads[1] \
            + self.dilate_w * (self.k_w - 1)

    def _calc_pads(self, h_in, w_in):
        """
        calculate pads
        """

        pad_h = utils.align(h_in, self.stride_h) - self.stride_h + \
            self.k_h_dilation - h_in
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = utils.align(w_in, self.stride_w) - self.stride_w + \
            self.k_w_dilation - w_in
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return [pad_left, pad_right, pad_up, pad_down]
