#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

conv2d backprop input tiling case
"""

import copy
import itertools
from collections import OrderedDict

import te.lang.cce
from te import tvm
from . import Pattern
from te.domain.tiling.get_tiling import get_tiling
from te.platform.operation import register_tiling_case
from te.platform.operation import get_te_var
from .cube_tilingcase import CubeTilingSelection
from .cube_tilingcase import TilingUtils as utils


H_RANGE = 4000
W_RANGE = 4000
W_DELTA = 1
H_LEN = 400
W_LEN = 400


@register_tiling_case(pattern=Pattern.CONV2D_BACKPROP_INPUT)
def calc_conv2dbp_input(outs, option=None):
    mode = te.op.get_compile_info().get("dynamic_mode")
    var_names = {'dynamic_batch': ('batch_n',),
                 'dynamic_hw': ('dx_h', 'dx_w')}
    tgt_area = [get_te_var(v).get_bound() for v in var_names[mode]]
    info = te.lang.cce.DynamicConv2dBpInputParams.tiling_info_dict
    cases = Conv2dBpInputTilingSelection(info, mode).calc_tiling(tgt_area)

    return cases


class Conv2dBpInputTilingSelection(CubeTilingSelection):

    def __init__(self, tiling_info, dynamic_mode):
        super().__init__(tiling_info, dynamic_mode)
        self.a_info = tiling_info['A_shape']
        self.b_info = tiling_info['B_shape']
        self.c_info = tiling_info['C_shape']
        self._get_calc_info()
        self._key = 'C_shape'

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.b_info[2:4]
        self.k_cout = self.b_info[1] * self.b_info[4]
        self.k_cin = self.b_info[0]
        self.stride_h, self.stride_w = self.tiling_info["strideH_expand"], \
                                       self.tiling_info["strideW_expand"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilationH"], \
                                       self.tiling_info["dilationW"],

        if isinstance(self.tiling_info["padl"], tvm.expr.Expr):
            self.pad_mode = "SAME"
            self.cur_pads = [-1, -1, -1, -1]
            for p in ("padl", "padr", "padu", "padd"):
                self.tiling_info[p] = -1
        else:
            self.pad_mode = "FIX"
            self.cur_pads = [self.tiling_info["padl"], self.tiling_info["padr"],
                             self.tiling_info["padu"], self.tiling_info["padd"]]

        self.k_h_dilation = (self.k_h - 1) * self.dilate_h + 1
        self.k_w_dilation = (self.k_w - 1) * self.dilate_w + 1

    def _get_tiling(self):
        tiling_list = get_tiling(self.tiling_info)
        for m in tiling_list:
            if m["C_shape"][2:4] == [0, 0]:
                m["C_shape"][2:4] = m["A_shape"][2:4]
        return tiling_list

    def _set_padding_list(self, cur_h, cur_w):
        """
        get padding list in cur dx shape
        """

        if self.pad_mode == "SAME":
            pad_h = max(utils.align(cur_h, self.stride_h) - self.stride_h +
                        self.k_h_dilation - cur_h, 0)
            pad_up = pad_h // 2
            pad_down = pad_h - pad_up
            pad_w = max(utils.align(cur_w, self.stride_w) - self.stride_w +
                        self.k_w_dilation - cur_w, 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            self.cur_pads = [pad_left, pad_right, pad_up, pad_down]

    def _get_input_h(self, h_o):
        return (h_o + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h *
                (self.k_h - 1) - 1) // self.stride_h + 1

    def _get_input_w(self, w_o):
        return (w_o + self.cur_pads[0] + self.cur_pads[1] - self.dilate_h *
                (self.k_w - 1) - 1) // self.stride_w + 1

    def _get_input_h_extend(self, h_o):
        return (h_o + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h *
                (self.k_h - 1) - 1) + 1

    def _get_input_w_extend(self, w_o):
        return (w_o + self.cur_pads[0] + self.cur_pads[1] - self.dilate_h *
                (self.k_w - 1) - 1) + 1

    def _use_cost_model(self, shape):
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
            self._set_padding_list(self.c_info[2], self.c_info[3])
            self.tiling_info['padl'] = self.cur_pads[0]
            self.tiling_info['padr'] = self.cur_pads[1]
            self.tiling_info['padu'] = self.cur_pads[2]
            self.tiling_info['padd'] = self.cur_pads[3]
            self.a_info[2] = self._get_input_h(self.c_info[2])
            self.a_info[3] = self._get_input_w(self.c_info[3])
        self.tiling_info["tiling_type"] = "cost_model_tiling"

        tiling = get_tiling(self.tiling_info)[0]
        return tiling

    def _get_tiling_range(self, tiling_in, c_shape):
        """
        get the covered area of a tiling
        """

        tiling = self._preprocess_tiling(tiling_in)
        _, _, h_o, w_o, _ = c_shape
        if not tiling["AL1_shape"]:
            return [1, h_o, 1, w_o]

        self._set_padding_list(h_o, w_o)

        # get min value
        ho_min = max(self.k_h - self.cur_pads[2] - self.cur_pads[3], 1)
        wo_min = max(self.k_w - self.cur_pads[0] - self.cur_pads[1], 1)

        # get max value
        ho_max = H_RANGE
        cur_w_size = w_o
        if not self._check_tiling_match(tiling, cur_w_size, h_o):
            return [0, 0, 0, 0]

        while self._check_tiling_match(tiling, cur_w_size, h_o) and \
            cur_w_size > max(self.k_w - self.cur_pads[0] - self.cur_pads[1], 1):
            wo_min = cur_w_size
            cur_w_size = cur_w_size - W_DELTA

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

        if not tiling['AL1_shape']:
            perf_range = [perf_ho_min, perf_ho_max, perf_wo_min, perf_wo_max]
        else:
            bool_check_case = utils.icd(utils.icd(utils.icd(h_o * w_o, tiling["block_dim"][2]), 
                utils.FP16_M), tiling["AL0_matrix"][0]) <= tiling["AL1_shape"][1]
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
        w_i = self._get_input_w_extend(out_w)

        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'].append(1)
        # m_direction in L1 :: AL1_1*mc*m0
        al1_m_data = tiling['CL0_matrix'][1] * utils.FP16_M * tiling['AL1_shape'][1]

        # load2d ::
        if sum(self.cur_pads) == 0 and self.k_h * self.k_w == 1:
            return al1_m_data

        # load3d ::
        # m_direction ho :: if mod==0. no need to load extend lines
        if al1_m_data % out_w == 0:
            extend_h = 0
        elif (al1_m_data * 2) % out_w == 0:
            extend_h = 1
        else:
            extend_h = 2
        l1_ho = al1_m_data // out_w + extend_h

        # hi cal from ho
        li_hi = self.k_h + (l1_ho - 1)

        return li_hi * w_i

    def _check_tiling_match(self, tiling, curent_w, current_h):
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
        w_o, h_o = curent_w, current_h
        self._set_padding_list(h_o, w_o)
        h_i = self._get_input_h_extend(h_o)
        w_i = self._get_input_w_extend(w_o)

        # fmap size
        if tiling['AL1_shape']:
            al1_bound = self._get_al1_bound(tiling, w_o)

            # fmap size in L1
            fmap_l1_size = utils.FP16_SIZE * al1_bound * tiling['AL1_shape'][0] * \
                utils.FP16_K * tiling['manual_pingpong_buffer']['AL1_pbuffer']
        else:
            fmap_l1_size = utils.FP16_SIZE * h_i * w_i * self.k_cin // tiling['block_dim'][2]

        # filter size
        if tiling['BL1_shape'] is None:
            filter_l1_size = 0
        elif len(tiling['BL1_shape']) == 0:
            filter_l1_size = utils.FP16_SIZE * self.k_cout * self.k_cin * self.k_h * \
                self.k_w // tiling['block_dim'][1]
        else:
            filter_l1_size = utils.FP16_SIZE * tiling['BL1_shape'][1] * \
                tiling['CL0_matrix'][0] * utils.FP16_N * tiling['BL1_shape'][0] * \
                utils.FP16_K * self.k_h * self.k_w * \
                tiling['manual_pingpong_buffer']['BL1_pbuffer']

        return fmap_l1_size + filter_l1_size <= utils.L1BUFFER

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

    def _assembly_case(self, tiling, coverage, cnt):
        var_range = OrderedDict()
        if self.dynamic_mode == "dynamic_hw":
            dx_h_low, dx_h_high = int(coverage[0]), int(coverage[1])
            dx_w_low, dx_w_high = int(coverage[2]), int(coverage[3])
            self._set_padding_list(dx_h_low, dx_w_low)
            dedy_h_low = self._get_input_h(dx_h_low)
            dedy_w_low = self._get_input_w(dx_w_low)
            self._set_padding_list(dx_h_high, dx_w_high)
            dedy_h_high = self._get_input_h(dx_h_high)
            dedy_w_high = self._get_input_w(dx_w_high)

            var_range['dx_h'] = (dx_h_low, dx_h_high)
            var_range['dx_w'] = (dx_w_low, dx_w_high)
            var_range['dedy_h'] = (dedy_h_low, dedy_h_high)
            var_range['dedy_w'] = (dedy_w_low, dedy_w_high)
        elif self.dynamic_mode == "dynamic_batch":
            var_range['batch_n'] = (int(coverage[0]), int(coverage[1]))

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range}
