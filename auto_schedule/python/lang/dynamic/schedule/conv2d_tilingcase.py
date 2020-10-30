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
from collections import OrderedDict

from te.domain.tiling.get_tiling import get_tiling

from te.platform.operation import register_tiling_case
from te.platform.operation import get_te_var

from te.lang.cce.te_compute.conv_compute import ConvParam
from te.lang.dynamic.schedule.cube_tilingcase import TilingSelection
from te.lang.dynamic.schedule.cube_tilingcase import CubeTilingOp
from te.lang.dynamic.schedule.cube_tilingcase import TilingUtils as utils
from te.lang.dynamic.schedule.constants import Pattern


H_RANGE = 4000
W_RANGE = 4000
W_DELTA = 1
H_LEN = 400
W_LEN = 400


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
    mode = ConvParam.dynamic_mode
    var_names = {'dynamic_batch': ('batch_n', ), 'dynamic_hw': ('fmap_h', 'fmap_w')}
    tgt_area = [get_te_var(v).get_bound() for v in var_names[mode]]
    conv_info = ConvParam.tiling_info_dict

    # check fusion
    if outs[-1].op.tag == 'elewise_single_relu':
        conv_info['fusion_type'] = 3
        conv_info['fused_coefficient'] = [0, 0, 1]

    tiling_op = Conv2dTiling(conv_info, mode)

    tiling_cases = TilingSelection(tiling_op).calc_tiling(tgt_area)
    return tiling_cases


class Conv2dTiling(CubeTilingOp):
    def __init__(self, tiling_info, dynamic_mode):
        super().__init__(tiling_info, dynamic_mode)
        self.a_info = tiling_info['a_shape']
        self.b_info = tiling_info['b_shape']
        self.c_info = tiling_info['c_shape']
        self._get_calc_info()
        self.key = 'A_shape'
        self.op_type = "conv2d"

    def get_repo_tiling(self):
        tiling_list = get_tiling(self.tiling_info)
        return tiling_list

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
            self.a_info[2], self.a_info[3] = shape[0], shape[1]
            self.c_info[2] = self._get_output_h(self.a_info[2])
            self.c_info[3] = self._get_output_w(self.a_info[3])
        self.tiling_info["tiling_type"] = "cost_model_tiling"

        tiling = get_tiling(self.tiling_info)[0]
        return tiling

    def get_tiling_range(self, tiling_in, a_shape):
        """
        get the covered area of a tiling
        """

        tiling = self._preprocess_tiling(tiling_in)
        _, _, fmap_h, fmap_w, _ = a_shape
        if not tiling["AL1_shape"]:
            return [1, fmap_h, 1, fmap_w]

        h_o = self._get_output_h(fmap_h)
        w_o = self._get_output_w(fmap_w)
        # get min value
        hi_min = max(self.k_h - self.padt - self.padb, 1)
        wi_min = max(self.k_w - self.padl - self.padr, 1)

        # get max value
        hi_max = H_RANGE
        cur_w_size = fmap_w
        if not self._check_tiling_match(tiling, cur_w_size, fmap_h):
            return [0, 0, 0, 0]

        while self._check_tiling_match(tiling, cur_w_size, fmap_h) \
                and cur_w_size > max(self.k_w - self.padl - self.padr, 1):
            wi_min = cur_w_size
            cur_w_size = cur_w_size - W_DELTA

        cur_w_size = fmap_w
        while self._check_tiling_match(tiling, cur_w_size, fmap_h) \
                and cur_w_size <= W_RANGE:
            wi_max = cur_w_size
            cur_w_size = cur_w_size + W_DELTA

        perf_hi_min = max(hi_min, fmap_h - H_LEN)
        perf_wi_min = max(wi_min, fmap_w - W_LEN)
        perf_hi_max = min(hi_max, fmap_h + H_LEN)
        perf_wi_max = min(wi_max, fmap_w + W_LEN)

        if perf_wi_min > perf_wi_max:
            return [0, 0, 0, 0]

        if not tiling['AL1_shape']:
            perf_range = [perf_hi_min, perf_hi_max, perf_wi_min, perf_wi_max]
        else:
            bool_check_case = utils.icd(
                utils.icd(utils.icd(h_o * w_o, tiling["block_dim"][2]), utils.FP16_M),
                tiling["AL0_matrix"][0]) <= tiling["AL1_shape"][1]
            if not bool_check_case:
                perf_range = [perf_hi_min, perf_hi_max, perf_wi_min, perf_wi_max]
            else:
                range_max = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * \
                            utils.FP16_M * tiling["block_dim"][2]
                perf_ho = self._get_output_h(perf_hi_max)
                perf_wo = self._get_output_w(perf_wi_max)
                if perf_ho * perf_wo <= range_max:
                    perf_range = [perf_hi_min, perf_hi_max, perf_wi_min, perf_wi_max]
                else:
                    perf_ho_max = range_max // w_o
                    perf_hi_max_rev = self._get_input_h(perf_ho_max)
                    perf_hi_max = min(perf_hi_max, perf_hi_max_rev)
                    perf_range = [perf_hi_min, perf_hi_max, perf_wi_min, fmap_w]

        perf_range = [int(v) for v in perf_range]
        return perf_range

    def assembly_case(self, tiling, coverage, cnt):
        var_range = OrderedDict()
        if self.dynamic_mode == "dynamic_hw":
            var_range['fmap_h'] = (int(coverage[0]), int(coverage[1]))
            var_range['fmap_w'] = (int(coverage[2]), int(coverage[3]))
            var_range['ho'] = (self._get_output_h(var_range['fmap_h'][0]),
                               self._get_output_h(var_range['fmap_h'][1]))
            var_range['wo'] = (self._get_output_w(var_range['fmap_w'][0]),
                               self._get_output_w(var_range['fmap_w'][1]))

        elif self.dynamic_mode == "dynamic_batch":
            var_range['batch_n'] = (int(coverage[0]), int(coverage[1]))

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
        int, al1_load_length (al1_bound)

        """

        # shape info
        h_i, w_i = curent_size, curent_size
        out_w = self._get_output_w(w_i)
        strideh_opti_flag = self.k_h == 1 and self.stride_h > 1 and \
            (self.padl + self.padr + self.padt + self.padb == 0)

        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'].append(1)
        # m_direction in L1 :: AL1_1*mc*m0
        al1_m_data = tiling['CL0_matrix'][1] * utils.FP16_M * \
                     tiling['AL1_shape'][1]

        # load2d ::
        if (self.padl + self.padr + self.padt + self.padb == 0) \
            and (self.stride_h * self.stride_w == 1) \
            and (self.k_h * self.k_w == 1):
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
        if not strideh_opti_flag:
            li_hi = self.k_h + (l1_ho - 1) * self.stride_h
        else:
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
        h_i, w_i = current_h, current_w

        # fmap size
        if tiling['AL1_shape']:
            al1_bound = self._get_al1_bound(tiling, current_w)

            # fmap size in L1
            fmap_l1_size = utils.FP16_SIZE * al1_bound * tiling['AL1_shape'][0] * \
                utils.FP16_K * tiling['manual_pingpong_buffer']['AL1_pbuffer']
        else:
            fmap_l1_size = utils.FP16_SIZE * h_i * w_i * self.k_cin / \
                tiling['block_dim'][2]

        # filter size
        if tiling['BL1_shape'] is None:
            filter_l1_size = 0
        elif len(tiling['BL1_shape']) == 0:
            filter_l1_size = utils.FP16_SIZE * self.k_cout * self.k_cin * self.k_h * \
                self.k_w / tiling['block_dim'][1]
        else:
            filter_l1_size = utils.FP16_SIZE * tiling['BL1_shape'][1] * \
                tiling['CL0_matrix'][0] * utils.FP16_N * tiling['BL1_shape'][0] * \
                utils.FP16_K * self.k_h * self.k_w * \
                tiling['manual_pingpong_buffer']['BL1_pbuffer']

        return int(fmap_l1_size) + int(filter_l1_size) <= utils.L1BUFFER

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.b_info[2:4]
        self.k_cin = self.b_info[1] * self.b_info[4]
        self.k_cout = self.b_info[0]
        self.padl, self.padr, self.padt, self.padb = self.tiling_info["pad"]
        self.stride_h, self.stride_w = self.tiling_info["stride"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilation"]

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

    def _get_output_h(self, h_in):
        return (h_in + self.padt + self.padb - self.dilate_h *
                (self.k_h - 1) - 1) // self.stride_h + 1

    def _get_output_w(self, w_in):
        return (w_in + self.padl + self.padr - self.dilate_w *
                (self.k_w - 1) - 1) // self.stride_w + 1

    def _get_input_h(self, h_out):
        return ((h_out - 1) + 1) * self.stride_h - self.padt - self.padb \
            + self.dilate_h * (self.k_h - 1) + 1

    def _get_input_w(self, w_out):
        return ((w_out - 1) + 1) * self.stride_w - self.padl - self.padr \
            + self.dilate_w * (self.k_w - 1) + 1
