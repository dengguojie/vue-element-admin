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
conv3d backprop input tiling case
"""
import copy
import math
from collections import OrderedDict
from functools import reduce

from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.compute.conv3d_backprop_input_compute import DynamicConv3dBpInputParams
from tbe.dsl.unify_schedule.cube_tilingcase import CubeTilingOp
from tbe.dsl.unify_schedule.cube_tilingcase import TilingSelection
from tbe.dsl.unify_schedule.cube_tilingcase import TilingUtils
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.tvm.expr import Expr
from tbe.tvm.intrin import abs as tvm_abs


H_RANGE = 4096
W_RANGE = 4096
W_DELTA = 1
D_DELTA = 1
H_LEN = 400
W_LEN = 400
D_LEN = 400
VALID_TILING_NUM = 32
_DEFAULT_TILING_FLAG = 32
_C0_SIZE = tbe_platform.C0_SIZE
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}


@register_tiling_case(pattern=Pattern.CONV3D_BACKPROP_INPUT)
def calc_conv3dbp_input(outs, option=None):
    """
    tiling_case func for dynamic shape conv3d_bp_input

    Parameters
    ----------
    outs : tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """

    var_names = ("batch_n", "dedx_d", "dedx_h", "dedx_w")

    tgt_area = {}

    conv_info = DynamicConv3dBpInputParams.tiling_info_dict

    shape_dict = {"batch_n": conv_info.get("c_shape")[0],
                  "dedx_d": conv_info.get("c_shape")[1],
                  "dedx_h": conv_info.get("c_shape")[2],
                  "dedx_w": conv_info.get("c_shape")[3]}

    for var_name in var_names:
        if get_te_var(var_name):
            tgt_area[var_name] = tuple(get_te_var(var_name).get_bound())
        else:
            tgt_area[var_name] = (int(shape_dict.get(var_name)), int(shape_dict.get(var_name)))

    tiling_op = Conv3dBpInputTiling(conv_info, DynamicConv3dBpInputParams.var_map)
    tiling_cases = TilingSelection(tiling_op).calc_tiling(tgt_area, var_names)
    add_compile_info("dedy_c1", conv_info.get("a_shape")[2])
    return tiling_cases


class Conv3dBpInputTiling(CubeTilingOp):
    def __init__(self, tiling_info, var_map):
        super().__init__(tiling_info, None, var_map)
        self.a_info = self.tiling_info["a_shape"]
        self.b_info = self.tiling_info["b_shape"]
        self.c_info = self.tiling_info["c_shape"]
        self.a_type = self.tiling_info["a_dtype"]
        self.c_type = self.tiling_info["c_dtype"]
        self.var_map = var_map
        self._get_calc_info()
        self.key = "C_shape"
        self.op_type = "conv3d_backprop_input"

    def _get_d_factor(self, tiling, stride_d, kernel_d, dedy_d):
        """
        get d_factor value

        Parameters
        ----------
        tiling: tiling information

        stride_d: stride in d dimension

        kernel_d: kernel size in d dimension

        dedy_d: stride in d dimension

        Returns
        ---------
        d_factor: d value in AL1 and UB
        """
        al0_tiling_dfactor = tiling["AL0_matrix"][-1]
        if tiling.get("BL0_matrix"):
            bl0_tiling_kd = tiling["BL0_matrix"][-1]
        else:
            bl0_tiling_kd = kernel_d
        if tiling.get("BL1_shape"):
            b_factor = min(tiling["BL1_shape"][-1] * bl0_tiling_kd, kernel_d)
        else:
            b_factor = kernel_d
        ext = ((al0_tiling_dfactor - 1) + stride_d - 1) // stride_d
        estimate_d = ((b_factor - 1) + stride_d -1) // stride_d + ext + 1
        d_factor = min(estimate_d, dedy_d)
        return d_factor

    def get_padding(self, padh, padt, padu, padl):
        """
        get padding in tiling info for cost model's - get tiling function

        Parameters
        ----------
        padh: head pad in d dimension

        padt: tail pad in d dimension

        padu: head pad in h dimension

        padl: head pad in w dimension

        Returns
        --------
        pads: pads using for get tiling
        """
        new_up_pad = self.k_h_dilation - padu - 1
        new_left_pad = self.k_w_dilation - padl - 1
        new_down_pad = (self.c_info[2] - 1) + self.k_h_dilation - self.a_info[3] * self.stride_h - new_up_pad
        new_right_pad = (self.c_info[3] - 1) + self.k_w_dilation - self.a_info[4] * self.stride_w - new_left_pad
        if (self.stride_h > 1 or self.stride_w > 1) and (new_down_pad < 0 or new_right_pad < 0):
            new_down_pad = (new_down_pad + abs(new_down_pad)) // 2
            new_right_pad = (new_right_pad + abs(new_right_pad)) // 2
        return [padh, padt, new_up_pad, new_down_pad, new_left_pad, new_right_pad]

    def get_repo_tiling(self):
        """
        get tiling from repository

        Returns
        -------
        tiling: shape and tiling retrieved from repository
        """
        self.tiling_info["pad"] = [-1, -1, -1, -1, -1, -1]
        tiling_list = get_tiling(self.tiling_info)

        res_list = []
        for tiling_mess in tiling_list:
            # pad set -1 to get tilings from repository, so we need to
            # check A_shape&C_shape to filter tilings not matched with
            # current kernel_info out
            dedx_d, dedx_h, dedx_w = tiling_mess["C_shape"][1], tiling_mess["C_shape"][2], tiling_mess["C_shape"][3]
            t_d = self._get_dedy_d(dedx_d, self.stride_d)
            t_h = self._get_dedy_h(dedx_h, self.stride_h)
            t_w = self._get_dedy_w(dedx_w, self.stride_w)
            if (t_d == tiling_mess["A_shape"][1] and t_h == tiling_mess["A_shape"][3] and
                t_w == tiling_mess["A_shape"][4] and
                self._check_exceed_ub_buffer(tiling_mess.get("tiling"), dedx_d, dedx_h, dedx_w)):
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

        if "batch_n" in self.var_map:
            self.a_info[0] = shape if isinstance(shape, int) else shape[0]
            self.c_info[0] = shape if isinstance(shape, int) else shape[0]
        if "dedx_d" in self.var_map:
            self.c_info[1] = shape[1]
            self.a_info[1] = self._get_dedy_d(self.c_info[1], self.stride_d)
        if "dedx_h" in self.var_map:
            self.c_info[2] = shape[2]
            self.a_info[3] = self._get_dedy_h(self.c_info[2], self.stride_h)
        if "dedx_w" in self.var_map:
            self.c_info[3] = shape[3]
            self.a_info[4] = self._get_dedy_w(self.c_info[3], self.stride_w)
        self.tiling_info["tiling_type"] = "cost_model_tiling"

        if self.pad_mode == "VAR":
            pad_d = ((self.c_info[1] + self.stride_d - 1) // self.stride_d * self.stride_d - self.stride_d
                     + self.k_d_dilation - self.c_info[1])
            pad_d = max(pad_d, 0)
            pad_head = pad_d // 2
            pad_tail = pad_d - pad_head
            pad_h = ((self.c_info[2] + self.stride_h - 1) // self.stride_h * self.stride_h - self.stride_h
                     + self.k_h_dilation - self.c_info[2])
            pad_h = max(pad_h, 0)
            pad_up = pad_h // 2
            pad_w = ((self.c_info[3] + self.stride_w - 1) // self.stride_w * self.stride_w - self.stride_w
                     + self.k_w_dilation - self.c_info[3])
            pad_w = max(pad_w, 0)
            pad_left = pad_w // 2
        else:
            pad_head, pad_tail, pad_up, pad_left = self.padh, self.padt, self.padu, self.padl
        self.tiling_info["pad"] = self.get_padding(pad_head, pad_tail, pad_up, pad_left)
        cost_seeds = get_tiling(self.tiling_info)
        tiling = self._check_and_set_default_tiling(cost_seeds[0])

        return tiling

    def get_batch_range(self, batch, paras):
        """
        get batch covering range
        """
        if "batch_n" in paras.get("var_map"):
            core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
            if batch >= core_num:
                return core_num, -1
            if core_num == TilingUtils.N_BASE:
                return 1, -1
            batch_log = int(math.log(batch, TilingUtils.N_BASE))
            return TilingUtils.N_BASE ** batch_log, TilingUtils.N_BASE ** (int(batch_log + 1))
        return batch, batch

    def get_h_range(self, fmap_h, tiling, paras):
        """
        get h covering range
        """
        if "dedx_h" in paras.get("var_map") or "dedy_h" in paras.get("var_map"):
            if not tiling["AL1_shape"]:
                return 1, fmap_h
            hi_min = TilingUtils.HW_MIN
            if paras.get("pad_mode") != "VAR":
                hi_min = max(paras.get("k_h") - self.padu - self.padd, hi_min)
            hi_min = max(hi_min, fmap_h - H_LEN)
            hi_max = min(TilingUtils.NHW_MAX, fmap_h + H_LEN)
            return hi_min, hi_max
        return fmap_h, fmap_h

    def get_w_range(self, fmap_h, fmap_w, fmap_d, tiling, paras):
        """
        get w covering range
        """
        if "dedx_w" in paras.get("var_map") or "dedy_w" in paras.get("var_map"):
            if not tiling["AL1_shape"]:
                return 1, fmap_w
            wi_min = TilingUtils.HW_MIN
            if paras.get("pad_mode") != "VAR":
                wi_min = max(paras.get("k_w") - self.padl - self.padr, wi_min)
            support_w_min = wi_min
            cur_w_size = fmap_w
            # searching down-ward fo w_min
            while cur_w_size >= support_w_min and self._check_tiling_match(tiling, cur_w_size, fmap_h, fmap_d):
                wi_min = cur_w_size
                cur_w_size -= W_DELTA
            # searching up-ward for w_max
            cur_w_size = fmap_w
            wi_max = fmap_w
            while cur_w_size <= TilingUtils.NHW_MAX and self._check_tiling_match(tiling, cur_w_size, fmap_h, fmap_d):
                wi_max = cur_w_size
                cur_w_size += W_DELTA
            wi_min = max(wi_min, fmap_w - W_LEN)
            wi_max = min(wi_max, fmap_w + W_LEN)
            if wi_min > wi_max:
                return 0, 0
            return wi_min, wi_max
        return fmap_w, fmap_w

    def get_d_range(self, fmap_h, fmap_w_min, fmap_w_max, fmap_d, tiling, paras):
        """
        get d covering range
        """
        if "dedx_d" in paras.get("var_map") or "dedy_d" in paras.get("var_map"):
            if not tiling["AL1_shape"]:
                return 1, fmap_d
            di_min = TilingUtils.HW_MIN
            if paras.get("pad_mode") != "VAR":
                di_min = max(paras.get("k_d") - self.padh - self.padt, di_min)
            support_d_min = di_min
            cur_d_size = fmap_d
            di_max = -1
            # searching down-ward fo d_min
            while cur_d_size >= support_d_min and self._check_tiling_match(tiling, fmap_w_min, fmap_h, cur_d_size):
                di_min = cur_d_size
                cur_d_size -= D_DELTA
            di_min = max(di_min, fmap_d - D_LEN)
            return di_min, di_max
        return fmap_d, fmap_d

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
        def _get_perf_range():
            # modify range for curv performance line
            if len(tiling["AL1_shape"]) == 0:
                al1_k_modify = (self.k_cout + TilingUtils.CUBE_SIZE - 1) // TilingUtils.CUBE_SIZE
                m0 = tbe_platform.CUBE_MKN[self.c_type]["mac"][0]
                m_aligned = (c_shape[2] * c_shape[3] + m0 - 1) // m0
                cl0_tiling_mc = tiling_in["CL0_matrix"][1]
                m_dim = tiling_in["block_dim"][2]
                tiling["AL1_shape"] = [al1_k_modify, m_aligned // cl0_tiling_mc // m_dim]
            bool_check_case = TilingUtils.icd(
                TilingUtils.icd(TilingUtils.icd(h_o * w_o, tiling["block_dim"][2]), TilingUtils.FP16_M),
                tiling["AL0_matrix"][0]) <= tiling["AL1_shape"][1]
            if bool_check_case:
                range_max = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * \
                            TilingUtils.FP16_M * tiling["block_dim"][2]
                if h_range_max * w_range_max > range_max:
                    return range_max // w_o, w_o
            return h_range_max, w_range_max

        def _modify_max_range():
            """
            modify h_max and w_max according to the limit of ub buffer,
            ensure that aub + cub < ub buffer
            aub = ma * ka * db_flag * bit_num * d_factor
            cub = mc * nc * m0 * n0 * db_flag * bit_num
            """
            if tiling_in.get("AUB_shape"):
                cub_buffer = (reduce(lambda x, y: x * y, tiling_in["CUB_matrix"][:4]) *
                              tiling_in.get("manual_pingpong_buffer").get("CUB_pbuffer") *
                              BIT_RATIO_DICT.get(self.c_type))
                tiling_k_aub = tiling_in.get("AUB_shape")[0] // (self.b_info[3] * self.b_info[4])
                dy_d = self._get_dedy_d(d_o, self.stride_d)
                d_factor = self._get_d_factor(tiling_in, self.stride_d, self.k_d, dy_d)
                m_aub_max = ((tbe_platform_info.get_soc_spec("UB_SIZE") - cub_buffer) //
                             BIT_RATIO_DICT.get(self.a_type) //
                             tiling_in.get("manual_pingpong_buffer").get("AUB_pbuffer") //
                             tiling_k_aub // d_factor / (1 + 1 / self.stride_w))

                if tiling_in.get("AUB_shape")[1] >= 1:
                    w_range = min(w_range_max, max(m_aub_max // tiling_in.get("AUB_shape")[1], c_shape[3]))
                    return w_range
            return w_range_max

        tiling = self._preprocess_tiling(tiling_in, c_shape)
        n_o, d_o, h_o, w_o, _ = c_shape

        paras = {
            "var_map": self.var_map,
            "k_h": self.k_h,
            "k_w": self.k_w,
            "k_d": self.k_d,
            "pad_mode": self.pad_mode,
        }
        n_range_min, n_range_max = self.get_batch_range(n_o, paras)
        tiling_range_n = [n_range_min, n_range_max]

        if not self._check_tiling_match(tiling, w_o, h_o, d_o) or h_o > H_RANGE or w_o > W_RANGE:
            return tiling_range_n + [0, 0, 0, 0, 0, 0]
        h_range_min, h_range_max = self.get_h_range(h_o, tiling, paras)

        w_range_min, w_range_max = self.get_w_range(h_o, w_o, d_o, tiling, paras)

        d_range_min, d_range_max = self.get_d_range(h_o, w_range_min, w_range_max, d_o, tiling, paras)

        w_range_max = _modify_max_range()

        h_range_max, w_range_max = _get_perf_range()

        return tiling_range_n + [d_range_min, d_range_max, h_range_min, h_range_max, w_range_min, w_range_max]

    def assembly_case(self, tiling, coverage, cnt):
        """
        Configure dict of tiling strategy and coverage

        Parameters
        ----------
        tiling: dict, tiling from repository or cost model

        coverage: list of tuple, coverage of tiling

        cnt: serial number of tiling

        Returns
        -------
        dict: describe a tiling strategy
        """
        var_range = OrderedDict()
        if "batch_n" in self.var_map:
            var_range['batch_n'] = (TilingUtils.trans_to_int(coverage[0]), TilingUtils.trans_to_int(coverage[1]))
        if "dedy_d" in self.var_map:
            dx_d_low, dx_d_high = TilingUtils.trans_to_int(coverage[2]), TilingUtils.trans_to_int(coverage[3])
            dedy_d_low = self._get_dedy_d(dx_d_low, self.stride_d)
            dedy_d_high = self._get_dedy_d(dx_d_high, self.stride_d)
            var_range['dedx_d'] = (dx_d_low, dx_d_high)
            var_range['dedy_d'] = (dedy_d_low, dedy_d_high)
        if "dedy_h" in self.var_map:
            dx_h_low, dx_h_high = TilingUtils.trans_to_int(coverage[4]), TilingUtils.trans_to_int(coverage[5])
            dedy_h_low = self._get_dedy_h(dx_h_low, self.stride_h)
            dedy_h_high = self._get_dedy_h(dx_h_high, self.stride_h)
            var_range['dedx_h'] = (dx_h_low, dx_h_high)
            var_range['dedy_h'] = (dedy_h_low, dedy_h_high)
        if "dedy_w" in self.var_map:
            dx_w_low, dx_w_high = TilingUtils.trans_to_int(coverage[6]), TilingUtils.trans_to_int(coverage[7])
            dedy_w_low = self._get_dedy_w(dx_w_low, self.stride_w)
            dedy_w_high = self._get_dedy_w(dx_w_high, self.stride_w)
            var_range['dedx_w'] = (dx_w_low, dx_w_high)
            var_range['dedy_w'] = (dedy_w_low, dedy_w_high)
        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range}

    def get_default_range(self, tgt_area):
        if not tgt_area[5]:
            tgt_area[5] = H_RANGE

        if not tgt_area[7]:
            fmap_w = 1
            while fmap_w <= W_RANGE:
                if not self._check_l1_limitation(fmap_w):
                    break
                dedy_w = self._get_dedy_w(fmap_w, self.stride_w)
                if not self._check_ub_limitation(dedy_w):
                    break
                fmap_w += 1
            tgt_area[7] = fmap_w - 1

        return super(Conv3dBpInputTiling, self).get_default_range(tgt_area)

    def _check_ub_limitation(self, dedy_w_upper):
        w_value = dedy_w_upper * self.stride_w

        aub_dedy_size_min = dedy_w_upper * _C0_SIZE * 2
        aub_filling_size_min = w_value * _C0_SIZE * 2
        cub_size_min = _C0_SIZE * _C0_SIZE * 2
        ub_size = tbe_platform_info.get_soc_spec("UB_SIZE")

        return (aub_dedy_size_min + aub_filling_size_min + cub_size_min) <= ub_size

    def _check_l1_limitation(self, fmap_w_upper):
        w_value = fmap_w_upper * self.stride_w
        if fmap_w_upper > _C0_SIZE:
            h_value_max = self.k_h_dilation + 1
        elif _C0_SIZE % fmap_w_upper == 0:
            h_value_max = self.k_h_dilation + _C0_SIZE // fmap_w_upper - 1
        else:
            h_value_max = self.k_h_dilation + _C0_SIZE // fmap_w_upper + 1

        a_l1_size = h_value_max * w_value * ((self.k_d_dilation - 2) // self.stride_d + 2) * _C0_SIZE * 2
        b_l1_size = self.k_h_dilation * self.k_w_dilation * self.k_d_dilation * _C0_SIZE * _C0_SIZE * 2
        l1_size = tbe_platform_info.get_soc_spec("L1_SIZE")
        return (a_l1_size + b_l1_size) <= l1_size

    def get_default_tiling(self):
        """
        get default tiling for unlimited range or special case

        Returns
        -------
        dict: default tiling for conv3d_bp_input
        """
        tiling = {}
        # defaut value 16
        k0_size = tbe_platform.CUBE_MKN[self.a_type]["mac"][1]
        k_al1 = self.k_h * self.k_w * k0_size
        dsl_flag = DynamicConv3dBpInputParams.tiling_info_dict["fused_coefficient"][0]

        if self.stride_h > 1 or self.stride_w > 1 or dsl_flag:
            tiling["AUB_shape"] = [k_al1, 1, 1, 1]
            tiling["BUB_shape"] = None
        else:
            tiling["AUB_shape"] = None
            tiling["BUB_shape"] = None

        tiling["AL1_shape"] = [k_al1, 1, 1, 1]
        tiling["BL1_shape"] = [k0_size, 1, 1, 1]
        tiling["AL0_matrix"] = [1, 1, 16, k0_size, 1, 1]
        tiling["BL0_matrix"] = [1, 1, 16, k0_size, 1, 1]
        tiling["CL0_matrix"] = [1, 1, 16, 16, 1, 1]
        tiling["CUB_matrix"] = [1, 1, 16, 16, 1, 1]
        tiling["block_dim"] = [1, 1, 1, 1]
        tiling["n_bef_batch_flag"] = 0
        tiling["n_bef_group_flag"] = 0
        tiling["batch_bef_group_flag"] = 0
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

        return tiling

    def _check_and_set_default_tiling(self, tiling_in):
        """
        get default tiling for unlimited range or special case

        Returns
        -------
        dict: default tiling for conv3d_bp_input
        """
        if tiling_in.get("tiling").get("AL0_matrix")[2] == _DEFAULT_TILING_FLAG:
            tiling = self.get_default_tiling()
            return {"tiling": tiling, "A_shape": self.a_info,
                    "B_shape": self.b_info, "C_shape": self.c_info}
        else:
            return tiling_in

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info, self.c_info)
        self.k_d, self.k_h, self.k_w = self.b_info[1], self.b_info[3], self.b_info[4]
        self.k_cin = self.b_info[2] * self.b_info[5]
        self.k_cout = self.b_info[0]
        self.stride_d = self.tiling_info["stride"][0]
        self.stride_h = self.tiling_info["strideh_expand"]
        self.stride_w = self.tiling_info["stridew_expand"]
        self.dilate_d, self.dilate_h, self.dilate_w = self.tiling_info["dilation"]

        self.k_d_dilation = (self.k_d - 1) * self.dilate_d + 1
        self.k_h_dilation = (self.k_h - 1) * self.dilate_h + 1
        self.k_w_dilation = (self.k_w - 1) * self.dilate_w + 1

        self.pad_mode = "FIX"
        if any(isinstance(i, Expr) for i in self.tiling_info["pad"]):
            self.pad_mode = "VAR"
        self.padh, self.padt, self.padu, self.padd, self.padl, self.padr = self.tiling_info["pad"]

    def _preprocess_tiling(self, tiling_in, c_shape):
        """
        preprocess tiling for get tiling range
        """
        tiling = copy.deepcopy(tiling_in)
        if tiling["AL1_shape"]:
            tiling["AL1_shape"][0] = tiling["AL1_shape"][0] // (self.k_h * self.k_w * TilingUtils.CUBE_SIZE)
        else:
            al1_k_modify = (self.k_cout + TilingUtils.CUBE_SIZE - 1) // TilingUtils.CUBE_SIZE
            m0 = tbe_platform.CUBE_MKN[self.c_type]["mac"][0]
            m_aligned = (c_shape[2] * c_shape[3] + m0 - 1) // m0
            cl0_tiling_mc = tiling_in["CL0_matrix"][1]
            m_dim = tiling_in["block_dim"][2]
            tiling["AL1_shape"] = [al1_k_modify, m_aligned // cl0_tiling_mc // m_dim]
        if tiling["BL1_shape"]:
            tiling["BL1_shape"][0] = tiling["BL1_shape"][0] // (self.k_h * self.k_w * TilingUtils.CUBE_SIZE)
        return tiling

    def _get_al1_bound(self, tiling, current_size_w, current_size_h):
        """
        get al1 bound info

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size_w : int, size of w

        current_size_h : int, size of h

        Returns
        -------
        int, al1_load_length (al1_bound)

        """

        # shape info
        out_w, out_h = current_size_w, current_size_h
        w_i = self._get_dedy_w(out_w, stride_w=1)

        if len(tiling['AL1_shape']) == 0:
            m_aligned = (out_w * out_h + TilingUtils.CUBE_SIZE - 1) // TilingUtils.CUBE_SIZE
            l0c_tiling_mc = tiling["CL0_matrix"][1]
            m0 = tbe_platform.CUBE_MKN[self.c_type]["mac"][0]
            al1_tiling_m = m_aligned // m0 // l0c_tiling_mc // tiling["block_dim"][2]
        else:
            al1_tiling_m = tiling["AL1_shape"][1]

        # M axis theorically loading length in al1
        al1_m_data = tiling['CL0_matrix'][1] * TilingUtils.FP16_M * al1_tiling_m

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

    def _check_exceed_l1_buffer(self, tiling, current_w, current_h, current_d):
        # shape info
        w_o, h_o, d_o = current_w, current_h, current_d

        # get M axis length in al1
        al1_bound = self._get_al1_bound(tiling, w_o, h_o)

        # get d
        dy_d = self._get_dedy_d(d_o, self.stride_d)
        d_factor = self._get_d_factor(tiling, self.stride_d, self.k_d, dy_d)

        # fmap size in L1 (d * M * K * db * 2byte)
        fmap_l1_size = (d_factor * al1_bound * tiling["AL1_shape"][0] *
                        TilingUtils.FP16_K * tiling["manual_pingpong_buffer"]["AL1_pbuffer"] * TilingUtils.FP16_SIZE)

        # filter size
        if tiling["BL1_shape"] is None:
            # not using BL1
            filter_l1_size = 0
        elif len(tiling["BL1_shape"]) == 0:
            # fully load in BL1
            filter_l1_size = (self.k_d * self.k_h * self.k_w * self.k_cin * self.k_cout *
                              TilingUtils.FP16_SIZE // tiling['block_dim'][1])
        else:
            # fmap size in L1 (d * K * N * db * 2byte)
            kd_tiling_l1_factor = tiling["BL1_shape"][3]
            kd_factor = tiling['BL0_matrix'][5] if tiling.get("BL0_matrix") else self.k_d
            bl1_d = kd_factor * kd_tiling_l1_factor
            filter_l1_size = (bl1_d * tiling["BL1_shape"][1] * tiling["CL0_matrix"][0] *
                              TilingUtils.FP16_N * tiling["BL1_shape"][0] * TilingUtils.FP16_K * self.k_h * self.k_w *
                              tiling["manual_pingpong_buffer"]["BL1_pbuffer"] * TilingUtils.FP16_SIZE)
        return fmap_l1_size + filter_l1_size <= TilingUtils.L1BUFFER

    def _check_exceed_ub_buffer(self, tiling, current_d, current_w, current_h):
        # shape info
        if self.stride_w == 1 and self.stride_h == 1:
            return True
        w_o, h_o, d_o = current_w, current_h, current_d
        dy_d = self._get_dedy_d(d_o, self.stride_d)
        dy_w = self._get_dedy_w(w_o, self.stride_w)
        aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
        aub_tiling_k_factor = aub_tiling_k // (self.k_h * self.k_w * TilingUtils.CUBE_SIZE)
        aub_tiling_m_factor = aub_tiling_m

        d_factor = self._get_d_factor(tiling, self.stride_d, self.k_d, dy_d)

        dedy_ub_size = ((d_factor * aub_tiling_k_factor * dy_w * TilingUtils.CUBE_SIZE * TilingUtils.FP16_SIZE *
                        TilingUtils.icd(aub_tiling_m_factor, self.stride_h)) *
                        tiling["manual_pingpong_buffer"]["AUB_pbuffer"])

        dy_filing_size = (d_factor * aub_tiling_k_factor * aub_tiling_m_factor * (dy_w * self.stride_w) *
                          TilingUtils.CUBE_SIZE * TilingUtils.FP16_SIZE * tiling["manual_pingpong_buffer"]["AUB_pbuffer"])

        cub_size = (tiling["CUB_matrix"][0] * tiling["CUB_matrix"][1] * TilingUtils.CUBE_SIZE**2 * TilingUtils.FP16_SIZE *
                     tiling["manual_pingpong_buffer"]["CUB_pbuffer"])

        return (dedy_ub_size + dy_filing_size + cub_size) <= tbe_platform_info.get_soc_spec("UB_SIZE")



    def _check_tiling_match(self, tiling, current_w, current_h, current_d):
        """

        check whether this tiling matches the shape

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_d : int, size of d

        current_h : int, size of h

        current_w : int, size of w

        Returns
        -------
        bool, True: match
        False: do not match

        """
        if not self._check_exceed_l1_buffer(tiling, current_w, current_h, current_d):
            return False

        if self.stride_h > 1 or self.stride_w > 1:
            if not self._check_exceed_ub_buffer(tiling, current_d, current_w, current_h):
                return False

        return True

    def _get_dedy_d(self, fmap_d, stride_d):
        """
        calculate output d
        """
        if not fmap_d:
            return None
        if self.pad_mode == "VAR":
            return TilingUtils.icd(fmap_d, stride_d)
        return (fmap_d + self.padh + self.padt - self.k_d_dilation) // stride_d + 1

    def _get_dedy_h(self, fmap_h, stride_h):
        """
        calculate output h
        """
        if not fmap_h:
            return None
        if self.pad_mode == "VAR":
            return TilingUtils.icd(fmap_h, stride_h)
        return (fmap_h + self.padu + self.padd - self.k_h_dilation) // stride_h + 1

    def _get_dedy_w(self, fmap_w, stride_w):
        """
        calculate output w
        """
        if not fmap_w:
            return None
        if self.pad_mode == "VAR":
            return TilingUtils.icd(fmap_w, stride_w)
        return (fmap_w + self.padl + self.padr - self.k_w_dilation) // stride_w + 1
