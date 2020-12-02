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
gemm tiling case
"""
import collections
import copy
import math

from te.domain.tiling.get_tiling import get_tiling
from te.lang.base.operation import register_tiling_case
from te.lang.base.operation import get_te_var
from te.platform import get_soc_spec

from te.lang.cce.te_compute.gemm_compute import GEMMParam
from te.lang.dynamic.schedule.cube_tilingcase import TilingSelection
from te.lang.dynamic.schedule.cube_tilingcase import CubeTilingOp
from te.lang.dynamic.schedule.constants import Pattern

K_LEN = 2
M_LEN = 2
N_LEN = 2
UNIT_LEN = 16
DEFAULT_K_VALUE = 32
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
BIT_DIR = {"float32": 16, "int32": 16, "float16": 16, "int8": 32}


def set_var_value(info, tgt_area):
    """
    set range value for tiling

    Parameters
    ----------
    info: ops information

    tgt_area: range value of m k n.

    Returns
    -------
    total info of ops
    """

    info["ha_var_range"] = tgt_area[0]
    info["ca1_var_range"] = tgt_area[1]
    info["cb1_var_range"] = tgt_area[2]
    return info


@register_tiling_case(pattern=Pattern.MAT_MUL)
def calc_matmul(outs, option=None):
    """
    tiling_case func for dynamic shape matmul

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """

    mode = GEMMParam.dynamic_mode
    var_names = ("m", "k", "n")
    tgt_area = [get_te_var(v).get_bound() for v in var_names]
    info = GEMMParam.tiling_info_dict
    info = set_var_value(info, tgt_area)

    tiling_op = MatmulTiling(info, mode)

    tiling_cases = TilingSelection(tiling_op).calc_tiling(tgt_area)
    return tiling_cases


class MatmulTiling(CubeTilingOp):
    def __init__(self, tiling_info, dynamic_mode):
        super().__init__(tiling_info, dynamic_mode)
        self.a_info = self.tiling_info["A_shape"]
        self.b_info = self.tiling_info["B_shape"]
        self.c_info = self.tiling_info["C_shape"]
        self.a_type = self.tiling_info["A_dtype"]
        self.b_type = self.tiling_info["B_dtype"]
        self.c_type = self.tiling_info["C_dtype"]

        self._get_calc_info()
        self.key = ("A_shape", "B_shape")
        self.op_type = "matmul"

    def get_repo_tiling(self):
        """
        get tiling using repository model
        """

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
        self.tiling_info["tiling_type"] = "cost_model_tiling"
        self.a_info[1] = shape[1]
        self.a_info[2] = shape[0]
        self.b_info[0] = shape[1] * self.a_info[4]
        self.b_info[1] = shape[2]

        cost_seeds = get_tiling(self.tiling_info)
        tiling = self._check_and_set_default_tiling(cost_seeds[0])

        return tiling

    def get_tiling_range(self, tiling_in, shape_info):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        shape_info : list, size of m, k, n

        Returns
        -------
        list, range covered for tiling_in
        """

        tiling = self._preprocess_tiling(tiling_in)
        # get shape info (align to 16 or 32)
        m_value, k_value, n_value = shape_info
        # get block_dim value
        block_n, block_m = tiling["block_dim"][1:3]
        # get double buffer value
        is_al1_double = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
        is_bl1_double = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
        # get no full load value
        mal1, kal1, kbl1, nbl1 = 0, 0, 0, 0
        l1_size = get_soc_spec("L1_SIZE")
        if tiling["AL1_shape"]:
            mal1 = tiling["AL1_shape"][1] * tiling["CL0_matrix"][1] * BIT_DIR[self.a_type]
            kal1 = tiling["AL1_shape"][0]
        if tiling["BL1_shape"]:
            nbl1 = tiling["BL1_shape"][1] * tiling["CL0_matrix"][0] * BIT_DIR[self.b_type]
            kbl1 = tiling["BL1_shape"][0]

        def _get_left_l1_size(m_value, n_value, k_value):
            """
            get left size of L1
            """
            al1_size = m_value * k_value * is_al1_double * UNIT_LEN * BIT_DIR[self.a_type] * BIT_RATIO_DICT[self.a_type]
            bl1_size = n_value * k_value * is_bl1_double * UNIT_LEN * BIT_DIR[self.b_type] * BIT_RATIO_DICT[self.b_type]
            return (l1_size - al1_size - bl1_size)

        def _get_max_m_n_value(left_size, k_value, m_value, n_value):
            """
            get the max m&n value
            """
            extend_value = math.floor(
                left_size / (tiling["CL0_matrix"][1] * k_value * is_al1_double * UNIT_LEN * BIT_DIR[self.a_type] *
                             BIT_RATIO_DICT[self.a_type] +
                             tiling["CL0_matrix"][0] * k_value * is_bl1_double * UNIT_LEN * BIT_DIR[self.b_type] *
                             BIT_RATIO_DICT[self.b_type]))
            m_max = m_value + extend_value * block_m * tiling["CL0_matrix"][1]
            n_max = n_value + extend_value * block_n * tiling["CL0_matrix"][0]

            return m_max, n_max

        perf_range = []
        # caculate no full load scen
        if tiling["AL1_shape"] and tiling["BL1_shape"]:
            m_range = [max(1, m_value - M_LEN), m_value + M_LEN]
            k_range = [max(1, k_value - K_LEN), k_value + K_LEN]
            n_range = [max(1, n_value - N_LEN), n_value + N_LEN]
        elif tiling["AL1_shape"] and not tiling["BL1_shape"]:
            m_range = [max(1, m_value - M_LEN), m_value + M_LEN]
            al1_size = mal1 * kal1 * is_al1_double * BIT_RATIO_DICT[self.a_type]
            n_l0c_value = tiling["CL0_matrix"][0]
            n_split_value = math.ceil(math.ceil(n_value / n_l0c_value) / block_n) * n_l0c_value
            k_range = [max(1, k_value - K_LEN), min(k_value + K_LEN, (l1_size - al1_size) //
                                                    (is_bl1_double * n_split_value * BIT_RATIO_DICT[self.b_type] *
                                                     BIT_DIR[
                                                         self.b_type] * UNIT_LEN))]
            n_range = [max(1, n_value - N_LEN), (l1_size - al1_size) //
                       (is_bl1_double * k_range[1] * BIT_RATIO_DICT[self.b_type] * BIT_DIR[
                           self.b_type] * UNIT_LEN * n_l0c_value) * block_n * n_l0c_value]
        elif not tiling["AL1_shape"] and tiling["BL1_shape"]:
            bl1_size = kbl1 * nbl1 * is_bl1_double * BIT_RATIO_DICT[self.b_type]
            n_range = [max(1, n_value - N_LEN), n_value + N_LEN]
            m_l0c_value = tiling["CL0_matrix"][1]
            m_split_value = math.ceil(math.ceil(m_value / m_l0c_value) / block_m) * m_l0c_value
            k_range = [max(1, k_value - K_LEN), min(k_value + K_LEN, (l1_size - bl1_size) //
                                                    (is_al1_double * m_split_value * BIT_RATIO_DICT[self.a_type] *
                                                     BIT_DIR[
                                                         self.a_type] * UNIT_LEN))]
            m_range = [max(1, m_value - M_LEN), (l1_size - bl1_size) //
                       (is_al1_double * k_range[1] * BIT_RATIO_DICT[self.a_type] * BIT_DIR[
                           self.a_type] * UNIT_LEN * m_l0c_value) * m_l0c_value * block_m]
        else:
            actual_m_value = math.ceil(math.ceil(m_value / tiling["CL0_matrix"][1]) / block_m) * tiling["CL0_matrix"][1]
            actual_n_value = math.ceil(math.ceil(n_value / tiling["CL0_matrix"][0]) / block_n) * tiling["CL0_matrix"][0]
            left_size = _get_left_l1_size(actual_m_value, actual_n_value, k_value + K_LEN)

            if left_size >= 0:
                m_max, n_max = _get_max_m_n_value(left_size, k_value + K_LEN, m_value, n_value)
                m_range = [max(1, m_value - M_LEN), min(m_value + M_LEN, m_max)]
                k_range = [max(1, k_value - K_LEN), k_value + K_LEN]
                n_range = [max(1, n_value - N_LEN), min(n_value + N_LEN, n_max)]
            else:
                m_range = [max(1, m_value - M_LEN), m_value]
                k_range = [max(1, k_value - K_LEN), k_value]
                n_range = [max(1, n_value - N_LEN), n_value]

        perf_range = m_range + k_range + n_range
        return perf_range

    def assembly_case(self, tiling, coverage, cnt):
        """
        get the covered info of a tiling

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        coverage : list, size of m, k, n

        cnt: index of tiling

        Returns
        -------
        tiling_case, range covered for tiling
        """

        var_range = collections.OrderedDict()
        var_range['m'] = (int(coverage[0]), int(coverage[1]))
        var_range['k'] = (int(coverage[2]), int(coverage[3]))
        var_range['n'] = (int(coverage[4]), int(coverage[5]))

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range}

    def _check_and_set_default_tiling(self, tiling_in):
        """
        check and set default tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling

        Returns
        -------
        tiling_in
        """

        if tiling_in.get("tiling").get("AL0_matrix")[2] == 32:
            tiling = {}

            a_dtype = self.tiling_info["A_dtype"]
            b_dtype = self.tiling_info["B_dtype"]

            if a_dtype in BIT_DIR.keys():
                k_al1 = BIT_DIR[a_dtype]
                k_al0 = BIT_DIR[a_dtype]
            else:
                # default value 32
                k_al1 = DEFAULT_K_VALUE
                k_al0 = DEFAULT_K_VALUE

            if b_dtype in BIT_DIR.keys():
                k_bl1 = BIT_DIR[b_dtype]
                k_bl0 = BIT_DIR[b_dtype]
            else:
                # default value 32
                k_bl1 = DEFAULT_K_VALUE
                k_bl0 = DEFAULT_K_VALUE

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
        """
        preprocess info, convert tvm var to -1
        """

        self._convert_type(self.a_info, self.b_info)

    def _preprocess_tiling(self, tiling_in):
        """
        preprocess tiling for get tiling range

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        Returns
        -------
        tiling_case, range covered for tiling
        """

        tiling = copy.deepcopy(tiling_in)
        return tiling
