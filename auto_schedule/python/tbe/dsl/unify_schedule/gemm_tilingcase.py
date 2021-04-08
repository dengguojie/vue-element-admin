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
import json
import math
from functools import reduce

from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.context import op_context
from tbe.common import platform as tbe_platform
from tbe.common.utils.errormgr import error_manager_cube
from tbe.dsl.compute.gemm_compute import GEMMComputeParam
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import register_tiling_case

from .cube_tilingcase import CubeTilingOp
from .cube_tilingcase import TilingSelection
from .constants import Pattern


K_LEN = 2
M_LEN = 2
N_LEN = 2
UNIT_LEN = 16
DEFAULT_K_VALUE = 32
INT_32_MAX = 2147483647
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
BIT_DIR = {"float32": 16, "int32": 16, "float16": 16, "int8": 32}
UNKNOWN_DIM = -1
INITIAL_TILING_ID = 10000


def set_var_value(info, target_area):
    """
    set range value for tiling

    Parameters
    ----------
    info: ops information

    target_area: range value of m k n b

    Returns
    -------
    total info of ops
    """
    key_list = ["ha_var_range", "ca1_var_range", "cb1_var_range", "batch_var_range"]
    for index, value in enumerate(target_area):
        info[key_list[index]] = value
    return info


def check_range_value(target_area):
    """
    check range value of target_area to find out None exit

    Parameters
    ----------
    target_area: range value of dymanic elements

    Returns
    -------
    return True means None exit,else return False
    """
    for value in target_area:
        if value[1] is None:
            return True
    return False


def set_default_compile_info(tiling_op, tiling_case, target_area_list):
    """
    add compile info for default case

    Parameters
    ----------
    target_area: range value of dymanic elements

    tiling_case: default tiling, default range

    tiling_op: instance of MatmulTiling

    -------
    """
    add_compile_info("dynamic_mode", tiling_op.dynamic_mode)
    add_compile_info("repo_range", {})
    add_compile_info("repo_seeds", {})

    cost_range = {}
    cost_range[0] = target_area_list
    add_compile_info("cost_range", cost_range)

    if "trans_a" in tiling_op.tiling_info and "trans_b" in tiling_op.tiling_info:
        add_compile_info("attrs", {"transpose_a": tiling_op.tiling_info["trans_a"],
                                   "transpose_b": tiling_op.tiling_info["trans_b"]})

    tiling_blockdim = {}
    for case in tiling_case:
        tiling_blockdim[case['key']] = (case["block_dim"] if "block_dim" in case else
            int(reduce(lambda x, y: x * y, case['tiling_strategy']['block_dim'])))

    add_compile_info("block_dim", tiling_blockdim)


def set_default_tiling_case(target_area, tiling_op):
    """
    when range exit None, set default tiling_case with default elements

    Parameters
    ----------
    target_area: range value of dymanic elements
    tiling_op: instance of MatmulTiling

    Returns
    -------
    default tiling_case: default tiling, default range
    """
    default_tiling_seed = tiling_op._set_default_tiling()
    default_tiling = default_tiling_seed["tiling"]
    target_area_list = []
    for value in target_area:
        if value[1] is None:
            value[1] = INT_32_MAX
        target_area_list += value
    default_seed_shape = [target_area[0][1], target_area[1][1], target_area[2][1]]
    if len(target_area) == 4:
        default_seed_shape.append(target_area[3][1])
    tiling_case = [tiling_op.assembly_case(default_seed_shape, default_tiling, target_area_list, 0)]

    set_default_compile_info(tiling_op, tiling_case, target_area_list)

    return tiling_case


def change_full_load_to_value(tiling_list):
    """
    change tiling [] to concrete values
    """
    transed_tiling_list = []
    for seed in tiling_list:
        seed_batch_value, seed_k_value, seed_m_value = seed["A_shape"][0:3]
        seed_n_value = seed["B_shape"][1]
        tiling_value = seed["tiling"]
        block_batch, block_n, block_m = tiling_value["block_dim"][0:3]
        loc_n_value, loc_m_value = tiling_value["CL0_matrix"][0:2]
        k0_value = seed["A_shape"][4]
        if not tiling_value["AL1_shape"]:
            k_al1 = seed_k_value * k0_value
            multi_m_al1 = math.ceil(math.ceil(seed_m_value / loc_m_value) / block_m)
            tiling_value["AL1_shape"] = [k_al1, multi_m_al1, 1, 1]

        if not tiling_value["BL1_shape"]:
            k_bl1 = seed_k_value * k0_value
            multi_n_bl1 = math.ceil(math.ceil(seed_n_value / loc_n_value) / block_n)
            tiling_value["BL1_shape"] = [k_bl1, multi_n_bl1, 1, 1]

        seed["tiling"] = tiling_value
        transed_tiling_list.append(seed)

    return transed_tiling_list


def _get_kernel_compile_info(tiling_key, compile_info_ori, change_keys, tiling_case):
    """
    get compile info for kernel list

    Parameters
    ----------
    tiling_key: tiling_key
    compile_info_ori: compile info
    change_keys: keys in compile_info_ori to be updated
    tiling_case: tiling case

    Returns
    -------
    dict, compile info
    """
    compile_info = copy.deepcopy(compile_info_ori)
    for key in change_keys:
        content = compile_info_ori[key].get(tiling_key, [])
        compile_info[key] = {tiling_key: content} if len(content) > 0 else {}
    block_dim = reduce(lambda x, y: x * y, tiling_case["tiling_strategy"]["block_dim"])
    compile_info["block_dim"] = {tiling_key: block_dim}
    return compile_info


def _get_kernel_support_info(tiling_case, mode):
    """
    get support info for kernel list

    Parameters
    ----------
    tiling_case: tiling_case
    mode: dynamic mode

    Returns
    -------
    dict, support info, range supported
    """

    order_dict = tiling_case['var_range']
    m_range = [
        (order_dict["m"][0] - 1) * tbe_platform.BLOCK_REDUCE + 1,
        min(INT_32_MAX, order_dict["m"][1] * tbe_platform.BLOCK_REDUCE)
    ]
    k_range = [
        (order_dict["k"][0] - 1) * tbe_platform.BLOCK_REDUCE + 1,
        min(INT_32_MAX, order_dict["k"][1] * tbe_platform.BLOCK_REDUCE)
    ]
    n_range = [
        (order_dict["n"][0] - 1) * tbe_platform.BLOCK_REDUCE + 1,
        min(INT_32_MAX, order_dict["n"][1] * tbe_platform.BLOCK_REDUCE)
    ]
    input0_shape = [UNKNOWN_DIM, UNKNOWN_DIM]
    input1_shape = [UNKNOWN_DIM, UNKNOWN_DIM]
    output_shape = [UNKNOWN_DIM, UNKNOWN_DIM]
    trans_a = GEMMComputeParam.tiling_info_dict["trans_a"]
    trans_b = GEMMComputeParam.tiling_info_dict["trans_b"]
    input0_range = []
    input1_range = []
    output_range = []
    if mode == "dynamic_mknb":
        batch_range = [
            order_dict["batch"][0],
            min(INT_32_MAX, order_dict["batch"][1])
        ]
        input0_shape.append(UNKNOWN_DIM)
        input1_shape.append(UNKNOWN_DIM)
        output_shape.append(UNKNOWN_DIM)
        input0_range.append(batch_range)
        input1_range.append(batch_range)
        output_range.append(batch_range)
    if trans_a:
        input0_range.append(k_range)
        input0_range.append(m_range)
    else:
        input0_range.append(m_range)
        input0_range.append(k_range)
    if trans_b:
        input1_range.append(n_range)
        input1_range.append(k_range)
    else:
        input1_range.append(k_range)
        input1_range.append(n_range)
    output_range.append(m_range)
    output_range.append(n_range)
    support_info = {
        "inputs": [
            {
                "index": 0,
                "tensor": [
                    {
                        "shape": input0_shape,
                        "range": input0_range
                    }
                ]
            },
            {
                "index": 1,
                "tensor": [
                    {
                        "shape": input1_shape,
                        "range": input1_range
                    }
                ]
            }
        ],
        "outputs": [
            {
                "index": 0,
                "tensor": [
                    {
                        "shape": output_shape,
                        "range": output_range
                    }
                ]
            },
        ]
    }
    return support_info


def _set_build_json_info(tiling_cases_list, mode):
    """
    set kernel info to context

    Parameters
    ----------
    tiling_cases_list: tiling case list
    mode: dynamic mode
    """

    kernel_list = []
    context = op_context.get_context()
    compile_info_ori = context.get_compile_info()
    change_keys = [
        "cost_range", "repo_seeds", "repo_range"
    ]
    max_tiling_key = tiling_cases_list[0]["key"]
    for tiling_case in tiling_cases_list:
        tiling_key = tiling_case["key"]
        if tiling_key > max_tiling_key:
            max_tiling_key = tiling_key
        kernel = {
            "supportInfo": _get_kernel_support_info(tiling_case, mode),
            "compileInfo": _get_kernel_compile_info(tiling_key, compile_info_ori, change_keys, tiling_case),
            "implMode": "high_performance"
        }
        kernel_list.append(kernel)
    context.add_build_json_result("kernelList", kernel_list)
    context.add_build_json_result("maxKernelId", max_tiling_key)


def _list_comprehensive(array, factor):
    return list(map(lambda x: (x + factor - 1) // factor, array))


def _calc_tiling_case(mode, target_area, cnt):
    """
    calculate tiling case

    Parameters
    ----------
    mode: dynamic mode
    target_area: range to be compiled
    cnt: initial value of tiling key

    Returns
    -------
    list of dict, each dict for a tiling case
    """

    info = copy.deepcopy(GEMMComputeParam.tiling_info_dict)
    info = set_var_value(info, target_area)

    tiling_op = MatmulTiling(info, mode)

    if check_range_value(target_area):
        return set_default_tiling_case(target_area, tiling_op)

    tiling_cases = TilingSelection(tiling_op, cnt).calc_tiling(target_area)
    return tiling_cases


def _calc_intersection(range_a, range_b):
    """
    calculate intersection of range_a and range_b

    Parameters
    ----------
    range_a: list or tuple
    range_b: list or tuple

    Returns
    -------
    list: intersection of range_a and range_b
    """

    intersection = [max(range_a[0], range_b[0]), min(range_a[1], range_b[1])]
    if intersection[0] > intersection[1]:
        return []
    return intersection


def _calc_tiling_case_with_support_info(missing_support_info_list, mode, tiling_key_cnt, context):
    """
    calculate tiling cases with support info

    Parameters
    ----------
    missing_support_info_list: missing ranges
    mode: dynamic mode
    tiling_key_cnt: initial value of tiling key
    context: context

    Returns
    -------
    list of dict, each dict for a tiling case
    """

    tiling_case = []
    op_name = "Matmul" if mode == "dynamic_mkn" else "BatchMatMul"
    if not isinstance(missing_support_info_list, str):
        error_manager_cube.raise_err_one_para(
            "E62306",
            op_name,
            "invalud missing support info"
        )
    missing_support_info_list = json.loads(missing_support_info_list)
    max_kernel_id = context.get_addition("max_kernel_id")
    if isinstance(max_kernel_id, int) and max_kernel_id > 0:
        tiling_key_cnt = max_kernel_id + 1
    else:
        error_manager_cube.raise_err_one_para(
            "E62306",
            op_name,
            "invalid max_kernel_id"
        )
    dtype = GEMMComputeParam.tiling_info_dict["A_dtype"]
    factor = tbe_platform.BLOCK_REDUCE_INT8 if dtype == "int8" else tbe_platform.BLOCK_REDUCE
    for missing_support_info in missing_support_info_list:
        inputs = missing_support_info["inputs"]
        input_a = inputs[0]
        input_b = inputs[1]
        trans_a = GEMMComputeParam.tiling_info_dict["trans_a"]
        trans_b = GEMMComputeParam.tiling_info_dict["trans_b"]

        offset = 0
        if mode == "dynamic_mknb":
            offset = 1
        m_index = 1 + offset if trans_a else 0 + offset
        k_m_index = 0 + offset if trans_a else 1 + offset
        k_n_index = 1 + offset if trans_b else 0 + offset
        n_index = 0 + offset if trans_b else 1 + offset
        
        m_range = _list_comprehensive(input_a["tensor"][0]["range"][m_index], factor)
        k_m_range = _list_comprehensive(input_a["tensor"][0]["range"][k_m_index], factor)
        k_n_range = _list_comprehensive(input_b["tensor"][0]["range"][k_n_index], factor)
        n_range = _list_comprehensive(input_b["tensor"][0]["range"][n_index], factor)

        k_range = _calc_intersection(k_m_range, k_n_range)
        if not k_range:
            error_manager_cube.raise_err_one_para(
                "E62306",
                op_name,
                "k_range in input1 has no intersection with k_range in input2"
            )
        area = [m_range, k_range, n_range]

        if mode == "dynamic_mknb":
            batch_range_a = input_a["tensor"][0]["range"][0]
            batch_range_b = input_b["tensor"][0]["range"][0]
            batch_range = _calc_intersection(batch_range_a, batch_range_b)
            if not batch_range:
                error_manager_cube.raise_err_one_para(
                    "E62306",
                    op_name,
                    "batch_range in input1 has no intersection with batch_range in input2"
                )
            area.append(batch_range)
        
        tiling_case_part = _calc_tiling_case(mode, area, tiling_key_cnt + len(tiling_case))
        tiling_case += tiling_case_part
    
    return tiling_case


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

    mode = GEMMComputeParam.dynamic_mode
    var_names = {"dynamic_mkn": ("m", "k", "n"), "dynamic_mknb": ("m", "k", "n", "batch")}
    target_area = [get_te_var(v).get_bound() for v in var_names[mode]]

    context = op_context.get_context()
    fuzzy_build = (context.get_build_type() == "fuzzily_build")
    if fuzzy_build:
        missing_support_info_list = context.get_addition("missing_support_info")
        if not missing_support_info_list:
            tiling_case = _calc_tiling_case(mode, target_area, INITIAL_TILING_ID)
        else:
            tiling_case = _calc_tiling_case_with_support_info(missing_support_info_list, 
                                                              mode, 
                                                              INITIAL_TILING_ID, 
                                                              context)
        _set_build_json_info(tiling_case, mode)
        return tiling_case

    return _calc_tiling_case(mode, target_area, INITIAL_TILING_ID)   


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
        self.tiling_info["tiling_type"] = None
        tiling_list = get_tiling(self.tiling_info)
        tiling_list = change_full_load_to_value(tiling_list)
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
        self.a_info[0] = 1
        self.a_info[1] = shape[1]
        self.a_info[2] = shape[0]
        self.b_info[0] = shape[1] * self.a_info[4]
        self.b_info[1] = shape[2]

        cost_seeds = get_tiling(self.tiling_info)
        cost_seeds = change_full_load_to_value(cost_seeds)
        tiling = cost_seeds[0]

        # check whether the tiling is default
        def _check_defualt_tiling(tiling):
            if tiling.get("tiling").get("AL0_matrix")[2] == DEFAULT_K_VALUE:
                return True
            return False

        if _check_defualt_tiling(tiling):
            tiling = self._set_default_tiling()

        return tiling

    def get_tiling_range(self, tiling_in, shape_info):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        shape_info : list, size of m, k, n, align to 16 or 32

        Returns
        -------
        list, range covered for tiling_in
        """

        tiling = self._preprocess_tiling(tiling_in)
        m_value, k_value, n_value = shape_info
        block_n, block_m = tiling["block_dim"][1:3]
        # get double buffer value
        is_al1_double = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
        is_bl1_double = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
        # get no full load value
        mal1, kal1, kbl1, nbl1 = 0, 0, 0, 0
        l1_size = tbe_platform_info.get_soc_spec("L1_SIZE")
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
                                                     BIT_DIR[self.b_type] * UNIT_LEN))]
            n_range = [max(1, n_value - N_LEN), (l1_size - al1_size) //
                       (is_bl1_double * k_range[1] * BIT_RATIO_DICT[self.b_type] *
                       BIT_DIR[self.b_type] * UNIT_LEN * n_l0c_value) * block_n * n_l0c_value]
        elif not tiling["AL1_shape"] and tiling["BL1_shape"]:
            bl1_size = kbl1 * nbl1 * is_bl1_double * BIT_RATIO_DICT[self.b_type]
            n_range = [max(1, n_value - N_LEN), n_value + N_LEN]
            m_l0c_value = tiling["CL0_matrix"][1]
            m_split_value = math.ceil(math.ceil(m_value / m_l0c_value) / block_m) * m_l0c_value
            k_range = [max(1, k_value - K_LEN), min(k_value + K_LEN, (l1_size - bl1_size) //
                                                    (is_al1_double * m_split_value * BIT_RATIO_DICT[self.a_type] *
                                                     BIT_DIR[self.a_type] * UNIT_LEN))]
            m_range = [max(1, m_value - M_LEN), (l1_size - bl1_size) //
                       (is_al1_double * k_range[1] * BIT_RATIO_DICT[self.a_type] *
                       BIT_DIR[self.a_type] * UNIT_LEN * m_l0c_value) * m_l0c_value * block_m]
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

    def assembly_case(self, m_k_n_shape, tiling, coverage, cnt):
        """
        get the covered info of a tiling

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        coverage : list, size of dymanic element

        cnt: index of tiling

        Returns
        -------
        tiling_case, range covered for tiling
        """

        var_range = collections.OrderedDict()

        var_range["m"] = (coverage[0], coverage[1])
        var_range["k"] = (coverage[2], coverage[3])
        var_range["n"] = (coverage[4], coverage[5])
        if self.dynamic_mode == "dynamic_mknb":
            var_range["batch"] = (coverage[6], coverage[7])

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range, "m_k_n_shape": m_k_n_shape}

    def _set_default_tiling(self):
        """
        check and set default tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling

        Returns
        -------
        tiling_in
        """

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
