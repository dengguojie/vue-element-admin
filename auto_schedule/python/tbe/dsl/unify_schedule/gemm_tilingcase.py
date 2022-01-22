#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
import itertools
from functools import reduce
from itertools import product

from tbe.common.platform import intrinsic_check_support
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.context import op_context
from tbe.common import platform as tbe_platform
from tbe.common.utils.errormgr import error_manager_cube

from tbe.dsl.compute.mmad_compute import MatMulComputeParam as GEMMComputeParam1
from tbe.dsl.compute.gemm_integrated_compute import GEMMComputeParam as GEMMComputeParam2
from tbe.dsl.compute.util import int_ceil_div
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import get_context

from .cube_tilingcase import CubeTilingOp
from .cube_tilingcase import MAX_RANGE
from .cube_tilingcase import TilingSelection
from .cube_tilingcase import TilingUtils as utils
from .constants import Pattern


GEMMComputeParam = GEMMComputeParam2
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
# Gerneral schedule pattern: 10000~20000; Aligned schedule pattern: 20000~30000.
# There is a related flag in file: gemm.cc which also uses this offset.
INITIAL_TILING_ID = 10000
ALIGNED_TILING_ID_OFFSET = 10000
SHAPE_BMKN_LEN = 4
BANK_THRESHOLD = 64
BANK_GAP = 16


def _get_bit(input_dtype):
    """
    get bit ratio of input dtype
    """
    return BIT_RATIO_DICT.get(input_dtype, 1)


def _get_reduce(input_dtype):
    """
    get reduce axis of input dtype
    """
    return BIT_DIR.get(input_dtype, DEFAULT_K_VALUE)


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
    info["none_range_area"] = None in sum(target_area, [])
    return info


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
    cost_range[INITIAL_TILING_ID] = target_area_list
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
    default_tiling_seed = tiling_op.set_default_tiling()
    default_tiling = default_tiling_seed["tiling"]
    target_area_list = []
    for value in target_area:
        if value[1] is None:
            value[1] = INT_32_MAX
        target_area_list += value
    default_seed_shape = [target_area[0][1], target_area[1][1], target_area[2][1]]
    if len(target_area) == 4:
        default_seed_shape.append(target_area[3][1])
    tiling_case = [tiling_op.assembly_case(default_seed_shape, default_tiling, target_area_list, INITIAL_TILING_ID)]

    set_default_compile_info(tiling_op, tiling_case, target_area_list)
    return tiling_case


def _is_fuzzily_build():
    """
    check fuzzily build flag
    """
    context = op_context.get_context()
    return context.get_build_type() == "fuzzily_build"


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
    input0_range = []
    input1_range = []
    output_range = []
    if mode == "dynamic_mknb":
        batch_range = [
            order_dict["batch"][0],
            min(INT_32_MAX, order_dict["batch"][1])
        ]
        for item in op_context.get_context().get_addition("batch_range_x1"):
            input0_shape.append(UNKNOWN_DIM)
            input0_range.append(item)
        for item in op_context.get_context().get_addition("batch_range_x2"):
            input1_shape.append(UNKNOWN_DIM)
            input1_range.append(item)
        output_shape.append(UNKNOWN_DIM)
        output_range.append(batch_range)
    if GEMMComputeParam.tiling_info_dict["trans_a"]:
        input0_range.append(k_range)
        input0_range.append(m_range)
    else:
        input0_range.append(m_range)
        input0_range.append(k_range)
    if GEMMComputeParam.tiling_info_dict["trans_b"]:
        input1_range.append(n_range)
        input1_range.append([1, INT_32_MAX])
    else:
        input1_range.append([1, INT_32_MAX])
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
    if tiling_op.use_default_tiling_case(target_area):
        return set_default_tiling_case(target_area, tiling_op)

    tiling_cases = TilingSelection(tiling_op, cnt).calc_tiling(target_area)
    return tiling_cases


def _calc_intersection(range_a, range_b, op_name="MatMul", param_name="k_range"):
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
        error_manager_cube.raise_err_one_para(
            "E62306",
            op_name,
            "{} in input1 has no intersection with it in input2".format(param_name)
        )
    return intersection


def _calc_batch_range(range_x1, range_x2):
    """
    get range of batch
    """
    batch_range = [1, 1]
    range_x = []
    if range_x2:
        if len(range_x1) != len(range_x2):
            error_manager_cube.raise_err_one_para(
                "E62306",
                "BatchMatMul",
                "the batch length of x1 should be equal to x2"
            )
        for range_mem1, range_mem2 in zip(range_x1, range_x2):
            range_ins = _calc_intersection(range_mem1, range_mem2, "BatchMatMul", "batch_range")
            range_x.append(range_ins)
    else:
        range_x = range_x1

    for range_mem in range_x:
        if range_mem[1] is None:
            batch_range = [1, None]
            break
        batch_range[0] = min(batch_range[0] * range_mem[0], INT_32_MAX)
        batch_range[1] = min(batch_range[1] * range_mem[1], INT_32_MAX)

    return batch_range


def _cal_area(missing_support_info, mode, op_name):
    """
    cal the area of matmul or batchmatmul
    """
    factor = tbe_platform.BLOCK_REDUCE
    if GEMMComputeParam.tiling_info_dict["A_dtype"] == "int8":
        factor = tbe_platform.BLOCK_REDUCE_INT8
    inputs = missing_support_info["inputs"]
    m_index, k_m_index = (-1, -2) if GEMMComputeParam.tiling_info_dict["trans_a"] else (-2, -1)
    k_n_index, n_index = (-1, -2) if GEMMComputeParam.tiling_info_dict["trans_b"] else (-2, -1)

    m_range = _list_comprehensive(inputs[0]["tensor"][0]["range"][m_index], factor)
    k_m_range = _list_comprehensive(inputs[0]["tensor"][0]["range"][k_m_index], factor)
    k_n_range = _list_comprehensive(inputs[1]["tensor"][0]["range"][k_n_index], factor)
    n_range = _list_comprehensive(inputs[1]["tensor"][0]["range"][n_index], factor)

    k_range = _calc_intersection(k_m_range, k_n_range, op_name, "k_range")
    area = [m_range, k_range, n_range]

    if mode == "dynamic_mknb":
        batch_range_a = inputs[0]["tensor"][0]["range"][:-2]
        batch_range_b = inputs[1]["tensor"][0]["range"][:-2]
        batch_range = _calc_batch_range(batch_range_a, batch_range_b)
        area.append(batch_range)

    return area


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
    for missing_support_info in missing_support_info_list:
        area = _cal_area(missing_support_info, mode, op_name)
        tiling_case += _calc_tiling_case(mode, area, tiling_key_cnt + len(tiling_case))

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
    if intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
        global GEMMComputeParam
        GEMMComputeParam = GEMMComputeParam1

    mode = GEMMComputeParam.dynamic_mode
    # The variables is named x_ori in ND format, otherwise named x
    m_name = GEMMComputeParam.m_var_name
    k_name = GEMMComputeParam.k_var_name
    n_name = GEMMComputeParam.n_var_name

    var_names = {"dynamic_mkn": (m_name, k_name, n_name),
                 "dynamic_mknb": (m_name, k_name, n_name, "batch")}

    target_area = copy.deepcopy([get_te_var(v).get_bound() for v in var_names.get(mode)])
    # process target_area result in ND mode. make it M1/ K1 / N1
    if GEMMComputeParam.format_a == "ND":
        target_area[0][0] = math.ceil(target_area[0][0] / UNIT_LEN)
        target_area[0][1] = None if target_area[0][1] is None else math.ceil(target_area[0][1] / UNIT_LEN)
        target_area[1][0] = math.ceil(target_area[1][0] / UNIT_LEN)
        target_area[1][1] = None if target_area[1][1] is None else math.ceil(target_area[1][1] / UNIT_LEN)
    if GEMMComputeParam.format_b == "ND":
        target_area[2][0] = math.ceil(target_area[2][0] / UNIT_LEN)
        target_area[2][1] = None if target_area[2][1] is None else math.ceil(target_area[2][1] / UNIT_LEN)

    add_compile_info("format_a", GEMMComputeParam.format_a)
    add_compile_info("format_b", GEMMComputeParam.format_b)

    context = op_context.get_context()
    if _is_fuzzily_build():
        # for batchmatmul, supported scenes are list as below:
        # (1). input1 has one or more batch dims and input2 has no batch
        # (2). input1 and input2 has same batch dims and no need to broadcast
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
    tiling_cases = _calc_tiling_case(mode, target_area, INITIAL_TILING_ID)
    # Generate Aligned schedule for ND input
    if GEMMComputeParam.format_a == "ND" and GEMMComputeParam.format_b == "ND":
        tiling_cases = _generate_aligned_tilingcase(tiling_cases)
    return tiling_cases


def _generate_aligned_tilingcase(tiling_cases):
    # For aligned schedule pattern in MatMul/BatchMatMul
    case_length = len(tiling_cases)
    aligned_tiling_cases = []
    if case_length >= ALIGNED_TILING_ID_OFFSET:
        error_manager_cube.raise_err_one_para(
            "E62306",
            "MatMul/BatchMatMul",
            "The compiled kernel number exceeds 10000."
        )
    for tiling in tiling_cases:
        # cache_tiling doesn't modify tilingkey
        if tiling["tiling_strategy"].get("attach_at_flag"):
            return tiling_cases
        aligned_tiling = copy.deepcopy(tiling)
        # The general tilingkey is '1xxxx' and adding this offset makes it '2xxxx'
        aligned_tiling["key"] = aligned_tiling["key"] + ALIGNED_TILING_ID_OFFSET
        aligned_tiling.get("tiling_strategy")["schedule_pattern"] = "Aligned"
        aligned_tiling_cases.append(aligned_tiling)
    tiling_cases += aligned_tiling_cases
    return tiling_cases


class MatmulTiling(CubeTilingOp):
    """
    the cube tiling op of matmul
    """
    DEFAULT_COMPILE_TIME = 4096
    GEAR_M_N = [1, 4, 8, 16, 32, 64, 128, 192, 256, 512, 768, 1024]
    GEAR_BATCH = [1, 2, 4, 8, 16, 32]
    NONE_RANGE_M_N = [1, 16, 32, 64, 128, 256]
    NONE_RANGE_BATCH = [1, 16, 32]

    def __init__(self, tiling_info, dynamic_mode):
        super().__init__(tiling_info, dynamic_mode)
        self.a_info = self.tiling_info["A_shape"]
        self.b_info = self.tiling_info["B_shape"]
        self.c_info = self.tiling_info["C_shape"]
        self.a_type = self.tiling_info["A_dtype"]
        self.b_type = self.tiling_info["B_dtype"]
        self.c_type = self.tiling_info["C_dtype"]
        self.format_a = GEMMComputeParam.format_a
        self.format_b = GEMMComputeParam.format_b
        self.bias_flag = self.tiling_info["bias_flag"]
        self.none_range_area = self.tiling_info["none_range_area"]
        self.use_cache_tiling = (not self.bias_flag and self.none_range_area and
                                 "Ascend910" in tbe_platform_info.get_soc_spec("SOC_VERSION"))

        if intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
            self.use_cache_tiling = False
        get_context().add("_use_cache_tiling", self.use_cache_tiling)
        self._get_calc_info()
        self.key = ("A_shape", "B_shape")
        self.op_type = "matmul"

    def get_repo_tiling(self):
        """
        get tiling using repository model
        """
        self.tiling_info["tiling_type"] = None
        tiling_list = get_tiling(self.tiling_info)
        tiling_list = self.change_full_load_to_value(tiling_list)
        return tiling_list

    @staticmethod
    def change_full_load_to_value(tiling_list):
        """
        change tiling [] to concrete values
        """
        transed_tiling_list = []
        for seed in tiling_list:
            seed_k_value, seed_m_value = seed.get("A_shape")[1:3]
            seed_n_value = seed.get("B_shape")[1]
            tiling_value = seed.get("tiling")
            loc_n_value, loc_m_value = tiling_value.get("CL0_matrix")[0:2]
            k0_value = seed.get("A_shape")[4]
            if not tiling_value.get("AL1_shape"):
                k_al1 = seed_k_value * k0_value
                multi_m_al1 = math.ceil(math.ceil(seed_m_value / loc_m_value) / tiling_value.get("block_dim")[2])
                tiling_value["AL1_shape"] = [k_al1, multi_m_al1, 1, 1]

            if not tiling_value.get("BL1_shape"):
                k_bl1 = seed_k_value * k0_value
                multi_n_bl1 = math.ceil(math.ceil(seed_n_value / loc_n_value) / tiling_value.get("block_dim")[1])
                tiling_value["BL1_shape"] = [k_bl1, multi_n_bl1, 1, 1]

            seed["tiling"] = tiling_value
            transed_tiling_list.append(seed)

        return transed_tiling_list

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
        if len(shape) == SHAPE_BMKN_LEN:
            self.a_info[0] = shape[3]
        else:
            self.a_info[0] = 1
        self.a_info[1] = shape[1]
        self.a_info[2] = shape[0]
        self.b_info[0] = shape[1] * self.a_info[4]
        self.b_info[1] = shape[2]

        cost_seeds = get_tiling(self.tiling_info)
        cost_seeds = self.change_full_load_to_value(cost_seeds)
        tiling = cost_seeds[0]

        tiling.get("tiling")["attach_same_to_static"] = False
        return tiling

    def get_cache_tiling(self):
        '''
        according to size in l1, generate 9 kind of templates, each subdivided into 132 different
        templates as follows templates according to size in l1 sub template
        --------------------------------------------|-----
        al1 @l0c and bl1 @l0c                       | 48
        al1 @l0c and bl1 @ddr                       | 16
        al1 @l0c and bl1 full load                  | 8
        al1 @ddr and bl1 @l0c                       | 16
        al1 @ddr and bl1 @ddr                       | 16
        al1 @ddr and bl1 full load                  | 8
        al1 full load and bl1 @l0c                  | 8
        al1 full load and bl1 @ddr                  | 8
        al1 full load and bl1 full load             | 4

        Returns
        ----------
        cache_tiling_all: list, include 132 different tiling templates
        '''
        # add compile_info
        info_dict = self.tiling_info
        bias_flag = info_dict.get("bias_flag")
        nd_flag = GEMMComputeParam.format_a == "ND" and GEMMComputeParam.format_b == "ND"
        add_compile_info("binary_mode_flag", True)
        add_compile_info("binary_attrs", {"bias_flag": bias_flag,
                                          "nd_flag": nd_flag})
        # get cache_tiling
        cache_tiling_all = {}
        (al1_pb, bl1_pb, l0c_pb, abkl1_attach, al1_attach_flag,
        bl1_attach_flag, min_kl1_cmp_kl0, aub_multi_flag, bub_multi_flag) = (
            [utils.DB_OFF, utils.DB_ON], [utils.DB_OFF, utils.DB_ON], [utils.DB_OFF, utils.DB_ON],
            [utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL, utils.ATTACH_LESS],
            [utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL, utils.ATTACH_LESS],
            [utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL, utils.ATTACH_LESS],
            [0, 1], [utils.ABUB_NOT_FULL_LOAD, utils.ABUB_INNER_FULL_LOAD, utils.ABUB_FULL_LOAD],
            [utils.ABUB_NOT_FULL_LOAD, utils.ABUB_INNER_FULL_LOAD, utils.ABUB_FULL_LOAD])
        if nd_flag:
            l1_choice = list(
                product(al1_pb, bl1_pb, l0c_pb, abkl1_attach, al1_attach_flag,
                        bl1_attach_flag, min_kl1_cmp_kl0, aub_multi_flag, bub_multi_flag))
        else:
            l1_choice = list(
                product(al1_pb, bl1_pb, l0c_pb, abkl1_attach, al1_attach_flag, bl1_attach_flag, min_kl1_cmp_kl0))
        for choice in l1_choice:
            cache_tiling = {
                'block_dim': [-1, -1, -1, 1],
                'AL0_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'BL0_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'CL0_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'CUB_matrix': [-1, -1, utils.CUBE_SIZE, utils.CUBE_SIZE, 1, 1],
                'BUB_shape': [-1, -1, 1, 1],
                'AL1_shape': [-1, -1, 1, 1], 'BL1_shape': [-1, -1, 1, 1],
                'AUB_shape': [-1, -1, 1, 1],
                'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'batch_bef_group_flag': 0,
                'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0,
                'AUB_channel_wise_flag': None, 'BUB_channel_wise_flag': None, 'CUB_channel_wise_flag': None,
                'manual_pingpong_buffer': {'AUB_pbuffer': utils.DB_ON, 'BUB_pbuffer': utils.DB_ON,
                'AL1_pbuffer': utils.DB_ON, 'BL1_pbuffer': utils.DB_ON,
                'AL0_pbuffer': utils.DB_ON, 'BL0_pbuffer': utils.DB_ON, 'CL0_pbuffer': utils.DB_ON,
                'CUB_pbuffer': utils.DB_ON, 'UBG_pbuffer': utils.DB_OFF},
                'attach_at_flag': {'cub_attach_flag': utils.ATTACH_LESS,
                'cl0_attach_flag': utils.ATTACH_LARGE, 'al0_attach_flag': utils.ATTACH_LESS,
                'bl0_attach_flag': utils.ATTACH_LESS,
                'al1_attach_flag': -1, 'bl1_attach_flag': -1, 'aub_attach_flag': utils.ATTACH_LESS,
                'abkl1_attach_flag': -1, 'aub_multi_flag': -1, 'bub_multi_flag': -1}
            }

            # if bl1 attach at l0c, nbl1, should be 1
            if choice[5] == utils.ATTACH_LESS:
                cache_tiling.get("BL1_shape")[1] = 1
            # al1 attach at l0c
            if choice[4] == utils.ATTACH_LESS:
                # if al1 attach at l0c, mal1 should be 1
                cache_tiling.get('AL1_shape')[1] = 1

            # al1 full load
            invalid_choice = (choice[4] == utils.ATTACH_FULL_LOAD) and (
                (choice[5] in (utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL) and choice[3] != 0) or
                (choice[5] == 2 and choice[3] != 1))

            # al1 attach at c_ddr
            invalid_choice = invalid_choice or (choice[4] == utils.ATTACH_EQUAL and (
                (choice[5] in (utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL) and
                choice[3] != 0) or (choice[5] == utils.ATTACH_LESS and choice[3] != 1)))

            # if al1 attach at l0c and full load in l1 buffer, there is no need to open double buffer
            invalid_choice = invalid_choice or (choice[4] == utils.ATTACH_LESS and
                (choice[5] in (utils.ATTACH_FULL_LOAD, utils.ATTACH_EQUAL) and choice[3] != 2))
            if invalid_choice:
                continue

            cache_tiling.get('manual_pingpong_buffer')['AL1_pbuffer'] = choice[0]
            cache_tiling.get('manual_pingpong_buffer')['BL1_pbuffer'] = choice[1]
            cache_tiling.get('manual_pingpong_buffer')['CL0_pbuffer'] = choice[2]
            cache_tiling.get('attach_at_flag')['abkl1_attach_flag'] = choice[3]
            cache_tiling.get('attach_at_flag')['al1_attach_flag'] = choice[4]
            cache_tiling.get('attach_at_flag')['bl1_attach_flag'] = choice[5]
            cache_tiling.get('attach_at_flag')['min_kl1_cmp_kl0'] = choice[6]
            if nd_flag:
                cache_tiling.get('attach_at_flag')['aub_multi_flag'] = choice[7]
                cache_tiling.get('attach_at_flag')['bub_multi_flag'] = choice[8]
                cache_tiling["schedule_pattern"] = "Aligned"
            name = int(''.join((str(i) for i in choice)))
            cache_tiling_all[name] = [[], cache_tiling, []]

        return cache_tiling_all

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

        tiling = self.preprocess_tiling(tiling_in)
        m_value, k_value, n_value = shape_info[0:3]
        block_n, block_m = tiling.get("block_dim")[1:3]
        k_l0a = tiling.get("AL0_matrix")[1]
        n_l0c_value, m_l0c_value = tiling.get("CL0_matrix")[0:2]
        # get double buffer value
        is_al1_double = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
        is_bl1_double = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
        # get no full load value
        mal1, kal1, kbl1, nbl1 = 0, 0, 0, 0
        l1_size = tbe_platform_info.get_soc_spec("L1_SIZE")
        if tiling.get("AL1_shape"):
            mal1 = tiling.get("AL1_shape")[1] * m_l0c_value * _get_reduce(self.a_type)
            kal1 = tiling.get("AL1_shape")[0]
        if tiling.get("BL1_shape"):
            nbl1 = tiling.get("BL1_shape")[1] * n_l0c_value * _get_reduce(self.b_type)
            kbl1 = tiling.get("BL1_shape")[0]

        def _get_left_l1_size(m_value, n_value, mk_value, nk_value):
            """
            get left size of L1
            """
            al1_shape = m_value * mk_value * UNIT_LEN * _get_reduce(self.a_type)
            al1_size = al1_shape * is_al1_double * _get_bit(self.a_type)
            bl1_shape = n_value * nk_value * UNIT_LEN * _get_reduce(self.b_type)
            bl1_size = bl1_shape * is_bl1_double * _get_bit(self.b_type)
            return l1_size - al1_size - bl1_size

        def _get_max_m_n_value(left_size, k_value, m_value, n_value):
            """
            get the max m&n value
            """
            extend_value = math.floor(
                left_size / (m_l0c_value * k_value * is_al1_double * UNIT_LEN * _get_reduce(self.a_type) *
                             _get_bit(self.a_type) +
                             n_l0c_value * k_value * is_bl1_double * UNIT_LEN * _get_reduce(self.b_type) *
                             _get_bit(self.b_type)))
            m_max = m_value + extend_value * block_m * m_l0c_value
            n_max = n_value + extend_value * block_n * n_l0c_value

            return m_max, n_max

        perf_range = []

        if not tiling.get("attach_same_to_static"):
            m_range = [max(1, m_value - M_LEN), m_value + M_LEN]
            k_range = [max(1, k_value - K_LEN), k_value + K_LEN]
            n_range = [max(1, n_value - N_LEN), n_value + N_LEN]
            perf_range = m_range + k_range + n_range
            return perf_range

        # get full load flag
        al1_full_load_flag = True
        bl1_full_load_flag = True
        bl1_k_full_load_flag = True
        al1_k_full_load_flag = True

        if tiling.get("AL1_shape"):
            al1_m_full_load_flag = tiling.get("AL1_shape")[1] * m_l0c_value * block_m == m_value
            al1_k_full_load_flag = tiling.get("AL1_shape")[0] == (k_value * _get_reduce(self.a_type))
            al1_full_load_flag = al1_k_full_load_flag and al1_m_full_load_flag

        if tiling.get("BL1_shape"):
            bl1_n_full_load_flag = tiling.get("BL1_shape")[1] * n_l0c_value * block_n == n_value
            bl1_k_full_load_flag = tiling.get("BL1_shape")[0] == (k_value * _get_reduce(self.a_type))
            bl1_full_load_flag = bl1_k_full_load_flag and bl1_n_full_load_flag

        if not al1_full_load_flag and not bl1_full_load_flag:
            m_range = [max(1, m_value - M_LEN), m_value + M_LEN]
            k_range = [max(1, k_value - K_LEN), k_value + K_LEN]
            n_range = [max(1, n_value - N_LEN), n_value + N_LEN]
            l1size_left = _get_left_l1_size(mal1 // UNIT_LEN, nbl1 // UNIT_LEN, kal1 // UNIT_LEN, kbl1 // UNIT_LEN)
            k_max = K_LEN
            if (al1_k_full_load_flag and not bl1_k_full_load_flag):
                k_max = l1size_left // (mal1 * is_al1_double * _get_reduce(self.a_type) *\
                        _get_bit(self.a_type) * k_l0a) * k_l0a
            if (not al1_k_full_load_flag and bl1_k_full_load_flag):
                k_max = l1size_left // (nbl1 * is_bl1_double * _get_reduce(self.b_type) *\
                        _get_bit(self.b_type) * k_l0a) * k_l0a
            if (al1_k_full_load_flag and bl1_k_full_load_flag):
                k_max = l1size_left // ((nbl1 * is_bl1_double + mal1 * is_al1_double) * _get_reduce(self.a_type) *\
                        _get_bit(self.a_type) * k_l0a) * k_l0a
            k_max = math.floor(k_max)
            k_range = [max(1, k_value - K_LEN), min(k_value + K_LEN, k_value + k_max)]
        elif bl1_full_load_flag and not al1_full_load_flag:
            m_range = [max(1, m_value - M_LEN), m_value + M_LEN]
            al1_size = mal1 * kal1 * is_al1_double * _get_bit(self.a_type)
            n_split_value = math.ceil(math.ceil(n_value / n_l0c_value) / block_n) * n_l0c_value
            if al1_k_full_load_flag:
                m_split_value = mal1 // _get_reduce(self.a_type)
                k_range_min = max(1, k_value - K_LEN)
                k_range_max = min(k_value + K_LEN,
                                  l1_size // ((is_bl1_double * n_split_value + is_al1_double * m_split_value) *
                                  _get_bit(self.b_type) * _get_reduce(self.b_type) * UNIT_LEN * k_l0a) * k_l0a)
            else:
                k_range_min = max(1, k_value - K_LEN)
                k_range_max = min(k_value + K_LEN,
                                  (l1_size - al1_size) // (is_bl1_double * n_split_value * _get_bit(self.b_type) *\
                                  _get_reduce(self.b_type) * UNIT_LEN * k_l0a) * k_l0a)
            k_range = [k_range_min, k_range_max]
            k_max_align_value = math.ceil(k_range[1] / k_l0a) * k_l0a
            n_range = [max(1, n_value - N_LEN), min(n_value + N_LEN, (l1_size - al1_size) //
                       (is_bl1_double * k_max_align_value * _get_bit(self.b_type) *
                       _get_reduce(self.b_type) * UNIT_LEN * n_l0c_value) * block_n * n_l0c_value)]
        elif al1_full_load_flag and not bl1_full_load_flag:
            bl1_size = kbl1 * nbl1 * is_bl1_double * _get_bit(self.b_type)
            n_range = [max(1, n_value - N_LEN), n_value + N_LEN]
            m_split_value = math.ceil(math.ceil(m_value / m_l0c_value) / block_m) * m_l0c_value
            if bl1_k_full_load_flag:
                n_split_value = nbl1 // _get_reduce(self.a_type)
                k_range = [max(1, k_value - K_LEN), min(k_value + K_LEN, l1_size // ((is_bl1_double * n_split_value +\
                                                    is_al1_double * m_split_value) * _get_bit(self.b_type) *
                                                     _get_reduce(self.b_type) * UNIT_LEN * k_l0a) * k_l0a)]
            else:
                k_range = [max(1, k_value - K_LEN), min(k_value + K_LEN, (l1_size - bl1_size) //
                                                    (is_al1_double * m_split_value * _get_bit(self.a_type) *
                                                     _get_reduce(self.a_type) * UNIT_LEN * k_l0a) * k_l0a)]
            k_max_align_value = math.ceil(k_range[1] / k_l0a) * k_l0a
            m_range = [max(1, m_value - M_LEN), min(m_value + M_LEN, (l1_size - bl1_size) //
                       (is_al1_double * k_max_align_value * _get_bit(self.a_type) *
                       _get_reduce(self.a_type) * UNIT_LEN * m_l0c_value) * m_l0c_value * block_m)]
        elif al1_full_load_flag and bl1_full_load_flag:
            actual_m_value = math.ceil(math.ceil(m_value / m_l0c_value) / block_m) * m_l0c_value
            actual_n_value = math.ceil(math.ceil(n_value / n_l0c_value) / block_n) * n_l0c_value
            left_size = _get_left_l1_size(actual_m_value, actual_n_value, k_value, k_value)

            if left_size >= 0:
                m_max, n_max = _get_max_m_n_value(left_size, k_value, m_value, n_value)
                m_range = [max(1, m_value - M_LEN), min(m_value + M_LEN, m_max)]
                n_range = [max(1, n_value - N_LEN), min(n_value + N_LEN, n_max)]
                m_align_value = math.ceil(math.ceil(m_range[1] / m_l0c_value) / block_m) * m_l0c_value
                n_align_value = math.ceil(math.ceil(n_range[1] / n_l0c_value) / block_n) * n_l0c_value
                k_max = math.floor(l1_size // ((is_al1_double * m_align_value +\
                                   is_bl1_double * n_align_value) * _get_bit(self.b_type) *\
                                   _get_reduce(self.b_type) * UNIT_LEN * k_l0a) * k_l0a)
                k_range = [max(1, k_value - K_LEN), min(k_value + K_LEN, k_max)]
            else:
                m_range = [max(1, m_value - M_LEN), m_value]
                k_range = [max(1, k_value - K_LEN), k_value]
                n_range = [max(1, n_value - N_LEN), n_value]
        k_range = self.update_range_by_ub(k_range, tiling, shape_info)
        perf_range = m_range + k_range + n_range
        return perf_range

    def _get_bank_space(self, out_axis, in_axis, trans_flag, db_buf):
        """
        get space to solve bank_conflict in aub/bub
        """
        dtype_size = _get_bit(self.a_type)
        out_axis, in_axis = [in_axis, out_axis] if trans_flag else [out_axis, in_axis]
        bank_space = 0
        if in_axis % BANK_THRESHOLD == 0:
            bank_space = out_axis * BANK_GAP * db_buf * dtype_size
        return bank_space

    def update_range_by_ub(self, k_range, tiling, shape_info):
        """
        aub/bub attach at c_gm, k_range should be updated
        """
        block_in = tbe_platform_info.BLOCK_IN
        block_reduce = tbe_platform_info.BLOCK_REDUCE
        block_out = tbe_platform_info.BLOCK_OUT

        cub_n1, cub_m1 = tiling.get("CUB_matrix")[0:2]
        aub_db = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
        bub_db = tiling.get("manual_pingpong_buffer").get("BUB_pbuffer")
        cub_db = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")

        ub_size = tbe_platform_info.get_soc_spec("UB_SIZE")
        aub_k, aub_m1 = tiling.get("AUB_shape")[0:2] if tiling.get("AUB_shape") else [0, 0]
        bub_k, bub_n1 = tiling.get("BUB_shape")[0:2] if tiling.get("BUB_shape") else [0, 0]
        c_fused_num = self.tiling_info.get("fused_double_operand_num")
        a_fused_num = self.tiling_info.get("padl") // 10
        b_fused_num = self.tiling_info.get("padr") // 10

        if tiling.get("AL1_shape"):
            al1_k, al1_m1 = tiling.get("AL1_shape")[0:2]
            al1_m1 = al1_m1 * tiling.get("CL0_matrix")[1]
        else:
            al1_k, al1_m1 = shape_info[1] * block_reduce, int_ceil_div(shape_info[0], tiling.get("block_dim")[2])
        if tiling.get("BL1_shape"):
            bl1_k, bl1_n1 = tiling.get("BL1_shape")[0:2]
            bl1_n1 = bl1_n1 * tiling.get("CL0_matrix")[0]
        else:
            bl1_k, bl1_n1 = shape_info[1] * block_reduce, int_ceil_div(shape_info[2], tiling.get("block_dim")[1])
        aub_k_full_load = (tiling.get("AUB_shape") and [aub_k, aub_m1] == [al1_k, al1_m1]
                           and al1_k // block_reduce == shape_info[1])
        bub_k_full_load = (tiling.get("BUB_shape") and [bub_k, bub_n1] == [bl1_k, bl1_n1]
                           and bl1_k // block_reduce == shape_info[1])
        # aub/bub k_factor same as tiling
        if not aub_k_full_load and not bub_k_full_load:
            return k_range

        cub_space = (cub_n1 * cub_m1 * block_in * block_out * (c_fused_num + 1) *
                     _get_bit(self.c_type) * cub_db)
        aub_space = (aub_m1 * block_in * aub_k * (a_fused_num + 1) *
                     _get_bit(self.a_type) * aub_db)
        bub_space = (bub_n1 * block_out * bub_k * (b_fused_num + 1) *
                     _get_bit(self.b_type) * bub_db)
        # cub bank_conflict add space, same as schedule
        cub_add_space = cub_n1 * block_out * _get_bit(self.c_type) * cub_db
        remaining_space = ub_size - cub_space - aub_space - bub_space - cub_add_space

        trans_a_flag = GEMMComputeParam.tiling_info_dict["trans_a"]
        trans_b_flag = GEMMComputeParam.tiling_info_dict["trans_b"]
        # Both here and the calculation in schedule do not support attaching aub/bub at c_gm to solve bank conflict.
        if aub_k_full_load and bub_k_full_load:
            # the max extra k1 when compute at c_gm and k_factor is not k_aub/k_bub
            k_len = math.floor(remaining_space //
                               ((aub_m1 * (a_fused_num + 1) * aub_db + bub_n1 * (b_fused_num + 1) * bub_db) *
                               block_in * block_reduce * _get_bit(self.a_type)))
        elif aub_k_full_load:
            # bub not full load, may need solve bank conflict.
            bub_add_space = self._get_bank_space(bub_k, bub_n1 * block_out, trans_b_flag, bub_db)
            remaining_space = remaining_space - bub_add_space
            k_len = math.floor(remaining_space //
                               (aub_m1 * (a_fused_num + 1) * aub_db * block_in * block_reduce *
                                _get_bit(self.a_type)))
        else:
            # aub not full load, may need solve bank conflict.
            aub_add_space = self._get_bank_space(aub_m1 * block_in, aub_k, trans_a_flag, aub_db)
            remaining_space = remaining_space - aub_add_space
            k_len = math.floor(remaining_space //
                               (bub_n1 * (b_fused_num + 1) * bub_db * block_out * block_reduce *
                               _get_bit(self.b_type)))
        if remaining_space <= 0:
            return [k_range[0], shape_info[1]]
        k_max = min(shape_info[1] + k_len, k_range[1])
        for k in range(shape_info[1], k_max + 1):
            # in dynamic, specially update k_bound of ub when k_axis is not divided by factor
            if int_ceil_div((k + 1), tiling.get("AL0_matrix")[1]) * tiling.get("AL0_matrix")[1] * \
                block_in > k_max:
                k_max = k
                break
        return [k_range[0], k_max]

    def get_repo_candidate(self, seed, target_area):
        """
        update repository with tiling range
        """
        def _correct_seed_range():
            # dynamic_mknb or dynamic_mkn only compare m, k, n value
            funcs = (max, min, max, min, max, min)
            return [func(ta, sa) for func, ta, sa in zip(funcs, range_area, seed_range_ori)]

        range_area = tuple(target_area[0] + target_area[1] + target_area[2])
        seed_batch_value, seed_k_value, seed_m_value = seed["A_shape"][0:3]
        seed_n_value = seed["B_shape"][1]
        seed_shape_info = [seed_m_value, seed_k_value, seed_n_value]
        seed_range_ori = self.get_tiling_range(seed["tiling"], seed_shape_info)
        seed_range = _correct_seed_range()
        if self.dynamic_mode == "dynamic_mknb":
            seed_range += target_area[3]
            seed_shape_info += [seed_batch_value]
        candidate = [seed_range, seed["tiling"], seed_shape_info]
        return candidate

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

        m_name = GEMMComputeParam.m_var_name
        k_name = GEMMComputeParam.k_var_name
        n_name = GEMMComputeParam.n_var_name

        block_in = GEMMComputeParam.block_in
        block_out = GEMMComputeParam.block_out
        block_reduce = GEMMComputeParam.block_reduce
        if coverage:
            if GEMMComputeParam.format_a == "ND":
                m_range = (coverage[0] * block_in, min(coverage[1] * block_in, INT_32_MAX))
                k_range = (coverage[2] * block_reduce, min(coverage[3] * block_reduce, INT_32_MAX))
            else:
                m_range = (coverage[0], coverage[1])
                k_range = (coverage[2], coverage[3])
            if GEMMComputeParam.format_b == "ND":
                n_range = (coverage[4] * block_out, min(coverage[5] * block_out, INT_32_MAX))
            else:
                n_range = (coverage[4], coverage[5])
            var_range[m_name] = m_range
            var_range[k_name] = k_range
            var_range[n_name] = n_range
            if self.dynamic_mode == "dynamic_mknb":
                var_range["batch"] = (coverage[6], coverage[7])

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range, "m_k_n_shape": m_k_n_shape}

    def set_default_tiling(self):
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

        k_al1 = _get_reduce(self.tiling_info["A_dtype"])
        k_al0 = _get_reduce(self.tiling_info["A_dtype"])
        k_bl1 = _get_reduce(self.tiling_info["B_dtype"])
        k_bl0 = _get_reduce(self.tiling_info["B_dtype"])

        tiling["AUB_shape"] = [16, 1, 1, 1]
        tiling["BUB_shape"] = [16, 1, 1, 1]

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
        tiling["attach_same_to_static"] = False
        tiling = {"tiling": tiling, "A_shape": self.a_info,
                    "B_shape": self.b_info, "C_shape": self.c_info}

        return tiling

    def _get_calc_info(self):
        """
        preprocess info, convert tvm var to -1
        """

        self._convert_type(self.a_info, self.b_info)

    @staticmethod
    def preprocess_tiling(tiling_in):
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

    @staticmethod
    def get_compile_time(target_area):
        """
        caculate total all compile time depends on target_area
        """
        compile_time = 1
        for value in target_area:
            compile_time *= (value[1] - value[0] + 1)
        return compile_time

    def _get_gear_element(self, range_value, gear, is_batch=False):
        """
        Parameters
        ----------
        range_value: format [m_min,m_max]

        gear format: [1, 4, 8, 16, 32, 64, 128, 192, 256, 512, 768, 1024]

        Returns
        -------
        element_list: list, element to cover the range_value
        """
        if self.none_range_area:
            return gear
        left = 0
        right = 0
        element_list = []
        for index, value in enumerate(gear):
            if value <= range_value[0]:
                left = index
            if index < (len(gear) - 1) and gear[index] <= range_value[1] < gear[index + 1]:
                right = index + 1
        if left == (len(gear) - 1):
            element_list = [gear[-1]]
        if right == 0:
            element_list = gear[left:]
        else:
            element_list = gear[left:right]
        if is_batch and _is_fuzzily_build():
            element_list = [element_list[0]]
        return element_list

    def get_gear_repo_shapes(self, target_area=None):
        """
        caculate all gear repo seeds during range

        Parameters
        ----------
        target_are: format [[m_min,m_max],[k_min,k_max],[n_min,n_max]]

        Returns
        -------
        gear_repo_shapes: list, [(m_value, k_value, n_value),...]
        """
        cls = self.__class__
        target_mkn = cls.NONE_RANGE_M_N if self.none_range_area else cls.GEAR_M_N
        target_batch = cls.NONE_RANGE_BATCH if self.none_range_area else cls.GEAR_BATCH

        gear_m_list = self._get_gear_element(target_area[0], target_mkn)
        gear_k_list = self._get_gear_element(target_area[1], target_mkn)
        gear_n_list = self._get_gear_element(target_area[2], target_mkn)
        if len(target_area) == 4:
            gear_batch_list = self._get_gear_element(target_area[3], target_batch, True)
        else:
            gear_batch_list = [1]
        return list(itertools.product(gear_m_list, gear_k_list, gear_n_list, gear_batch_list))

    def check_tiling_special_value(self, tiling):
        """
        check tiling K full load or AL1 full load or BL1 full load

        Parameters
        ----------
        tiling: tiling value

        Returns
        -------
        bool: if any full load, return True else False
        """
        if tiling.get("AL1_shape") and tiling.get("BL1_shape"):
            if tiling.get("AL1_shape")[1] != 1 or tiling.get("BL1_shape")[1] != 1 or \
               tiling.get("AL1_shape")[0] == (self.tiling_info.get("ca1_var_range")[0] * _get_reduce(self.a_type)) or\
               tiling.get("BL1_shape")[0] == (self.tiling_info.get("ca1_var_range")[0] * _get_reduce(self.a_type)):
                tiling["attach_same_to_static"] = True
                return True
        if (tiling.get("AL1_shape") == []) or (tiling.get("BL1_shape") == []):
            tiling["attach_same_to_static"] = True
            return True
        return False

    def get_gear_tiling_range(self, gear_repo_shapes, seed_shape):
        """
        cacaulate gear repository range

        Parameters
        ----------
        gear_repo_shapes format:[(m_gear, k_gear, n_gear),...]

        seed_shape format: [m_value, k_value, n_value]

        Returns
        -------
        gear_tiling_range: [m_min, m_max, k_min, k_max, n_min, n_max]
        """
        cls = self.__class__

        def _calc_range(value, gear, is_fuzzily_batch=0):
            value_index = gear.index(value)
            if value_index == (len(gear) - 1) or is_fuzzily_batch == 1:
                return [value, MAX_RANGE]
            return [value, (gear[value_index + 1] - 1)]

        gear_tiling_range = []
        if seed_shape in gear_repo_shapes:
            for index, item in enumerate(seed_shape):
                if index == (len(seed_shape) - 1):
                    is_fuzzily_batch = 0
                    if _is_fuzzily_build():
                        is_fuzzily_batch = 1
                    gear_tiling_range += _calc_range(item, cls.GEAR_BATCH, is_fuzzily_batch)
                else:
                    gear_tiling_range += _calc_range(item, cls.GEAR_M_N)
        else:
            for item in seed_shape:
                gear_tiling_range += [item, item]
        return gear_tiling_range

    def use_default_tiling_case(self, target_area):
        """
        check range value of target_area to determine whether to use default_tiling case

        Parameters
        ----------

        target_area: range value of dynamic elements

        Returns
        ----------

        return True means use default tiling, else not
        """

        # cache tiling not support use bias yet
        if self.use_cache_tiling:
            return False

        if self.none_range_area:
            return True

        if len(self.get_gear_repo_shapes(target_area)) > (len(self.__class__.GEAR_M_N)**2):
            return True

        return False
