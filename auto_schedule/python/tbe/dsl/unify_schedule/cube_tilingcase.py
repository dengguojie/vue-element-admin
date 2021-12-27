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
cube ops tiling case base class
"""

import itertools
import copy
import math
from abc import abstractmethod
from collections import deque
from functools import reduce

import numpy as np

from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.context import get_context
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info
from tbe.tvm.expr import IntImm
from tbe.tvm.expr import Expr


C0_SIZE = 16
W_DELTA = 1
H_LEN = 400
W_LEN = 400
NHW_RANGE_LEN = 6
MAX_RANGE = 2**31 - 1
NDHW_RANGE_LEN = 8


class CubeTilingOp:
    """
    tiling public const and func
    """
    def __init__(self, tiling_info, dynamic_mode, var_map=None):
        self.tiling_info = tiling_info
        self.key = None
        self.var_map = var_map
        self.dynamic_mode = dynamic_mode
        self.op_type = None

    @staticmethod
    def _convert_type(*info_items):
        """
        convert tvm var to -1
        """
        for item in info_items:
            for i, element in enumerate(item):
                if isinstance(element, IntImm):
                    item[i] = int(element)
                elif isinstance(element, Expr):
                    item[i] = -1

    @staticmethod
    def get_batch_range(batch, paras):
        """
        get batch covering range
        """
        if "batch_n" in paras.get("var_map"):
            core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
            batch_max = MAX_RANGE
            if batch > MAX_RANGE:
                batch_max = batch
            if batch >= core_num:
                return core_num, batch_max
            if core_num == TilingUtils.N_BASE:
                return 1, batch_max
            batch_log = int(math.log(batch, TilingUtils.N_BASE))
            return TilingUtils.N_BASE ** batch_log, TilingUtils.N_BASE ** (int(batch_log + 1))
        return batch, batch

    @staticmethod
    def get_h_range(fmap_h, tiling, paras):
        """
        get h covering range
        """
        if "dx_h" in paras.get("var_map") or "fmap_h" in paras.get("var_map"):
            if not tiling["AL1_shape"]:
                return 1, fmap_h
            hi_min = TilingUtils.HW_MIN
            if paras.get("pad_mode") != "VAR":
                hi_min = max(paras.get("k_h") - paras.get("pads")[2] - paras.get("pads")[3], hi_min)
            hi_min = max(hi_min, fmap_h - H_LEN)
            hi_max = min(TilingUtils.NHW_MAX, fmap_h + H_LEN)
            return hi_min, hi_max
        return fmap_h, fmap_h

    def get_w_range(self, fmap_h, fmap_w, tiling, paras):
        """
        get w covering range
        """
        if "dx_w" in paras.get("var_map") or "fmap_w" in paras.get("var_map"):
            if not tiling["AL1_shape"]:
                return 1, fmap_w
            wi_min = TilingUtils.HW_MIN
            if paras.get("pad_mode") != "VAR":
                wi_min = max(paras.get("k_w") - paras.get("pads")[0] - paras.get("pads")[1], wi_min)
            support_w_min = wi_min
            cur_w_size = fmap_w
            # searching up-ward fo rw_max
            while self.check_tiling_match(tiling, cur_w_size, fmap_h) and cur_w_size > support_w_min:
                wi_min = cur_w_size
                cur_w_size = cur_w_size - W_DELTA
            # searching down-ward for w_min
            cur_w_size = fmap_w
            while self.check_tiling_match(tiling, cur_w_size, fmap_h) and cur_w_size <= TilingUtils.NHW_MAX:
                wi_max = cur_w_size
                cur_w_size = cur_w_size + W_DELTA

            wi_min = max(wi_min, fmap_w - W_LEN)
            wi_max = min(wi_max, fmap_w + W_LEN)
            if wi_min > wi_max:
                return 0, 0
            return wi_min, wi_max
        return fmap_w, fmap_w

    @staticmethod
    def check_tiling_match(tiling, current_w, current_h):
        """
        check whether this tiling matches the shape

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_w : int, size of w

        current_h : int, size of h

        Returns
        -------
        bool, True: match
        False: do not match

        """

    @abstractmethod
    def get_repo_tiling(self):
        """
        get tiling from repository

        Returns
        -------
        tiling: shape and tiling retrieved from repository
        """

    @abstractmethod
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

    @abstractmethod
    def get_tiling_range(self, tiling_in, shape_info):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        shape_info : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """

    @abstractmethod
    def assembly_case(self, tiling_strategy, covered, cnt):
        """
        tiling_strategy : dict, result of tiling fetch

        covered : list, size of dymanic element

        cnt: index of tiling

        Returns
        -------
        tiling_case, range covered for tiling
        """

    @staticmethod
    def get_default_range(tgt_area):
        """
        get default range
        Parameters
        ----------
        tgt_area: tuple, target area to be covered

        Returns
        -------
        tgt_area: list, target area to be covered
        """
        return [bound if bound is not None else MAX_RANGE for bound in tgt_area]


class TilingSelection:
    """
    Select tiling
    """
    def __init__(self, tiling_op: CubeTilingOp, cnt=None):
        """
        init TilingSelection
        Parameters
        ----------
        tiling_op: CubeTilingOp
        cnt: initial value of tiling key counter
        """
        self.op = tiling_op
        if not isinstance(cnt, int):
            cnt = 10000
            fuzz_build = get_context().get_build_type() == "fuzzily_build"
            if fuzz_build:
                # >>> start: get kernel id
                kernel_id = get_context().get_addition("max_kernel_id")
                valid = isinstance(kernel_id, int) and kernel_id > -2
                if valid:
                    cnt = kernel_id + 1
                # <<< end: get kernel id
        self.seed_cnt = itertools.count(cnt)

    def _handle_dynamic_nhw(self, target_area, var_names):
        """
        handle dynamic nhw
        Parameters
        ----------
        target_area: tuple, target area to be covered

        var_names: list or tuple, var name

        Returns
        -------
        tiling_cases: list, tiling result
        """
        batch_name, h_name, w_name = var_names
        tgt_area = [*target_area.get(batch_name), *target_area.get(h_name), *target_area.get(w_name)]
        if None in tgt_area:
            seed_cnt = next(self.seed_cnt)
            default_tiling = self.op.get_default_tiling(target_area.get(w_name)[0])
            tiling_cases = [self.op.assembly_case(default_tiling, tgt_area, seed_cnt)]
            add_compile_info("tiling_type", "default_tiling")
            add_compile_info("default_range", {str(seed_cnt): self.op.get_default_range(tgt_area)})
        else:
            add_compile_info("tiling_type", "dynamic_tiling")
            if h_name in self.op.var_map or w_name in self.op.var_map:
                tiling_cases = self._calc_nhw([target_area.get(key) for key in target_area])
            elif batch_name in self.op.var_map:
                if self.op.op_type == "conv2d_bp_filter":
                    tiling_cases = self._calc_batch_v2([target_area.get("batch")])
                else:
                    tiling_cases = self._calc_batch([target_area.get("batch_n")])
            else:
                raise RuntimeError("Only dynamic N/H/W is supported")
        return tiling_cases

    def _handle_dynamic_ndhw(self, target_area, var_names):
        """
        handle dynamic ndhw
        Parameters
        ----------
        tgt_area: tuple, target area to be covered

        var_names: list or tuple, var name

        Returns
        -------
        tiling_cases: list, tiling result
        """
        batch_name, d_name, h_name, w_name = var_names
        tgt_area = [*target_area.get(batch_name), *target_area.get(d_name),
                    *target_area.get(h_name), *target_area.get(w_name)]
        if None in tgt_area:
            seed_cnt = next(self.seed_cnt)
            default_tiling = self.op.get_default_tiling()
            tiling_cases = [self.op.assembly_case(default_tiling, tgt_area, seed_cnt)]
            add_compile_info("tiling_type", "default_tiling")
            add_compile_info("default_range", {str(seed_cnt): self.op.get_default_range(tgt_area)})
        else:
            add_compile_info("tiling_type", "dynamic_tiling")
            if h_name in self.op.var_map or w_name in self.op.var_map or d_name in self.op.var_map:
                tiling_cases = self._calc_ndhw([target_area.get(key) for key in target_area])
            elif batch_name in self.op.var_map:
                batch_func_map = {"conv3d_bp_filter": self._calc_batch_v2}
                batch_func = batch_func_map.get(self.op.op_type, self._calc_batch)
                tiling_cases = batch_func([target_area.get("batch_n")])
            else:
                raise RuntimeError("Only dynamic N/D/H/W is supported")
        return tiling_cases

    def _handle_block_dim(self, tiling_cases):
        """
        handle block dim
        Parameters
        ----------
        tiling_cases: list, tiling result
        """
        tiling_blockdim = {}
        correct_range_flag = False
        for case in tiling_cases:
            if (self.op.op_type in ("conv3d_backprop_input") and
                case['tiling_strategy']['BUB_shape'] is not None):
                tiling_blockdim[case['key']] = (case["block_dim"] if "block_dim" in case else
                                                int(reduce(lambda x, y: x * y,
                                                case['tiling_strategy']['block_dim'])) *
                                                case['tiling_strategy']['BUB_shape'][0])
            elif (self.op.op_type in ("matmul", "batch_matmul") and
                -1 in case['tiling_strategy']['block_dim']):
                tiling_blockdim[case['key']] = 32
            else:
                tiling_blockdim[case['key']] = (case["block_dim"] if "block_dim" in case else
                                                int(reduce(lambda x, y: x * y, case['tiling_strategy']['block_dim'])))
            correct_range_flag = case.get("correct_range_flag")
        add_compile_info("block_dim", tiling_blockdim)
        add_compile_info("correct_range_flag", correct_range_flag)

    def calc_tiling(self, target_area, var_names=()):
        """
        calculate tilings

        Parameters
        ----------
        target_area: tuple, target area to be covered

        Returns
        -------
        tilings_cases: list, calculated tilings
        """
        if self.op.op_type in ("conv2d", "conv2d_bp_input", "conv2d_bp_filter"):
            tiling_cases = self._handle_dynamic_nhw(target_area, var_names)
        elif self.op.op_type in ("conv3d_backprop_input", "convolution_3d", "conv3d_bp_filter"):
            tiling_cases = self._handle_dynamic_ndhw(target_area, var_names)
        else:
            add_compile_info("dynamic_mode", self.op.dynamic_mode)
            if self.op.dynamic_mode in ("dynamic_mkn", "dynamic_mknb"):
                tiling_cases = self._calc_matmul(target_area)
            else:
                raise RuntimeError("Only dynamic_hw/dynamic_batch "
                                   "is supported")
        self._handle_block_dim(tiling_cases)
        return tiling_cases

    def _modify_core_num(self, seed):
        """
        modify core num tilings

        Parameters
        ----------
        seed: dict, tiling seed

        Returns
        -------
        tiling: dict, tiling
        """
        tiling = seed.get("tiling")
        if self.op.op_type != "conv2d":
            return tiling
        block_dims = tiling.get("block_dim")
        block_nums = block_dims[0] * block_dims[1] * block_dims[2]
        if block_nums < tbe_platform_info.get_soc_spec("CORE_NUM"):
            modify_block_batch = seed["A_shape"][0] > 1 and block_dims[0] < seed["A_shape"][0] and \
                                 seed["A_shape"][0] * block_dims[1] * block_dims[2] <= \
                                     tbe_platform_info.get_soc_spec("CORE_NUM")
            if modify_block_batch:
                tiling["block_dim"][0] = seed["A_shape"][0]
                block_dims = tiling["block_dim"]
        tiling_valid = tiling["BL0_matrix"] and tiling["BL1_shape"]
        if tiling_valid:
            co1 = (seed["B_shape"][0] + C0_SIZE - 1) // C0_SIZE
            modify_block_n = block_dims[1] * tiling["BL1_shape"][1] * tiling["BL0_matrix"][1] * 2 < co1 and \
                             co1 // (tiling["BL1_shape"][1] * tiling["BL0_matrix"][1] * 2) * block_dims[0] * \
                             block_dims[2] <= tbe_platform_info.get_soc_spec("CORE_NUM")
            if modify_block_n:
                tiling["block_dim"][1] = co1 // (tiling["BL1_shape"][1] * tiling["BL0_matrix"][1] * 2)
                block_dims = tiling["block_dim"]
        block_nums = block_dims[0] * block_dims[1] * block_dims[2]
        tiling_valid = block_nums < tbe_platform_info.get_soc_spec("CORE_NUM") and tiling["AL1_shape"]
        if tiling_valid:
            hout = self.op.get_output_h(seed["A_shape"][2])
            wout = self.op.get_output_w(seed["A_shape"][3])
            tmp = hout * wout // (tiling["AL0_matrix"][0] * C0_SIZE * tiling["AL1_shape"][1] * block_dims[2])
            if tmp >= 1:
                tmp = tiling["AL0_matrix"][0] * C0_SIZE * tiling["AL1_shape"][1]
                used_core_num = block_dims[0] * block_dims[1]
                tiling["block_dim"][2] = min(
                    (hout*wout + tmp - 1) // tmp, tbe_platform_info.get_soc_spec("CORE_NUM") // used_core_num)

        return tiling

    def _calc_nhw(self, tgt_area):
        """
        calculate tilings for dynamic hw mode

        Parameters
        ----------
        tgt_area: tuple, hw range to be covered (h_min, h_amx, w_min, w_max)

        Returns
        -------
        tilings_cases: list, calculated tilings
        """

        def _correct_seed_range(seed_area):
            funcs = (max, min, max, min, max, min)
            return [func(ta, sa) for func, ta, sa in zip(funcs, tgt_area, seed_area)]

        tgt_area = reduce(lambda x, y: x + y, tgt_area)
        candidates = {}
        repo_seeds = self.op.get_repo_tiling()
        seed_points = set()
        seed_points_dup = set()

        for seed in repo_seeds:
            seed_nhw = (seed[self.op.key][0], seed[self.op.key][2],
                        seed[self.op.key][3])
            seed["tiling"] = self._modify_core_num(seed)
            seed_range = self.op.get_tiling_range(seed['tiling'], seed[self.op.key])
            if seed_range[1] == -1:
                seed_range[1] = tgt_area[1]
            seed_range = seed_range[0] if isinstance(seed_range[0], list) else seed_range

            if seed_nhw in seed_points_dup or _cal_overlap(seed_range, tgt_area)[0] == 0:
                seed_points_dup.add(seed_nhw)
                continue
            seed_points_dup.add(seed_nhw)
            seed_points.add(seed_nhw)
            seed_range = _correct_seed_range(seed_range)
            candidates[next(self.seed_cnt)] = [seed_range, seed['tiling'], seed_nhw]

        cost_cases = self._select_tiling(tgt_area, candidates)
        tiling_cases = [self.op.assembly_case(v[1], v[0], k) for k, v in candidates.items()]

        self._add_repo_compile_info(candidates)

        # call cost model
        cost_tilings, cost_range = self._calc_costmodel(cost_cases)
        tiling_cases += cost_tilings
        if not tiling_cases:
            raise RuntimeError("No tiling generated for this shape and range")

        add_compile_info("cost_range", cost_range)
        return tiling_cases

    def _add_repo_compile_info(self, candidates):
        add_compile_info("repo_seeds", {k: v[-1] for k, v in candidates.items()})
        repo_range = {k: v[0] for k, v in candidates.items()}
        add_compile_info("repo_range", repo_range)
        if "trans_a" in self.op.tiling_info and "trans_b" in self.op.tiling_info:
            add_compile_info("attrs", {"transpose_a": self.op.tiling_info["trans_a"],
            "transpose_b": self.op.tiling_info["trans_b"]})

    def _calc_gear_costmodel_matmul(self, cost_cases, gear_repo_shapes):
        """
        calculate tilingcase depends on gear costmodel

        Parameters
        ----------
        gear_repo_shapes format:[(m_gear, k_gear, n_gear),...]

        cost_cases format: {[m_value, k_value, n_value],...]

        Returns
        -------
        cost_tiling_range: [m_min, m_max, k_min, k_max, n_min, n_max]
        cost_tiling_seeds: list, calculated tilings
        """


        cost_tiling_seeds = []
        cost_tiling_range = {}
        for case in cost_cases:
            cost_seed = self.op.get_costmodel_tiling(case)
            seed_range = self.op.get_gear_tiling_range(gear_repo_shapes, case)
            seed_cnt = next(self.seed_cnt)
            cost_seed["tiling"]["attach_same_to_static"] = False
            cost_tiling_seeds.append(
                self.op.assembly_case(case, cost_seed['tiling'], seed_range, seed_cnt))
            cost_tiling_range[seed_cnt] = seed_range

        return cost_tiling_seeds, cost_tiling_range

    def _get_repo_range_matmul(self, gear_repo_shapes, cost_cases, candidates):
        """
        calculate repo range
        """
        for gear_shape in gear_repo_shapes:
            key_list = ["ha_var_range", "ca1_var_range", "cb1_var_range", "batch_var_range"]
            for index, value in enumerate(gear_shape):
                self.op.tiling_info[key_list[index]] = [value, value]
            gear_repo_seeds = self.op.get_repo_tiling()

            if len(gear_repo_seeds) == 0:
                cost_cases.append(gear_shape)

            for seed in gear_repo_seeds:
                if self.op.check_tiling_special_value(seed["tiling"]):
                    seed_range = self.op.get_tiling_range(seed["tiling"], gear_shape)
                    batch_range = [gear_shape[-1], gear_shape[-1]]
                    seed_range += batch_range
                    seed["tiling"]["attach_same_to_static"] = True
                    candidates[next(self.seed_cnt)] = [seed_range, seed["tiling"], gear_shape]
                seed_gear_range = self.op.get_gear_tiling_range(gear_repo_shapes, gear_shape)
                seed_chaged = copy.deepcopy(seed)
                seed_chaged = self.op.change_full_load_to_value([seed_chaged])[0]
                seed_chaged["tiling"]["attach_same_to_static"] = False
                candidates[next(self.seed_cnt)] = [seed_gear_range, seed_chaged["tiling"], gear_shape]

        self._add_repo_compile_info(candidates)
        return candidates, cost_cases

    def _calc_gear_matmul(self, target_area, fuzzy_build=False, use_cost_model=False):
        """
        calculate tilingcase depends on gear and repository

        Parameters
        ----------
        target_area:list, range to be covered [[m_min, m_max], [k_min, k_max],
                 [n_min, n_max],[batch_min, batch_max]], batch value exsit when
                 dynamic_mknb mode

        Returns
        -------
        tilings_cases: list, calculated tilings
        """
        candidates = {}
        cost_cases = []

        gear_repo_shapes = self.op.get_gear_repo_shapes(target_area)

        for seed in self.op.get_repo_tiling():
            seed_batch_k_m = seed["A_shape"][0:3]
            seed_n_value = seed["B_shape"][1]
            if (seed_batch_k_m[-1], seed_batch_k_m[1], seed_n_value, seed_batch_k_m[0]) not in gear_repo_shapes:
                seed["tiling"]["attach_same_to_static"] = True
                candidates[next(self.seed_cnt)] = self.op.get_repo_candidate(seed, target_area)

        candidates, cost_cases = self._get_repo_range_matmul(gear_repo_shapes, cost_cases, candidates)
        tiling_cases = [
            self.op.assembly_case(v[2], v[1], v[0], k) for k, v in candidates.items()]

        cost_range = []
        if fuzzy_build or use_cost_model:
            cost_tiling_seeds, cost_range = self._calc_gear_costmodel_matmul(cost_cases, gear_repo_shapes)
            add_compile_info("cost_range", cost_range)
            tiling_cases += cost_tiling_seeds
            if not tiling_cases:
                raise RuntimeError("No tiling generated for this shape and range")

        return tiling_cases, cost_range, candidates

    def _calc_matmul_fuzzy(self, target_area):
        """
        calculate tilings for dynamic mkn and mknb mode with fuzzy compile

        Parameters
        ----------
        target_area: lsit, range to be covered
        """

        key_list = ["ha_var_range", "ca1_var_range", "cb1_var_range", "batch_var_range"]
        for index, value in enumerate(target_area):
            self.op.tiling_info[key_list[index]] = value
        tiling_cases, cost_ranges, candidates = self._calc_gear_matmul(target_area, fuzzy_build=True)
        exist_cost_ranges = get_compile_info().get("cost_range", {})
        exist_cost_ranges.update(cost_ranges)
        add_compile_info("cost_range", exist_cost_ranges)
        repo_seeds = {k: v[-1] for k, v in candidates.items()}
        repo_range = {k: v[0] for k, v in candidates.items()}
        exist_repo_seeds = get_compile_info().get("repo_seeds", {})
        exist_repo_range = get_compile_info().get("repo_range", {})
        exist_repo_seeds.update(repo_seeds)
        exist_repo_range.update(repo_range)
        add_compile_info("repo_seeds", exist_repo_seeds)
        add_compile_info("repo_range", exist_repo_range)

        return tiling_cases

    def _calc_gear_repo_none_range(self, target_area):
        candidates = {}

        gear_repo_shapes = self.op.get_gear_repo_shapes(target_area)

        for gear_shape in gear_repo_shapes:
            key_list = ["ha_var_range", "ca1_var_range", "cb1_var_range", "batch_var_range"]
            for index, value in enumerate(gear_shape):
                self.op.tiling_info[key_list[index]] = [value, value]
            gear_repo_seeds = self.op.get_repo_tiling()

            for seed in gear_repo_seeds:
                if self.op.check_tiling_special_value(seed["tiling"]):
                    seed["tiling"]["attach_same_to_static"] = True
                    seed_range = np.array(gear_shape).repeat(2).tolist()
                    candidates[next(self.seed_cnt)] = [seed_range, seed["tiling"], gear_shape]
                else:
                    seed_changed = copy.deepcopy(seed)
                    seed_changed = self.op.change_full_load_to_value([seed_changed])[0]
                    seed_changed["tiling"]["attach_same_to_static"] = False
                    seed_range = np.array(gear_shape).repeat(2).tolist()
                    candidates[next(self.seed_cnt)] = [seed_range, seed_changed["tiling"], gear_shape]

        tiling_cases = [self.op.assembly_case(v[2], v[1], v[0], k) for k, v in candidates.items()]

        self._add_repo_compile_info(candidates)

        return tiling_cases

    def _calc_matmul(self, target_area):
        """
        calculate tilings for dynamic mkn or bmkn mode

        Parameters
        ----------
        target_area:list, range to be covered [[m_min, m_max], [k_min, k_max],
                 [n_min, n_max],[batch_min, batch_max]], batch value exsit when
                 dynamic_mknb mode

        Returns
        -------
        tilings_cases: list, calculated tilings
        """
        if get_context().get_build_type() == "fuzzily_build":
            return self._calc_matmul_fuzzy(target_area)

        tiling_cases = []
        if self.op.use_cache_tiling:
            template_candidates = self.op.get_cache_tiling()
            cache_tiling = [self.op.assembly_case(v[2], v[1], v[0], k) for k, v in template_candidates.items()]
            tiling_cases = cache_tiling

        if self.op.none_range_area:
            tiling_cases += self._calc_gear_repo_none_range(target_area)
            return tiling_cases

        use_cost_model = not self.op.use_cache_tiling
        compile_time = self.op.get_compile_time(target_area)
        if compile_time > self.op.DEFAULT_COMPILE_TIME and self.op.dynamic_mode in ("dynamic_mkn", "dynamic_mknb"):
            tiling_cases += self._calc_gear_matmul(target_area, use_cost_model=use_cost_model)[0]
            return tiling_cases

        candidates = {}
        repo_seeds = self.op.get_repo_tiling()

        for seed in repo_seeds:
            seed["tiling"]["attach_same_to_static"] = False
            candidates[next(self.seed_cnt)] = self.op.get_repo_candidate(seed, target_area)

        repo_tiling_cases = [self.op.assembly_case(v[2], v[1], v[0], k) for k, v in candidates.items()]
        tiling_cases += repo_tiling_cases
        self._add_repo_compile_info(candidates)
        if use_cost_model:
            range_area = tuple(target_area[0] + target_area[1] + target_area[2])
            cost_cases = self._select_tiling_mkn(range_area, candidates)
            cost_tilings, cost_range = self._calc_costmodel_matmul(cost_cases, target_area)
            add_compile_info("cost_range", cost_range)
            tiling_cases += cost_tilings
        if not tiling_cases:
            raise RuntimeError("No tiling generated for this shape and range")

        return tiling_cases

    def _calc_ndhw(self, tgt_area):
        """
        calculate tilings for dynamic ndhw mode

        Parameters
        ----------
        tgt_area: tuple, dhw range to be covered (n_min, n_max, d_min, d_max, h_min, h_max,
                                                  w_min, w_max)

        Returns
        -------
        tilings_cases: list, calculated tilings
        """

        def _correct_seed_range(seed_area):
            funcs = (max, min, max, min, max, min, max, min)
            return [func(ta, sa) for func, ta, sa in zip(funcs, tgt_area,
                                                         seed_area)]
        tgt_area = reduce(lambda x, y: x + y, tgt_area)
        candidates = {}
        repo_seeds = self.op.get_repo_tiling()
        seed_points = set()

        for seed in repo_seeds:
            seed_ndhw = (seed[self.op.key][0], seed[self.op.key][1],
                        seed[self.op.key][2], seed[self.op.key][3])
            if self.op.op_type in ("convolution_3d", "conv3d_bp_filter"):
                # a shape format is ndc1hwc0
                seed_ndhw = (seed[self.op.key][0], seed[self.op.key][1],
                            seed[self.op.key][3], seed[self.op.key][4])

            seed_range = self.op.get_tiling_range(seed['tiling'], seed[self.op.key])
            if seed_range[1] == -1:
                seed_range[1] = tgt_area[1]
            if seed_range[3] == -1:
                seed_range[3] = tgt_area[3]
            if seed_ndhw in seed_points or _cal_overlap(seed_range, tgt_area)[0] == 0:
                continue
            seed_points.add(seed_ndhw)
            seed_range = _correct_seed_range(seed_range)
            candidates[next(self.seed_cnt)] = [seed_range, seed['tiling'],
                                               seed_ndhw]

        cost_cases = self._select_tiling(tgt_area, candidates)
        tiling_cases = [
            self.op.assembly_case(v[1], v[0], k) for k, v in candidates.items()]
        self._add_repo_compile_info(candidates)

        # call cost model
        cost_tilings, cost_range = self._calc_costmodel(cost_cases)
        tiling_cases += cost_tilings
        if not tiling_cases:
            raise RuntimeError("No tiling generated for this shape and range")

        add_compile_info("cost_range", cost_range)
        return tiling_cases

    @staticmethod
    def _modify_conv2d_tiling_batch(conv2d_modify_tiling, seed):
        """
        modify conv2d tiling batch
        """
        if conv2d_modify_tiling:
            tiling = seed["tiling"]
            tiling["n_bef_batch_flag"] = 1
            seed["tiling"] = tiling
        return seed

    def _remove_duplicate_repo_tilings(self, tiling_seeds, batch_range, repo_seeds):
        """
        remove duplicate repo tilings

        Parameters
        ----------
        tiling_seeds: list, shape info and tiling info

        batch_range: list, batch dim range

        repo_seeds: dict, repo_seeds shape

        Returns
        -------
        repo_selections: dict, tiling_info and batch range

        repo_seeds: dict, repo_seeds shape
        """
        # using repo seeds
        repo_selections = {}
        tiling_seeds.sort(key=lambda x: (x['A_shape'][0], x['tiling']['block_dim'][0]))
        # remove duplicate repo tilings
        tiling_seeds_unique = []
        for i, seed in enumerate(tiling_seeds):
            if seed['A_shape'][0] == tiling_seeds[i - 1]['A_shape'][0]:
                continue
            tiling_seeds_unique.append(seed)
        lower_bound = batch_range[0]
        for i, seed in enumerate(tiling_seeds_unique[:-1]):
            next_batch = tiling_seeds_unique[i + 1]['A_shape'][0]
            if next_batch <= lower_bound:
                continue
            seed_cnt = next(self.seed_cnt)
            tiling_block_dims = seed["tiling"]["block_dim"]
            block_nums = tiling_block_dims[0]*tiling_block_dims[1]*tiling_block_dims[2]
            if block_nums < tbe_platform_info.get_soc_spec("CORE_NUM"):
                seed["tiling"]["block_dim"][0] = (tbe_platform_info.get_soc_spec("CORE_NUM")
                // (tiling_block_dims[1]*tiling_block_dims[2]))
            conv2d_modify_tiling = self.op.op_type == "conv2d" and \
                                   seed['A_shape'][0] > seed["tiling"]["block_dim"][0] and \
                                   seed["tiling"]["BL1_shape"] and seed["tiling"]["BL1_shape"][0] == \
                                   seed["B_shape"][1] * seed["B_shape"][2] * seed["B_shape"][3] * seed["B_shape"][4]
            seed = self._modify_conv2d_tiling_batch(conv2d_modify_tiling, seed)
            # cover upper range
            repo_selections[seed_cnt] = [seed['tiling'], (lower_bound, min(next_batch - 1, batch_range[1]))]
            lower_bound = next_batch
            repo_seeds[seed_cnt] = seed['A_shape'][0]
            if lower_bound > batch_range[1]:
                break
        else:
            seed_cnt = next(self.seed_cnt)
            repo_seeds[seed_cnt] = tiling_seeds[-1]['A_shape'][0]
            repo_selections[seed_cnt] = [tiling_seeds[-1]['tiling'], (lower_bound, batch_range[1])]
        return repo_selections, repo_seeds

    def _calc_batch(self, tgt_area):
        """
        calculate tilings for dynamic batch mode

        Parameters
        ----------
        tgt_area: tuple, batch range to be covered

        Returns
        -------
        tilings_cases: list, calculated tilings
        """

        batch_range = tuple(tgt_area[0])
        tiling_cases = []
        tiling_seeds = self.op.get_repo_tiling()
        repo_seeds = {}

        # for default tiling
        conv3d_default_tiling_mode = None in batch_range and self.op.op_type == "convolution_3d"
        if conv3d_default_tiling_mode:
            batch_range = [batch_range[0], MAX_RANGE]
            cur_seed = next(self.seed_cnt)
            default_tiling = self.op.get_default_tiling()
            tiling_cases.append(
                self.op.assembly_case(default_tiling, batch_range, cur_seed))
            add_compile_info("tiling_range", {cur_seed: batch_range})
            add_compile_info("repo_seeds", repo_seeds)
            return tiling_cases

        # call cost model
        if not tiling_seeds:
            cur_seed = next(self.seed_cnt)
            cost_seed = self.op.get_costmodel_tiling(sum(batch_range) // 2)
            if self.op.op_type == "conv2d_bp_input":
                tiling = copy.deepcopy(cost_seed.get("tiling"))
                if tiling["AL1_shape"]:
                    tiling["AL1_shape"][0] = tiling["AL1_shape"][0] // (cost_seed.get("B_shape")[2] * \
                                             cost_seed.get("B_shape")[3] * TilingUtils.CUBE_SIZE)
                if tiling["BL1_shape"]:
                    tiling["BL1_shape"][0] = tiling["BL1_shape"][0] // (cost_seed.get("B_shape")[2] * \
                                             cost_seed.get("B_shape")[3] * TilingUtils.CUBE_SIZE)
                if not self.op.check_tiling_match(tiling, cost_seed.get("C_shape")[3], cost_seed.get("C_shape")[2]):
                    raise RuntimeError("Input range is too large, the minimum tiling may exceed L1_Buffer")
            tiling_cases.append(
                self.op.assembly_case(cost_seed['tiling'], batch_range, cur_seed))
            add_compile_info("tiling_range", {cur_seed: batch_range})
            add_compile_info("repo_seeds", repo_seeds)
            return tiling_cases

        repo_selections, repo_seeds = self._remove_duplicate_repo_tilings(tiling_seeds, batch_range, repo_seeds)

        add_compile_info("tiling_range", {k: v[1] for k, v in repo_selections.items()})
        add_compile_info("repo_seeds", repo_seeds)
        tiling_cases = [self.op.assembly_case(v[0], v[1], k)
                        for k, v in repo_selections.items()]

        return tiling_cases

    def covered_cost_model_batch_v2(self, cost_cases, tiling_selections):
        """
        for batch_v2 mode, calc covered space of cost_model
        """
        while cost_cases:
            for _ in range(len(cost_cases)):
                cut_range = cost_cases.popleft()
                cost_seed = self.op.get_costmodel_tiling(sum(cut_range) // 2)
                seed_range = self.op.get_batch_range(cost_seed['tiling'],
                                                     cost_seed[self.op.key])
                overlap_line = _cal_overlap_line(cut_range, seed_range)
                if overlap_line:
                    cost_cases.extend(_cut_line(cut_range, seed_range))
                    tiling_selections[next(self.seed_cnt)] = \
                        [cost_seed['tiling'], overlap_line]
                else:
                    raise RuntimeError("totally uncovered!!!")
        return tiling_selections

    def _calc_batch_v2(self, tgt_area):
        """
        for some op_type, dynamic batch tiling have restricted coverage
        """

        batch_range = tuple(tgt_area[0])
        repo_seeds = {}

        # get tiling range
        candidates = []
        seed_points = set()
        for seed in self.op.get_repo_tiling():
            seed_n = seed[self.op.key][0]
            seed_n_invalid = seed_n < batch_range[0] or seed_n > batch_range[1] or seed_n in seed_points
            if seed_n_invalid:
                continue
            seed_points.add(seed_n)
            candidates.append((self.op.get_batch_range(seed['tiling'], seed[self.op.key]), seed['tiling'], seed_n))
        candidates.sort(key=lambda x: x[2])

        # add sentinel seed
        candidates.append((None, None, batch_range[1] + 1))

        tiling_selections = {}
        cost_cases = deque()
        lower_bound = batch_range[0]
        covered = {}
        for i, seed in enumerate(candidates[:-1]):
            covered["lower_covered"], covered["upper_covered"] = seed[0]
            if covered.get("lower_covered") > lower_bound:
                cost_cases.append((lower_bound, covered.get("lower_covered") - 1))
                lower_bound = covered.get("lower_covered")
            covered["upper_covered"] = min(covered.get("upper_covered"), candidates[i + 1][2] - 1)
            seed_cnt = next(self.seed_cnt)
            tiling_selections[seed_cnt] = [seed[1], (lower_bound, covered.get("upper_covered"))]
            repo_seeds[seed_cnt] = seed[2]
            lower_bound = covered.get("upper_covered") + 1
            if lower_bound > batch_range[1]:
                break
        else:
            cost_cases.append((lower_bound, batch_range[1]))

        # covered by cost_model
        tiling_selections = self.covered_cost_model_batch_v2(cost_cases, tiling_selections)

        add_compile_info("tiling_range", {k: v[1] for k, v in tiling_selections.items()})
        add_compile_info("repo_seeds", repo_seeds)
        return [self.op.assembly_case(v[0], v[1], k) for k, v in tiling_selections.items()]

    @staticmethod
    def _select_tiling(tgt_area, repo_tilings):
        """
        select repo seeds tiling to cover target area

        Parameters
        ----------
        tgt_area: tuple, hw range to be covered (n_min, n_max, h_min, h_amx, w_min, w_max)
        repo_tilings: dict, repo seeds tilings with id

        Returns
        -------
        res: default_dict, tilings with covered area
        rest_area: deque, uncovered areas
        """

        sort_tiling_list = sorted(repo_tilings.items(),
                                  key=lambda x: _cal_overlap(tgt_area, x[1][0])[0],
                                  reverse=True)
        rest_area = set()
        rest_area.add(tgt_area)

        for _, t_info in sort_tiling_list:
            generate_area = set()
            delete_area = set()
            for ra in rest_area:
                overlap, _ = _cal_overlap(ra, t_info[0])
                if overlap == 0:
                    continue
                generate_area |= set(_cut_rectangle(ra, t_info[0]))
                delete_area.add(ra)
            rest_area = (rest_area - delete_area) | generate_area

        return deque(rest_area)

    @staticmethod
    def _select_tiling_mkn(target_area, repo_tilings):
        """
        select repo seeds tiling to cover target area

        Parameters
        ----------
        target_area: tuple, m k n range to be covered
        repo_tilings: dict, repo seeds tilings with id

        Returns
        -------
        res: default_dict, tilings with covered area
        rest_area: deque, uncovered areas
        """

        sort_tiling_list = sorted(repo_tilings.items(),
                                  key=lambda x: _cal_overlap_three_dimesional(target_area, x[1][0])[0],
                                  reverse=True)
        rest_area = set()
        rest_area.add(target_area)

        for _, t_info in sort_tiling_list:
            generate_area = set()
            delete_area = set()
            for ra in rest_area:
                overlap, _ = _cal_overlap_three_dimesional(ra, t_info[0])
                if overlap == 0:
                    continue
                generate_area |= set(_cut_cuboid(ra, t_info[0]))
                delete_area.add(ra)

            rest_area = (rest_area - delete_area) | generate_area

        return deque(rest_area)

    def _calc_costmodel_range(self, cost_info, cut_range, cost_tilings, tiling_range):
        """
        calculate cost model range

        Parameters
        ----------
        cost_info: list, include cost_seed and cost_cases

        cut_range: list, cut range

        cost_tilings: list , cost tiling info

        tiling_range: list, each item means covered areas of a tiling cases

        Returns
        -------
        cost tilings: list, tilings calculated by cost model

        tiling_range: list, each item means covered areas of a tiling cases

        cost_cases: deque, uncovered area in (t, b, l, r) rectangle or cube format
        """
        cost_seed, cost_cases = cost_info
        seed_range = self.op.get_tiling_range(cost_seed['tiling'], cost_seed[self.op.key])
        if isinstance(seed_range[0], list):
            is_overlap, covered_area = _cal_overlap(cut_range, seed_range[0])
            _, covered_area_self = _cal_overlap(cut_range, seed_range[1])
            gen_rects = _cut_rectangle(cut_range, seed_range[0], seed_range[1])
            cost_cases.extend(gen_rects)
            if is_overlap:
                cur_seed_cnt = next(self.seed_cnt)
                cost_tilings.append(
                    self.op.assembly_case(cost_seed['tiling'], covered_area, cur_seed_cnt))
                tiling_range[cur_seed_cnt] = covered_area
            cur_seed_cnt = next(self.seed_cnt)
            cost_tilings.append(
                self.op.assembly_case(cost_seed['tiling'], covered_area_self, cur_seed_cnt))
            tiling_range[cur_seed_cnt] = covered_area_self
        else:
            if seed_range[1] == -1:
                seed_range[1] = cut_range[1]
            if seed_range[3] == -1:
                seed_range[3] = cut_range[3]

            is_overlap, covered_area = _cal_overlap(cut_range, seed_range)
            if is_overlap:
                gen_rects = _cut_rectangle(cut_range, seed_range)
                cost_cases.extend(gen_rects)
            else:
                raise RuntimeError("totally uncovered, need_range is {}".format(str(cut_range)),
                                    "seed_range is {}".format(str(seed_range)))

            cur_seed_cnt = next(self.seed_cnt)
            cost_tilings.append(
                self.op.assembly_case(cost_seed['tiling'], covered_area,
                                    cur_seed_cnt))
            tiling_range[cur_seed_cnt] = covered_area
        return cost_tilings, tiling_range, cost_cases

    def _calc_costmodel(self, cost_cases):
        """
        calculate cost model to cover rest area after repo seeds

        Parameters
        ----------
        cost_cases: deque, uncovered area in (t, b, l, r) rectangle or
            cube format

        tiling_range: list, each item means covered areas of a tiling cases

        Returns
        -------
        cost tilings: list, tilings calculated by cost model
        """

        cost_tilings = []
        tiling_range = {}
        while cost_cases:
            cost_len = len(cost_cases)
            for _ in range(cost_len):
                cut_range = cost_cases.popleft()
                conv2d_situation = self.op.op_type in ("conv2d", "conv2d_bp_input", "conv2d_bp_filter") \
                                   and len(cut_range) == NHW_RANGE_LEN
                conv3d_situation = self.op.op_type in ("conv3d_backprop_input", "convolution_3d", "conv3d_bp_filter") \
                                   and len(cut_range) == NDHW_RANGE_LEN
                seed_shape = tuple(cut_range[1::2])
                if conv2d_situation:
                    seed_shape = (cut_range[0], cut_range[3], cut_range[5])
                elif conv3d_situation:
                    seed_shape = (cut_range[0], cut_range[3], cut_range[5], cut_range[7])
                cost_seed = self.op.get_costmodel_tiling(seed_shape)
                if self.op.op_type == "conv2d_bp_input" and not self.op.check_tiling_al0(cost_seed):
                    cost_cases.append((cut_range[0], cut_range[1], cut_range[2],
                                       cut_range[3], cut_range[4], cut_range[5] - 1))
                    continue
                cost_info = [cost_seed, cost_cases]
                cost_tilings, tiling_range, cost_cases = self._calc_costmodel_range(cost_info, cut_range,
                                                                                    cost_tilings, tiling_range)
        return cost_tilings, tiling_range

    def _calc_costmodel_matmul(self, cost_cases, target_area):
        """
        calculate cost model to cover rest area after repo seeds

        Parameters
        ----------
        cost_cases: deque, uncovered area in (t, b, l, r, up, down) rectangle format

        target_area: range value of dymanic elements

        Returns
        -------
        cost tilings: list, tilings calculated by cost model

        tiling_range: list, each item means covered areas of a tiling cases
        """

        cost_tilings = []
        tiling_range = {}
        seed_value = {}
        while cost_cases:
            for _ in range(len(cost_cases)):
                cut_range = cost_cases.popleft()
                cost_seed = self.op.get_costmodel_tiling((cut_range[0], cut_range[2], cut_range[4]))
                seed_value["seed_k_value"] = cost_seed[self.op.key[0]][1]
                seed_value["seed_m_value"] = cost_seed[self.op.key[0]][2]
                seed_value["seed_n_value"] = cost_seed[self.op.key[1]][1]
                seed_value["seed_batch_value"] = cost_seed[self.op.key[0]][0]
                seed_range = self.op.get_tiling_range(cost_seed["tiling"], (seed_value.get("seed_m_value"),
                                                      seed_value.get("seed_k_value"), seed_value.get("seed_n_value")))
                is_overlap, covered_area = _cal_overlap_three_dimesional(cut_range, seed_range)
                if is_overlap:
                    gen_rects = _cut_cuboid(cut_range, seed_range)
                    cost_cases.extend(gen_rects)
                else:
                    raise RuntimeError("totally uncovered!!!")
                if self.op.dynamic_mode == "dynamic_mknb":
                    covered_area += target_area[-1]
                cur_seed_cnt = next(self.seed_cnt)
                m_k_n_shape_for_assembly = [seed_value.get("seed_m_value"), seed_value.get("seed_k_value"),
                                            seed_value.get("seed_n_value"), seed_value.get("seed_batch_value")]
                cost_tilings.append(
                    self.op.assembly_case(m_k_n_shape_for_assembly, cost_seed["tiling"], covered_area,
                                          cur_seed_cnt))
                tiling_range[cur_seed_cnt] = covered_area

        return cost_tilings, tiling_range


def _cal_overlap(rect1, rect2):
    """
    rect1, rect2: rectangle in (top, bottom, left, right) or
        (front, back, top, bottom, left, right) format
    """
    funcs = [max if i % 2 == 0 else min for i in range(len(rect1))]
    intersection = [func(pos1, pos2) for func, pos1, pos2 in zip(funcs, rect1, rect2)]
    overlaps = [0 if start > end else end - start + 1 for start, end in
                zip(intersection[0::2], intersection[1::2])]
    overlap = reduce(lambda x, y: x * y, overlaps)

    return (overlap, intersection)


def _cal_overlap_three_dimesional(rect1, rect2):
    """
    rect1, rect2: rectangle in (top, bottom, left, right, depth_down, depth_up) format
    """

    top = max(rect1[0], rect2[0])
    bottom = min(rect1[1], rect2[1])
    left = max(rect1[2], rect2[2])
    right = min(rect1[3], rect2[3])
    depth_down = max(rect1[4], rect2[4])
    depth_up = min(rect1[5], rect2[5])

    intersection = [top, bottom, left, right, depth_down, depth_up]

    if left > right or top > bottom or depth_down > depth_up:
        overlap = 0
    else:
        overlap = (right - left + 1) * (bottom - top + 1) * (depth_up - depth_down + 1)

    return (overlap, intersection)


def _cal_overlap_line(line1, line2):
    if line1[0] > line2[1] or line1[1] < line2[0]:
        return ()
    return (max(line1[0], line2[0]), min(line1[1], line2[1]))


def _cut_rectangle(base, cut, cut_self=()):
    """
    base, cut: rectangle in (top, bottom, left, right) format
    """

    gen_rects = []
    rect = list(base)
    i = 0
    while i < len(base):
        if i % 2 != 0:
            i = i + 2
            continue

        if cut[i] > base[i]:
            rect_tmp = copy.deepcopy(rect)
            rect_tmp[i] = base[i]
            rect_tmp[i + 1] = cut[i] - 1
            gen_rects.append(rect_tmp)

        if cut[i + 1] < base[i + 1]:
            rect_tmp = copy.deepcopy(rect)
            rect_tmp[i] = cut[i + 1] + 1
            rect_tmp[i + 1] = base[i + 1]
            gen_rects.append(rect_tmp)

        rect[i] = max(base[i], cut[i])
        rect[i + 1] = min(base[i + 1], cut[i + 1])

        i = i + 2

    if cut_self:
        cut_self = _cal_overlap(base, cut_self)[1]
        for index, rect in enumerate(gen_rects):
            if rect[:4] == cut_self[:4] and rect[-1] == cut_self[-1]:
                gen_rects[index][-1] -= 1
    gen_rects = [tuple(rect) for rect in gen_rects]
    return gen_rects


def _cut_cuboid(base, cut):
    """
    base, cut: rectangle in (top, bottom, left, right, depth_down, depth_up) format
    """

    gen_rects = []
    if cut[0] > base[0]:
        gen_rects.append((base[0], cut[0] - 1, base[2], base[3], base[4], base[5]))
    if cut[1] < base[1]:
        gen_rects.append((cut[1] + 1, base[1], base[2], base[3], base[4], base[5]))
    top = max(base[0], cut[0])
    bottom = min(base[1], cut[1])
    if cut[2] > base[2]:
        gen_rects.append((top, bottom, base[2], cut[2] - 1, base[4], base[5]))
    if cut[3] < base[3]:
        gen_rects.append((top, bottom, cut[3] + 1, base[3], base[4], base[5]))
    left = max(base[2], cut[2])
    right = min(base[3], cut[3])

    if cut[4] > base[4]:
        gen_rects.append((top, bottom, left, right, base[4], cut[4] - 1))
    if cut[5] < base[5]:
        gen_rects.append((top, bottom, left, right, cut[5] + 1, base[5]))

    return gen_rects


def _cut_line(base_line, cut_line):
    segments = []
    if base_line[0] < cut_line[0]:
        segments.append([base_line[0], cut_line[0] - 1])
    if base_line[1] > cut_line[1]:
        segments.append([cut_line[1] + 1, base_line[1]])
    return segments


class TilingUtils:
    """
    tilingcase public const and func
    """
    L1BUFFER = 1024 * 1024
    FP16_M = 16
    FP16_K = 16
    FP16_N = 16
    FP16_SIZE = 2
    CUBE_SIZE = 16
    DB_ON = 2
    DB_OFF = 1
    N_BASE = 2
    HW_MIN = 1
    ATTACH_FULL_LOAD = 0
    ATTACH_EQUAL = 1
    ATTACH_LESS = 2
    ATTACH_LARGE = 3
    NHW_MAX = 4096

    @staticmethod
    def icd(num_a, num_b):
        """
        upper division
        """
        return (num_a + num_b - 1) // num_b

    @staticmethod
    def align(num_a, num_b):
        """
        upper round
        """
        return TilingUtils.icd(num_a, num_b) * num_b

    @staticmethod
    def trans_to_int(num):
        """
        trans parameters to int
        Parameters
        ----------
        num: str, need to trans

        Returns
        -------
        num: int, type is int
        """
        return num if not num else int(num)
