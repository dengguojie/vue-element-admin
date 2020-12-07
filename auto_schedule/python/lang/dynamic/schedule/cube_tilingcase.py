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
from functools import reduce
from collections import defaultdict
from collections import deque
from abc import abstractmethod

from te.tvm.expr import IntImm
from te.tvm.expr import Expr

from te.lang.base.operation_impl import add_compile_info

CORE_NUM = 32
C0_SIZE = 16
class CubeTilingOp:
    def __init__(self, tiling_info, dynamic_mode):
        self.tiling_info = tiling_info

        self.key = None
        self.dynamic_mode = dynamic_mode
        self.op_type = None

    def _convert_type(self, *info_items):
        """
        convert tvm var to -1
        """
        for item in info_items:
            for i, element in enumerate(item):
                if isinstance(element, IntImm):
                    item[i] = int(element)
                elif isinstance(element, Expr):
                    item[i] = -1

    @abstractmethod
    def get_repo_tiling(self):
        pass

    @abstractmethod
    def get_costmodel_tiling(self, shape):
        pass

    @abstractmethod
    def get_tiling_range(self, tiling_in, shape_info):
        pass

    @abstractmethod
    def assembly_case(self, tiling_strategy, covered, cnt):
        pass


class TilingSelection:
    def __init__(self, tiling_op: CubeTilingOp):
        self.op = tiling_op
        self.seed_cnt = itertools.count(10000)

    def calc_tiling(self, tgt_area):
        """
        calculate tilings

        Parameters
        ----------
        tgt_area: tuple, target area to be covered

        Returns
        -------
        tilings_cases: list, calculated tilings
        """

        add_compile_info("dynamic_mode", self.op.dynamic_mode)
        if self.op.dynamic_mode == "dynamic_hw":
            tiling_cases = self._calc_hw(tgt_area)
        elif self.op.dynamic_mode == "dynamic_batch":
            batch_func_map = {"conv2d_bp_filter": self._calc_batch_v2}
            batch_func = batch_func_map.get(self.op.op_type, self._calc_batch)
            tiling_cases = batch_func(tgt_area)
        elif self.op.dynamic_mode == "dynamic_mkn":
            tiling_cases = self._calc_mkn(tgt_area)
        elif self.op.dynamic_mode == "dynamic_dhw":
            tiling_cases = self._calc_dhw(tgt_area)
        else:
            raise RuntimeError("Only dynamic_dhw/dynamic_hw/dynamic_batch "
                               "is supported")

        tiling_blockdim = {}
        for case in tiling_cases:
            tiling_blockdim[case['key']] = \
                case["block_dim"] if "block_dim" in case else int(
                reduce(lambda x, y: x * y, case['tiling_strategy']['block_dim']))
        add_compile_info("block_dim", tiling_blockdim)

        return tiling_cases

    def _calc_hw(self, tgt_area):
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
            funcs = (max, min, max, min)
            return [func(ta, sa) for func, ta, sa in zip(funcs, tgt_area, seed_area)]

        tgt_area = tuple(tgt_area[0] + tgt_area[1])
        candidates = {}
        repo_seeds = self.op.get_repo_tiling()
        seed_points = set()

        for seed in repo_seeds:
            seed_hw = tuple(seed[self.op.key][2:4])
            if self.op.op_type == "conv2d":
                tiling = seed["tiling"]
                block_dims = tiling["block_dim"]
                block_nums = block_dims[0]*block_dims[1]*block_dims[2]
                if block_nums < CORE_NUM:
                    if seed["A_shape"][0] > 1 and block_dims[0] < seed["A_shape"][0] and \
                            seed["A_shape"][0]*block_dims[1]*block_dims[2] <= CORE_NUM:
                        tiling["block_dim"][0] = seed["A_shape"][0]
                if tiling["BL0_matrix"] and tiling["BL1_shape"]:
                    co1 = (seed["B_shape"][0] + C0_SIZE - 1) // C0_SIZE
                    if block_dims[1]*tiling["BL1_shape"][1]*tiling["BL0_matrix"][1]*2 < co1 and \
                            co1 // (tiling["BL1_shape"][1]*tiling["BL0_matrix"][1]*2)*block_dims[0]* \
                            block_dims[2] <= CORE_NUM:
                        tiling["block_dim"][1] = co1 // (tiling["BL1_shape"][1]*tiling["BL0_matrix"][1]*2)
                block_nums = block_dims[0]*block_dims[1]*block_dims[2]
                if block_nums < CORE_NUM and tiling["AL1_shape"]:
                    hout = self.op._get_output_h(seed["A_shape"][2])
                    wout = self.op._get_output_w(seed["A_shape"][3])
                    tmp = hout*wout // (tiling["AL0_matrix"][0]*C0_SIZE*tiling["AL1_shape"][1]*block_dims[2])
                    if tmp >= 1:
                        tmp = tiling["AL0_matrix"][0]*C0_SIZE*tiling["AL1_shape"][1]
                        tiling["block_dim"][2] = min((hout*wout + tmp - 1) // tmp, CORE_NUM)
                seed["tiling"] = tiling
            seed_range = self.op.get_tiling_range(seed['tiling'], seed[self.op.key])
            if seed_hw in seed_points or _cal_overlap(seed_range, tgt_area)[0] == 0:
                continue
            seed_points.add(seed_hw)
            seed_range = _correct_seed_range(seed_range)
            candidates[next(self.seed_cnt)] = [seed_range, seed['tiling'], seed_hw]

        cost_cases = self._select_tiling(tgt_area, candidates)
        tiling_cases = [
            self.op.assembly_case(v[1], v[0], k) for k, v in candidates.items()]

        add_compile_info("repo_seeds", {k: v[-1] for k, v in candidates.items()})
        repo_range = {k: v[0] for k, v in candidates.items()}

        # call cost model
        cost_tilings, cost_range = self._calc_costmodel(cost_cases)
        tiling_cases += cost_tilings
        if not tiling_cases:
            raise RuntimeError("No tiling generated for this shape and range")

        add_compile_info("repo_range", repo_range)
        add_compile_info("cost_range", cost_range)
        return tiling_cases

    def _calc_mkn(self, tgt_area):
        """
        calculate tilings for dynamic mkn mode

        Parameters
        ----------
        tgt_area: tuple, mkn range to be covered (m_min, m_amx, k_min, k_max, n_min, n_max)

        Returns
        -------
        tilings_cases: list, calculated tilings
        """

        def _correct_seed_range(seed_area):
            funcs = (max, min, max, min, max, min)
            return [func(ta, sa) for func, ta, sa in zip(funcs, tgt_area, seed_area)]

        tgt_area = tuple(tgt_area[0] + tgt_area[1] + tgt_area[2])
        candidates = {}
        repo_seeds = self.op.get_repo_tiling()

        for seed in repo_seeds:
            seed_k_value, seed_m_value = seed[self.op.key[0]][1:3]
            seed_n_value = seed[self.op.key[1]][1]
            m_k_n_shape = (seed_m_value, seed_k_value, seed_n_value)
            seed_range = self.op.get_tiling_range(seed["tiling"], m_k_n_shape)
            seed_range = _correct_seed_range(seed_range)
            candidates[next(self.seed_cnt)] = [seed_range, seed["tiling"], m_k_n_shape]

        cost_cases = self._select_tiling_mkn(tgt_area, candidates)
        tiling_cases = [
            self.op.assembly_case(v[1], v[0], k) for k, v in candidates.items()]
        add_compile_info("repo_seeds", {k: v[-1] for k, v in candidates.items()})
        repo_range = {k: v[0] for k, v in candidates.items()}

        # call cost model
        cost_tilings, cost_range = self._calc_costmodel_mkn(cost_cases)
        tiling_cases += cost_tilings
        if not tiling_cases:
            raise RuntimeError("No tiling generated for this shape and range")

        add_compile_info("repo_range", repo_range)
        add_compile_info("cost_range", cost_range)
        if "trans_a" in self.op.tiling_info and "trans_b" in self.op.tiling_info:
            add_compile_info("attrs", {"transpose_a": self.op.tiling_info["trans_a"],
            "transpose_b": self.op.tiling_info["trans_b"]})

        return tiling_cases

    def _calc_dhw(self, tgt_area):
        """
        calculate tilings for dynamic dhw mode

        Parameters
        ----------
        tgt_area: tuple, dhw range to be covered (d_min, d_max, h_min, h_max,
                                                  w_min, w_max)

        Returns
        -------
        tilings_cases: list, calculated tilings
        """
        def _correct_seed_range(seed_area):
            funcs = (max, min, max, min, max, min)
            return [func(ta, sa) for func, ta, sa in zip(funcs, tgt_area,
                                                         seed_area)]
        tgt_area = tuple(tgt_area[0] + tgt_area[1] + tgt_area[2])
        candidates = {}
        repo_seeds = self.op.get_repo_tiling()
        seed_points = set()

        for seed in repo_seeds:
            seed_dhw = (seed[self.op.key][1], seed[self.op.key][3],
                        seed[self.op.key][4])
            seed_range = self.op.get_tiling_range(seed['tiling'], seed[self.op.key])
            if seed_range[1] == -1:
                seed_range[1] = tgt_area[1]
            if seed_dhw in seed_points or _cal_overlap(seed_range, tgt_area)[0] == 0:
                continue
            seed_points.add(seed_dhw)
            seed_range = _correct_seed_range(seed_range)
            candidates[next(self.seed_cnt)] = [seed_range, seed['tiling'],
                                               seed_dhw]

        cost_cases = self._select_tiling(tgt_area, candidates)
        tiling_cases = [
            self.op.assembly_case(v[1], v[0], k) for k, v in candidates.items()]
        add_compile_info("repo_seeds", {k: v[-1] for k, v in candidates.items()})
        repo_range = {k: v[0] for k, v in candidates.items()}

        # call cost model
        cost_tilings, cost_range = self._calc_costmodel(cost_cases)
        tiling_cases += cost_tilings
        if not tiling_cases:
            raise RuntimeError("No tiling generated for this shape and range")

        add_compile_info("repo_range", repo_range)
        add_compile_info("cost_range", cost_range)
        return tiling_cases

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

        # call cost model
        if not tiling_seeds:
            cur_seed = next(self.seed_cnt)
            cost_seed = self.op.get_costmodel_tiling(sum(batch_range) // 2)
            tiling_cases.append(
                self.op.assembly_case(cost_seed['tiling'], batch_range, cur_seed))
            add_compile_info("tiling_range", {cur_seed: batch_range})
            add_compile_info("repo_seeds", repo_seeds)
            return tiling_cases

        # using repo seeds
        repo_selections = {}
        tiling_seeds.sort(key=lambda x: (x['A_shape'][0], x['tiling']['block_dim'][0]))
        lower_bound = batch_range[0]
        for i, seed in enumerate(tiling_seeds[:-1]):
            cur_batch = seed['A_shape'][0]
            if cur_batch == tiling_seeds[i + 1]['A_shape'][0] or \
                    cur_batch < lower_bound:
                continue
            seed_cnt = next(self.seed_cnt)
            repo_selections[next(self.seed_cnt)] = \
                [seed['tiling'], (lower_bound, min(cur_batch, batch_range[1]))]
            lower_bound = cur_batch + 1
            repo_seeds[seed_cnt] = cur_batch
            if cur_batch >= batch_range[1]:
                break
        else:
            seed_cnt = next(self.seed_cnt)
            cur_batch = tiling_seeds[-1]['A_shape'][0]
            repo_seeds[seed_cnt] = cur_batch
            repo_selections[seed_cnt] = [
                tiling_seeds[-1]['tiling'], (lower_bound, batch_range[1])
            ]

        tiling_range = {k: v[1] for k, v in repo_selections.items()}
        add_compile_info("tiling_range", tiling_range)
        add_compile_info("repo_seeds", repo_seeds)
        tiling_cases = [self.op.assembly_case(v[0], v[1], k)
                        for k, v in repo_selections.items()]
        return tiling_cases

    def _calc_batch_v2(self, tgt_area):
        """
        for some op_type, dynamic batch tiling have restricted coverage
        """
        batch_range = tuple(tgt_area[0])
        tiling_seeds = self.op.get_repo_tiling()
        repo_seeds = {}

        # get tiling range
        candidates = []
        seed_points = set()
        for seed in tiling_seeds:
            seed_n = seed[self.op.key][0]
            seed_range = self.op.get_batch_range(
                seed['tiling'], seed[self.op.key])
            if seed_n < batch_range[0] or seed_n > batch_range[1] or \
                    seed_n in seed_points:
                continue
            seed_points.add(seed_n)
            candidates.append((seed_range, seed['tiling'], seed_n))
        candidates.sort(key=lambda x: x[2])

        # add sentinel seed
        sentinel_seed = (None, None, batch_range[1] + 1)
        candidates.append(sentinel_seed)

        tiling_selections = {}
        cost_cases = deque()
        lower_bound = batch_range[0]
        for i, seed in enumerate(candidates[:-1]):
            lower_covered, upper_covered = seed[0]
            if lower_covered > lower_bound:
                cost_cases.append((lower_bound, lower_covered - 1))
                lower_bound = lower_covered
            upper_covered = min(upper_covered, candidates[i + 1][2] - 1)
            seed_cnt = next(self.seed_cnt)
            tiling_selections[seed_cnt] = \
                [seed[1], (lower_bound, upper_covered)]
            repo_seeds[seed_cnt] = seed[2]
            lower_bound = upper_covered + 1
            if lower_bound > batch_range[1]:
                break
        else:
            cost_cases.append((lower_bound, batch_range[1]))

        # covered by cost_model
        while cost_cases:
            case_len = len(cost_cases)
            for i in range(case_len):
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

        tiling_range = {k: v[1] for k, v in tiling_selections.items()}
        add_compile_info("tiling_range", tiling_range)
        add_compile_info("repo_seeds", repo_seeds)

        tiling_cases = [self.op.assembly_case(v[0], v[1], k)
                        for k, v in tiling_selections.items()]
        return tiling_cases

    def _select_tiling(self, tgt_area, repo_tilings):
        """
        select repo seeds tiling to cover target area

        Parameters
        ----------
        tgt_area: tuple, hw range to be covered (h_min, h_amx, w_min, w_max)
        repo_tilings: dict, repo seeds tilings with id

        Returns
        -------
        res: default_dict, tilings with covered area
        rest_area: deque, uncovered areas
        """

        sort_tiling_list = sorted(repo_tilings.items(),
                                  key=lambda x: _cal_overlap(tgt_area, x[1][0])[0],
                                  reverse=True)
        rest_area = set([tgt_area])

        for t_id, t_info in sort_tiling_list:
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

    def _select_tiling_mkn(self, tgt_area, repo_tilings):
        """
        select repo seeds tiling to cover target area

        Parameters
        ----------
        tgt_area: tuple, m k n range to be covered
        repo_tilings: dict, repo seeds tilings with id

        Returns
        -------
        res: default_dict, tilings with covered area
        rest_area: deque, uncovered areas
        """

        sort_tiling_list = sorted(repo_tilings.items(),
                                  key=lambda x: _cal_overlap_three_dimesional(tgt_area, x[1][0])[0],
                                  reverse=True)
        rest_area = set([tgt_area])

        for t_id, t_info in sort_tiling_list:
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
                cost_seed = self.op.get_costmodel_tiling(tuple(cut_range[1::2]))
                seed_range = self.op.get_tiling_range(cost_seed['tiling'],
                                                      cost_seed[self.op.key])
                if self.op.dynamic_mode == "dynamic_dhw" and seed_range[1] == -1:
                    seed_range[1] = cut_range[1]
                is_overlap, covered_area = _cal_overlap(cut_range, seed_range)
                if is_overlap:
                    gen_rects = _cut_rectangle(cut_range, seed_range)
                    cost_cases.extend(gen_rects)
                else:
                    raise RuntimeError("totally uncovered!!!")

                cur_seed_cnt = next(self.seed_cnt)
                cost_tilings.append(
                    self.op.assembly_case(cost_seed['tiling'], covered_area,
                                          cur_seed_cnt))
                tiling_range[cur_seed_cnt] = covered_area

        return cost_tilings, tiling_range

    def _calc_costmodel_mkn(self, cost_cases):
        """
        calculate cost model to cover rest area after repo seeds

        Parameters
        ----------
        cost_cases: deque, uncovered area in (t, b, l, r, up, down) rectangle format

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
                cost_seed = self.op.get_costmodel_tiling((cut_range[0], cut_range[2], cut_range[4]))
                seed_k_value, seed_m_value = cost_seed[self.op.key[0]][1:3]
                seed_n_value = cost_seed[self.op.key[1]][1]
                m_k_n_shape = (seed_m_value, seed_k_value, seed_n_value)
                seed_range = self.op.get_tiling_range(cost_seed["tiling"],
                                                      m_k_n_shape)
                is_overlap, covered_area = _cal_overlap_three_dimesional(cut_range, seed_range)
                if is_overlap:
                    gen_rects = _cut_cuboid(cut_range, seed_range)
                    cost_cases.extend(gen_rects)
                else:
                    raise RuntimeError("totally uncovered!!!")

                cur_seed_cnt = next(self.seed_cnt)
                cost_tilings.append(
                    self.op.assembly_case(cost_seed["tiling"], covered_area,
                                          cur_seed_cnt))
                tiling_range[cur_seed_cnt] = covered_area

        return cost_tilings, tiling_range

def _cal_overlap(rect1, rect2):
    """
    rect1, rect2: rectangle in (top, bottom, left, right) or
        (front, back, top, bottom, left, right) format
    """
    funcs = [max if i % 2 == 0 else min for i in range(len(rect1))]
    intersection = [func(pos1, pos2) for func, pos1, pos2 in zip(funcs, rect1,
                                                                 rect2)]
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
        return None
    return (max(line1[0], line2[0]), min(line1[1], line2[1]))


def _cut_rectangle(base, cut):
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
            gen_rects.append(tuple(rect_tmp))

        if cut[i + 1] < base[i + 1]:
            rect_tmp = copy.deepcopy(rect)
            rect_tmp[i] = cut[i + 1] + 1
            rect_tmp[i + 1] = base[i + 1]
            gen_rects.append(tuple(rect_tmp))

        rect[i] = max(base[i], cut[i])
        rect[i + 1] = min(base[i + 1], cut[i + 1])

        i = i + 2

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
    L1BUFFER = 1024 * 1024
    FP16_M = 16
    FP16_K = 16
    FP16_N = 16
    FP16_SIZE = 2
    CUBE_SIZE = 16

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
