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

cube ops tiling case base class
"""

import itertools
from functools import reduce
from collections import defaultdict
from collections import deque
from abc import abstractmethod

import te.lang.cce
from te import tvm
from te.domain.tiling.get_tiling import get_tiling
from te.platform import operation


class CubeTilingSelection:

    def __init__(self, tiling_info, dynamic_mode):
        self.tiling_info = tiling_info

        self._key = None
        self.dynamic_mode = dynamic_mode
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

        if self.dynamic_mode == "dynamic_hw":
            tiling_cases = self.calc_hw(tgt_area)
        elif self.dynamic_mode == "dynamic_batch":
            tiling_cases = self.calc_batch(tgt_area)
        else:
            raise RuntimeError("Only dynamic_hw/dynamic_batch is supported")

        tiling_blockdim = {}
        for case in tiling_cases:
            tiling_blockdim[case['key']] = int(
                reduce(lambda x, y: x * y, case['tiling_strategy']['block_dim']))
        operation.add_compile_info("block_dim", tiling_blockdim)

        return tiling_cases

    def calc_hw(self, tgt_area):
        """
        calculate tilings for dynamic hw mode

        Parameters
        ----------
        tgt_area: tuple, hw range to be covered (h_min, h_amx, w_min, w_max)

        Returns
        -------
        tilings_cases: list, calculated tilings
        """

        tgt_area = tgt_area[0] + tgt_area[1]
        candidates = {}
        repo_seeds = self._get_tiling()
        seed_points = set()

        for seed in repo_seeds:
            seed_hw = tuple(seed[self._key][2:4])
            if not _point_in_rect(seed_hw, tgt_area) or seed_hw in seed_points:
                continue
            seed_points.add(seed_hw)
            seed_range = self._get_tiling_range(seed['tiling'], seed[self._key])
            candidates[next(self.seed_cnt)] = [seed_range, seed['tiling'], seed_hw]

        # seed_range may be changed in selection
        repo_selections, cost_cases = self._select_tiling(tgt_area, candidates)
        tiling_cases = [self._assembly_case(candidates[k][1], candidates[k][0], k) \
                        for k in repo_selections.keys()]

        operation.add_compile_info("repo_seeds", {k: v[-1] for k, v in candidates.items()})
        tiling_range = {k: v for k, v in repo_selections.items() if v}

        # call cost model
        tiling_cases += self._calc_cost_model(cost_cases, tiling_range)
        if not tiling_cases:
            raise RuntimeError("No tiling generated for this shape and range")

        operation.add_compile_info("tiling_range", tiling_range)
        return tiling_cases

    def calc_batch(self, tgt_area):
        """
        calculate tilings for dynamic batch mode

        Parameters
        ----------
        tgt_area: tuple, batch range to be covered

        Returns
        -------
        tilings_cases: list, calculated tilings
        """

        batch_range = tgt_area[0]
        tiling_cases = []
        tiling_seeds = get_tiling(self.tiling_info)

        # call cost model
        if not tiling_seeds:
            cur_seed = next(self.seed_cnt)
            cost_seed = self._use_cost_model(sum(batch_range) // 2)
            tiling_cases.append(self._assembly_case(cost_seed['tiling'], batch_range,
                                                    cur_seed))
            operation.add_compile_info("tiling_range", {cur_seed: batch_range})
            return tiling_cases

        # using repo seeds
        repo_selections = {}
        tiling_seeds.sort(key=lambda x: (x['A_shape'][0],
                                         x['tiling']['block_dim'][0]))
        lower_bound = batch_range[0]
        for i, seed in enumerate(tiling_seeds[:-1]):
            cur_batch = seed['A_shape'][0]
            if cur_batch == tiling_seeds[i + 1]['A_shape'][0] or \
                cur_batch < lower_bound:
                continue
            repo_selections[next(self.seed_cnt)] = [seed['tiling'], \
                (lower_bound, min(cur_batch, batch_range[1]))]
            lower_bound = cur_batch + 1
            if cur_batch >= batch_range[1]:
                break
        else:
            repo_selections[next(self.seed_cnt)] = [tiling_seeds[-1]['tiling'], \
                (lower_bound, batch_range[1])]

        tiling_range = {k: v[1] for k, v in repo_selections.items()}
        operation.add_compile_info("tiling_range", tiling_range)

        tiling_cases = [self._assembly_case(v[0], v[1], k) \
                        for k, v in repo_selections.items()]
        return tiling_cases

    def _calc_cost_model(self, cost_cases, tiling_range):
        """
        calculate cost model to cover rest area after repo seeds

        Parameters
        ----------
        cost_cases: deque, uncovered area in (t, b, l, r) rectangle format

        tiling_range: list, each item means covered areas of a tiling cases

        Returns
        -------
        cost tilings: list, tilings calculated by cost model
        """

        cost_tilings = []
        while cost_cases:
            cost_len = len(cost_cases)
            for _ in range(cost_len):
                cut_range = cost_cases.popleft()
                cost_seed = self._use_cost_model((cut_range[1], cut_range[3]))
                seed_range = self._get_tiling_range(cost_seed['tiling'],
                                                    cost_seed[self._key])
                is_overlap, covered_area = _cal_overlap(cut_range, seed_range)
                if is_overlap:
                    gen_rects = _cut_rectangle(cut_range, seed_range)
                    cost_cases.extend(gen_rects)
                else:
                    raise RuntimeError("totally uncovered!!!")

                cur_seed_cnt = next(self.seed_cnt)
                cost_tilings.append(self._assembly_case(cost_seed['tiling'],
                                                covered_area, cur_seed_cnt))
                tiling_range[cur_seed_cnt] = [covered_area]

        return cost_tilings

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
            key=lambda x: _cal_overlap(tgt_area, x[1][0])[0], reverse=True)
        rest_area = set([tgt_area])
        res = defaultdict(list)

        for t_id, t_info in sort_tiling_list:
            generate_area = set()
            delete_area = set()
            for ra in rest_area:
                overlap, covered = _cal_overlap(ra, t_info[0])
                if overlap == 0:
                    continue
                res[t_id].append(covered)
                generate_area |= set(_cut_rectangle(ra, t_info[0]))
                delete_area.add(ra)

            # if no area covered, use itself as range
            # otherwise set range to cover all covered_area
            if not res[t_id]:
                repo_tilings[t_id][0] = (t_info[2][0], t_info[2][0],
                                        t_info[2][1], t_info[2][1])
            else:
                tmp_covered = list(zip(*res[t_id]))
                repo_tilings[t_id][0] = [min(*tmp_covered[0], t_info[2][0]),
                                         max(*tmp_covered[1], t_info[2][0]),
                                         min(*tmp_covered[2], t_info[2][1]),
                                         max(*tmp_covered[3], t_info[2][1])]

            rest_area = (rest_area - delete_area) | generate_area

        return res, deque(rest_area)

    def _get_tiling(self):
        return get_tiling(self.tiling_info)

    def _convert_type(self, *info_items):
        """
        convert tvm var to -1
        """
        for item in info_items:
            for i, element in enumerate(item):
                if isinstance(element, tvm.expr.IntImm):
                    item[i] = int(element)
                elif isinstance(element, (tvm.expr.Expr)):
                    item[i] = -1

    @abstractmethod
    def _use_cost_model(self, shape):
        pass

    @abstractmethod
    def _assembly_case(self, tiling_strategy, covered, cnt):
        pass

    @abstractmethod
    def _get_tiling_range(self, tiling_in, shape_info):
        pass


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


def _cal_overlap(rect1, rect2):
    """
    rect1, rect2: rectangle in (top, bottom, left, right) format
    """

    top = max(rect1[0], rect2[0])
    bottom = min(rect1[1], rect2[1])
    left = max(rect1[2], rect2[2])
    right = min(rect1[3], rect2[3])
    intersection = (top, bottom, left, right)

    if left > right or top > bottom:
        overlap = 0
    else:
        overlap = (right - left + 1) * (bottom - top + 1)

    return (overlap, intersection)


def _cut_rectangle(base, cut):
    """
    base, cut: rectangle in (top, bottom, left, right) format
    """

    gen_rects = []
    if cut[0] > base[0]:
        gen_rects.append((base[0], cut[0] - 1, base[2], base[3]))
    if cut[1] < base[1]:
        gen_rects.append((cut[1] + 1, base[1], base[2], base[3]))
    top = max(base[0], cut[0])
    bottom = min(base[1], cut[1])
    if cut[2] > base[2]:
        gen_rects.append((top, bottom, base[2], cut[2] - 1))
    if cut[3] < base[3]:
        gen_rects.append((top, bottom, cut[3] + 1, base[3]))
    return gen_rects


def _point_in_rect(point, rect):
    """
    rect in (top, bottom, left, right) format
    """

    if rect[0] <= point[0] <= rect[1] and rect[2] <= point[1] <= rect[3]:
        return True
    return False
