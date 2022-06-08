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
general transdata
"""
from .constants import UNKNOWN_DIM
from .constants import GENERAL_FORWARD, GENERAL_BACKWARD
from .constants import BORROW_N_B8B16_FORWARD, BORROW_N_B8B16_BACKWARD
from .constants import BORROW_H_B8B16_FORWARD, BORROW_H_B8B16_BACKWARD
from .transdata_classifier import TransdataClassify


class GeneralForwardClassify(TransdataClassify):
    """
    GeneralForwardClassify
    """

    def __init__(self, ins):
        super().__init__(ins)
        self._axes_map = dict(sorted(self._axes_map.items()))

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return "forward"

    def classify(self):
        """
        Regulation:  dst_shape <dst> <---> <src> src_shape
        DST_SHAPE    DST_LIST    SHAPE_BEFORE_TRANSPOSE    SRC_LIST    PAD_LIST
           N            0                N                     0
           C1           2                H                     1
           H            3                W                     2
           W            1                C1                    3           1
           C0           4                C0                    3           4
        """
        self._do_preprocess()
        self.combine = _do_idx_fusion(self.src, self.dst, self.pad)
        src, dst, pad = [], [], []
        for i, j, k in self.combine:
            src.append(i)
            dst.append(j)
            pad.append(k)

        # update src_shape
        _src = list(set(src))
        self._src_shape = _do_shape_fusion(self._src_shape, _src)

        # update src
        back = sorted(_src)
        self.src = [back.index(x) for x in _src]

        # update dst_shape
        self._dst_shape = _do_shape_fusion(self._dst_shape, sorted(dst))

        # update dst
        back = sorted(dst)
        self.dst = [back.index(x) for x in dst]

        # update axes_map
        self._update_axes_map(_src)
        self.add_dynamic_compile_info(_src, self.dst, self._dst_shape)
        return self._update_res()

    def _do_preprocess(self):
        for k, v in self._axes_map.items():
            if isinstance(v, (list, tuple)):
                self.dst.extend(v)
                self.src.extend([k, ] * len(v))
                self.pad.extend([True] * len(v))
            else:
                self.dst.append(v)
                self.src.append(k)
                self.pad.append(False)

        self._src_shape, self._dst_shape, self.src, self.dst, self.pad = \
            _eliminate_one(self._src_shape, self._dst_shape, self.src, self.dst, self.pad)

    def _update_axes_map(self, src_list):
        self._axes_map = {v: [] for v in self.src}
        for index, (i, _, v) in enumerate(self.combine):
            if v:
                self._axes_map[src_list.index(i)].append(self.dst[index])
            else:
                self._axes_map[src_list.index(i)] = self.dst[index]

    def _update_res(self):
        self.create_base_result(GENERAL_FORWARD)
        return self._bh_bn_branch()

    def _bh_bn_branch(self):
        if self.choose_base_branch_by_model(self.dst):
            return [self._ins, ]

        if _is_const(self._src_shape):
            branch = self.choose_branch_by_value(self._src_shape, self.dst)
            return [self._ins] if branch == GENERAL_FORWARD else self._const_performance(branch)

        return self._dynamic_performance()

    def _dynamic_performance(self):
        result = [self._ins, self.create_bn_bh_result(BORROW_H_B8B16_FORWARD)]
        if self.dst[0] == 0:
            result.append(self.create_bn_bh_result(BORROW_N_B8B16_FORWARD))
        return result

    def _const_performance(self, branch):
        ins = self.create_bn_bh_result(branch)
        self.add_const_compile_info(ins[1], ins[2])
        return [ins, ]


class GeneralBackwardClassify(TransdataClassify):
    """
    GeneralBackwardClassify
    """

    def __init__(self, ins):
        super().__init__(ins)
        self._axes_map = dict(sorted(self._axes_map.items(), key=lambda x: x[1]))

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return "backward"

    def classify(self):
        # dst is : 0, 1, 2, 3
        # src is : 0, 2, 3, (1,4)
        # fuse dst and mapping src
        self._do_preprocess()
        self.combine = _do_idx_fusion(self.dst, self.src, self.de_pad)
        dst, src, de_pad = [], [], []
        for i, j, k in self.combine:
            dst.append(i)
            src.append(j)
            de_pad.append(k)

        # update dst_shape
        _dst = list(set(dst))
        self._dst_shape = _do_shape_fusion(self._dst_shape, _dst)

        # update dst
        back = sorted(_dst)
        self.dst = [back.index(x) for x in _dst]

        # update src_shape
        self._src_shape = _do_shape_fusion(self._src_shape, sorted(src))

        # update src
        back = sorted(src)
        self.src = [back.index(x) for x in src]

        # update axes_map
        self._update_axes_map(_dst)
        self.add_dynamic_compile_info(_dst, self.src, self._src_shape)
        return self._update_res()

    def _do_preprocess(self):
        for k, v in self._axes_map.items():
            if isinstance(k, (list, tuple)):
                self.dst.extend([v, ] * len(k))
                self.src.extend(k)
                self.de_pad.extend([True] * len(k))
            else:
                self.dst.append(v)
                self.src.append(k)
                self.de_pad.append(False)

        self._dst_shape, self._src_shape, self.dst, self.src, self.de_pad = \
            _eliminate_one(self._dst_shape, self._src_shape, self.dst, self.src, self.de_pad)

    def _update_axes_map(self, dst_list):
        axes_map = {v: [] for v in self.dst}
        for index, (i, _, v) in enumerate(self.combine):
            if v:
                axes_map[dst_list.index(i)].append(self.src[index])
            else:
                axes_map[dst_list.index(i)] = self.src[index]

        self._axes_map = {}
        for k, v in axes_map.items():
            if isinstance(v, list):
                self._axes_map[tuple(v)] = k
            else:
                self._axes_map[v] = k

    def _update_res(self):
        self.create_base_result(GENERAL_BACKWARD)
        return self._bh_bn_branch()

    def _bh_bn_branch(self):
        if self.choose_base_branch_by_model(self.src):
            return [self._ins, ]

        if _is_const(self._dst_shape):
            branch = self.choose_branch_by_value(self._dst_shape, self.src)
            return [self._ins] if branch == GENERAL_BACKWARD else self._const_performance(branch)

        return self._dynamic_performance()

    def _dynamic_performance(self):
        result = [self._ins, self.create_bn_bh_result(BORROW_H_B8B16_BACKWARD)]
        if self.src[0] == 0:
            result.append(self.create_bn_bh_result(BORROW_N_B8B16_BACKWARD))
        return result

    def _const_performance(self, branch):
        ins = self.create_bn_bh_result(branch)
        self.add_const_compile_info(ins[0].get("shape", None), ins[2])
        return [ins, ]


def _do_idx_fusion(src, dst, pad):
    result = []
    k, j = 0, 0
    info = [list(x) for x in list(zip(src, dst, pad))]
    length = len(info)

    while k <= length - 1:
        if k == length - 1:
            result.append(info[k])
            break

        j = k + 1
        while j <= length - 1:
            is_serial = info[j][1] == info[j - 1][1] + 1
            is_same_type = info[j][2] == info[j - 1][2]
            if not is_serial or not is_same_type:
                break
            j += 1

        result.append(info[k])
        k = j
    return result


def _do_shape_fusion(shape, perm):
    result = []
    k = 0
    while k <= len(perm) - 1:
        if k == len(perm) - 1 and perm[k] == len(shape) - 1:
            result.append(shape[perm[k]])
            break

        begin = perm[k]
        end = perm[k + 1] if k + 1 < len(perm) else len(shape)
        dim = 1
        for i in range(begin, end):
            dim = UNKNOWN_DIM if shape[i] == UNKNOWN_DIM or dim == UNKNOWN_DIM else dim * shape[i]
        result.append(dim)
        k += 1
    return result


def _eliminate_one(src_shape, dst_shape, src, dst, pad):
    # 1. Only work in Const (Don't do infer shape).
    # 2. Pad axis can not do eliminate.
    # 3. Template is forward.
    if not _is_const(src_shape) or not _is_const(dst_shape):
        src_shape = [UNKNOWN_DIM] * len(src_shape)
        return src_shape, dst_shape, src, dst, pad

    n_src, n_dst, n_pad = [], [], []
    for k, v in enumerate(src):
        if src_shape[v] == 1 and not pad[k]:
            continue
        n_pad.append(pad[k])
        n_src.append(v)
        n_dst.append(dst[k])

    back = sorted(n_dst)
    dst = [back.index(x) for x in n_dst]
    dst_shape = [dst_shape[x] for x in back]

    pad = n_pad
    src_shape = [src_shape[x] for x in set(n_src)]

    num = 0
    src = []
    for k, i in enumerate(n_src):
        if k == 0:
            src.append(num)
            num += 1
            continue

        if n_src[k - 1] == i:
            src.append(num - 1)
            continue

        src.append(num)
        num += 1

    return src_shape, dst_shape, src, dst, pad


def _is_const(shape):
    return UNKNOWN_DIM not in shape
