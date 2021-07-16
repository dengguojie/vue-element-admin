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
util for classifier
"""
from functools import reduce

VAR_BOUND_LIMIT = 2147483647


def generate_range(shape):
    """
    generate range by shape
    :param shape:
    :return:
    """
    return [(1, None) if v == -1 else (v, v) for v in shape]


def combine_dim(dims):
    """
    combine dim
    :param dims:
    :return:
    """
    return reduce(lambda a, b: -1 if -1 in (a, b) else a * b, dims)


def combine_range(ranges):
    """
    combine range
    :param ranges:
    :return:
    """

    def mul_ele(_a, _b):
        if _a is None or _b is None:
            return None
        _bound = _a * _b
        return VAR_BOUND_LIMIT if _bound > VAR_BOUND_LIMIT else _bound

    def mul(range1, range2):
        return [mul_ele(a, b) for a, b in zip(range1, range2)]

    return tuple(reduce(lambda r1, r2: mul(r1, r2), ranges))
