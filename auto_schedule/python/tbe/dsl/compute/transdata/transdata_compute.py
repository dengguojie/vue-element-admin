#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
transdata compute
"""
import abc

from tbe.common.utils.shape_util import shape_to_list
from tbe.dsl.base import operation

_computes = {}


def transdata(tensor, dst_shape, axes_map, pad_value):
    """
    transdata interface that choose different computes by _transdata_category
    """
    category = operation.get_context().get_current_compute().get("_transdata_category")
    return _computes.get(category)(tensor, dst_shape, axes_map, pad_value).do_compute()


class TransdataComputation(abc.ABC):
    """
    TransdataComputation
    """

    def __init_subclass__(cls, **kwargs):
        _computes[cls.get_category()] = cls

    @abc.abstractmethod
    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        """"""
        self._tensor = tensor
        self._dst_shape = dst_shape
        self._axes_map = axes_map
        self._pad_value = pad_value
        self._src_shape = shape_to_list(tensor.shape)
        self._dtype = tensor.dtype

    @classmethod
    @abc.abstractmethod
    def get_category(cls):
        """"""

    @abc.abstractmethod
    def do_compute(self):
        """"""
