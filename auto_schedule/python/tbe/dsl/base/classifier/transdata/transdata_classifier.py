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
TransdataClassify
"""
import copy
from abc import ABC
from abc import abstractmethod
from tbe.common.utils.errormgr import get_error_message

_classifies = {}


def classify(ins: list):
    from tbe.common.buildcfg import get_current_build_config
    if get_current_build_config("enable_op_prebuild"):
        return [ins]

    src_shape = ins[0].get("shape")
    dst_shape = ins[1]

    if len(src_shape) == len(dst_shape):
        dict_args = {"errCode": "E90001",
                     "detailed_cause": "Pad or DePad is required in transdata,"
                                       "but length of src is equal to dst"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    category = "forward" if len(src_shape) < len(dst_shape) else "backward"
    return _classifies.get(category)(ins).classify()


class TransdataClassify(ABC):
    """
    TransdataClassify
    """

    def __init_subclass__(cls, **kwargs):
        _classifies[cls.get_category()] = cls

    def __init__(self, ins):
        self._ins = copy.deepcopy(ins)
        self._src_shape = self._ins[0].get("shape")
        self._dst_shape = self._ins[1]
        self._axes_map = self._ins[2]
        self._dtype = self._ins[0].get("dtype")

    @classmethod
    @abstractmethod
    def get_category(cls):
        """"""

    @abstractmethod
    def classify(self):
        """"""
