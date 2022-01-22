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
build
"""
from __future__ import absolute_import as _abs

from tbe.common.platform import ASCEND_310
from tbe.common.platform import HI3796CV300CS
from tbe.common.platform import HI3796CV300ES
from tbe.common.platform import SD3403
from tbe.common.platform import SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from tbe.dsl.static_schedule.cce_schedule import cce_build_code as static_build

from .unify_auto_schedule import build as dynamic_build


def _check_dynamic_build_unsupported_versions(context):
    unsupported_versions = [ASCEND_310, HI3796CV300CS, HI3796CV300ES, SD3403]
    soc_version = get_soc_spec(SOC_VERSION)
    if soc_version not in unsupported_versions:
        return

    op_type_white_list = ["GlobalLpPool", "LpNormReduce", "LpNormUpdate", "ReduceMeanWithCount", "ReduceStdWithMean",
                          "SyncBNTrainingUpdate", "SyncBatchNormBackwardElemt", "SyncBatchNormBackwardReduce",
                          "SyncBatchNormGatherStatsWithCounts", "BNTrainingUpdate", "LayerNorm", "SoftmaxV2",
                          "SoftmaxGrad"]
    # fusion
    op_infos = operation.get_op_context().get_op_info()
    if len(op_infos) >= 2:
        return

    op_type = context.get_op_type()
    if op_type not in op_type_white_list:
        dict_args = {"errCode": "E90003",
                     "detailed_cause": "The dynamic process don't support current version."}
        raise RuntimeError(dict_args, get_error_message(dict_args))


def build(sch, config_map=None):
    """
    :param sch:
    :param config_map:
    :return:
    """
    _add_tensor_list_to_context(config_map)

    context = operation.get_context()
    if context is not None and context.get_mode() in ("dynamic", "static"):
        if context.get_mode() == "static":
            _check_dynamic_build_unsupported_versions(context)
        return dynamic_build(sch, config_map)

    sch = sch[0] if isinstance(sch, list) else sch
    tensors = config_map.get("tensor_list", [])
    if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
        config_map["tensor_list"] = tensors[0]
    return static_build(sch, config_map)


def _add_tensor_list_to_context(config_map):
    context = operation.get_op_context()
    if context and config_map and "tensor_list" in config_map:
        context.add_addition("tensor_list", config_map["tensor_list"])
