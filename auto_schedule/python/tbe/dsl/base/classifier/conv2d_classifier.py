#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2022-2022 Huawei Technologies Co., Ltd
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
classifier of shape in conv2d
"""

import copy
from enum import Enum
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.op_util.op_util_conv2d import BinaryInfoKey

KERNEL_H_INDEX = 0
KERNEL_W_INDEX = 1
ZERO_INS_INDEX = 0

ATTR_STRIDE_INDEX = 0
ATTR_PAD_INDEX = 1
ATTR_DILATION_INDEX = 2
ATTR_GROUP_INDEX = 3
ATTR_FORMAT_INDEX = 4
ATTR_OFFSETX_INDEX = 5

LOAD2D_KERNEL = "kernel"

COMPUTE_MULTI_BRANCH = [BinaryInfoKey.LOAD2D_FLAG, BinaryInfoKey.LOAD3D_FLAG, BinaryInfoKey.DMA_FLAG]
CONV_INS_ATTR = {ATTR_STRIDE_INDEX: "strides",
                 ATTR_PAD_INDEX: "pads",
                 ATTR_DILATION_INDEX: "dilation",
                 ATTR_GROUP_INDEX: "groups",
                 ATTR_OFFSETX_INDEX: "offset_x"}

CLASSIFY_INS_MAP = {
    "load2d": {"kernel": [1, 1], "pads": [0, 0, 0, 0], "strides": [1, 1, 1, 1], BinaryInfoKey.LOAD2D_FLAG: 1,
                BinaryInfoKey.LOAD3D_FLAG: 0, BinaryInfoKey.DMA_FLAG: 0},
    "load3d": {BinaryInfoKey.LOAD2D_FLAG: 0, BinaryInfoKey.LOAD3D_FLAG: 1, BinaryInfoKey.DMA_FLAG: 0},
    "dma": {BinaryInfoKey.LOAD2D_FLAG: 0, BinaryInfoKey.LOAD3D_FLAG: 0, BinaryInfoKey.DMA_FLAG: 1}
}


class BaseComputeMode(object):
    """
    compute classify public func
    """
    def __init__(self, ins: list or tuple, compute_paras: dict):
        self.input_list, self.attr_list, self.option_list = ins
        self.compute_paras = compute_paras
        self.attr_conv_idx = self.get_conv_attr_index()
        self.option_conv_idx = self.get_conv_option_index()
        self.attr_paras = self.get_attr_paras()
        self.option_paras = self.get_option_paras()
        self.set_attr_none()

    def get_conv_attr_index(self):
        for idx, value in enumerate(self.attr_list):
            if "name" in value and "Conv" in value.get("name"):
                return idx

        dict_args = {"errCode": "E90001", "detailed_cause": "get attr idx failed"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def get_conv_option_index(self):
        for idx, value in enumerate(self.option_list):
            if "name" in value and "Conv" in value.get("name"):
                return idx

        dict_args = {"errCode": "E90001", "detailed_cause": "get option idx failed"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def get_attr_paras(self):
        if self.attr_list[self.attr_conv_idx].get("val"):
            return self.attr_list[self.attr_conv_idx].get("val")

        dict_args = {"errCode": "E90001", "detailed_cause": "get conv attr failed"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def get_option_paras(self):
        if self.option_list[self.option_conv_idx].get("options"):
            return self.option_list[self.option_conv_idx].get("options")

        dict_args = {"errCode": "E90001", "detailed_cause": "get conv option failed"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def set_attr_none(self):
        "attr is None for binary mode"
        for idx, _ in CONV_INS_ATTR.items():
            if idx != ATTR_GROUP_INDEX and self.attr_paras[idx] is not None:
                self.attr_paras[idx] = None

    def update_options(self):
        for key in COMPUTE_MULTI_BRANCH:
            self.option_paras[key] = self.compute_paras.get(key, 0)

    def mode_update(self):
        return self.update_options()



class Load2dComputeMode(BaseComputeMode):
    def __init__(self, ins: list or tuple, compute_paras: dict):
        super().__init__(ins, compute_paras)

    def update_attrs(self):
        for idx, key in CONV_INS_ATTR.items():
            if self.compute_paras.get(key):
                self.attr_paras[idx] = self.compute_paras.get(key)

    def update_options(self):
        for key in COMPUTE_MULTI_BRANCH:
            self.option_paras[key] = self.compute_paras.get(key, 0)
        self.option_paras[LOAD2D_KERNEL] = self.compute_paras.get(LOAD2D_KERNEL)

    def mode_update(self):
        self.update_attrs()
        self.update_options()
        return


class Load3dComputeMode(BaseComputeMode):
    def __init__(self, ins: list or tuple, compute_paras: dict):
        super().__init__(ins, compute_paras)


class DmaComputeMode(BaseComputeMode):
    def __init__(self, ins: list or tuple, compute_paras: dict):
        super().__init__(ins, compute_paras)


compute_calltbl = {
    "load2d": Load2dComputeMode,
    "load3d": Load3dComputeMode,
    "dma": DmaComputeMode
}


def check_binary_mode(ins: list or tuple):
    input_list, _, _ = ins
    fmap_ori_shape = input_list[0]["ori_shape"]
    fmap_ori_range = input_list[0]["ori_range"]
    for idx, _ in enumerate(fmap_ori_shape):
        if fmap_ori_shape[idx] != -1 or fmap_ori_range[idx] != [1, -1]:
            return False

    return True


def classify(ins: list or tuple, extra_params: dict):
    outs = []
    if not check_binary_mode(ins):
        # no need active classify func
        outs.append(ins)
        return outs

    for key, func in compute_calltbl.items():
        temp_ins = copy.deepcopy(ins)
        compute_paras = CLASSIFY_INS_MAP.get(key)

        func(temp_ins, compute_paras).mode_update()

        outs.append(temp_ins)

    return outs

