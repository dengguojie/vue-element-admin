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
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

Description:
"""
from enum import IntEnum, auto
import tbe
from tbe.common.context import get_context
from tbe.common.utils import log

# conv2d_compress support alg
WEIGHT_UNZIP = "weight_unzip"
WEIGHT_SPARSE_4_2 = "weight_sparse_4_2"
COMPRESS_ALG_SUPPORT = [WEIGHT_UNZIP, WEIGHT_SPARSE_4_2]

# binary var range max
TILING_DIM_MAX = 32
CONV2D_DILATION_MAX = 255
CONV2D_STRIDE_MAX = 63
CONV2D_PAD_MAX = 255
CONV2D_KERNEL_MAX = 255
BINARY_CONFIG_KEY = "multiple_templates"

# mkn index
CUBE_MKN_IDX_M = 0
CUBE_MKN_IDX_K = 1
CUBE_MKN_IDX_N = 2


class AttachMode(object):
    """
    binary attach mode
    """
    ATTACH_FULL_LOAD = 0
    ATTACH_PASS = 3
    ATTACH_RES = 1
    ATTACH_CL0 = 2
    ATTACH_CL0_FOR_BL0 = 1


class Conv2dTensorName(object):
    """
    conv2d tensor map key.
    """
    # weight
    FILTER = "filter"
    BL0 = "bl0"
    WEIGHT_INDEX = "weight_index"

    # fm
    FMAP = "fmap"
    FMAP_IMG2COL = "fmap_im2col"
    FMAP_L1 = "fmap_l1"
    FMAP_ROW_MAJOR = "fmap_row_major"
    FMAP_RAW_MAJOR_RESHAPE = "fmap_row_major_reshape"

    # bias
    BIAS_L1 = "bias_l1"
    BIAS_BT = "bias_bt"
    BIAS_UB = "bias_ub"

    # mad
    CL0 = "mad1"

    # ub
    CUB = "cub"
    CUB_BIAS_ADD = "cub_bias_add"


class TilingDataIdx(IntEnum):
    """
    tiling data define index
    """
    TILINGDATA_IDX_START = 0
    IDX_BATCH_N = TILINGDATA_IDX_START
    IDX_C_IN = auto()
    IDX_FMAP_H = auto()
    IDX_FMAP_W = auto()
    IDX_C_OUT = auto()
    IDX_K_H = auto()
    IDX_K_W = auto()
    IDX_DILATION_H = auto()
    IDX_DILATION_W = auto()
    IDX_STRIDE_H = auto()
    IDX_STRIDE_W = auto()
    IDX_HO = auto()
    IDX_WO = auto()
    IDX_PAD_TOP = auto()
    IDX_PAD_BOTTOM = auto()
    IDX_PAD_LEFT = auto()
    IDX_PAD_RIGHT = auto()
    IDX_BATCH_SINGLE_CORE = auto()
    IDX_N_SINGLE_CORE = auto()
    IDX_BATCH_DIM = auto()
    IDX_N_DIM = auto()
    IDX_M_DIM = auto()
    IDX_GROUP_DIM = auto()
    IDX_AUB_H = auto()
    IDX_AUB_CI1 = auto()
    IDX_CUB_N1 = auto()
    IDX_N_UB_L0C_FACTOR = auto()
    IDX_M_L0 = auto()
    IDX_K_L0 = auto()
    IDX_M_AL1_FACTOR = auto()
    IDX_N_BL1_FACTOR = auto()
    IDX_KAL1_16 = auto()
    IDX_KBL1_16 = auto()
    IDX_KAL1_FACTOR = auto()
    IDX_KBL1_FACTOR = auto()
    TILINGDATA_IDX_END = auto()


class TilingDataKey(object):
    """
    tiling data key string.
    """
    BATCH_SINGLE_CORE = "batch_single_core"
    N_SINGLE_CORE = "n_single_core"
    BATCH_DIM = "batch_dim"
    N_DIM = "n_dim"
    M_DIM = "m_dim"
    GROUP_DIM = "group_dim"
    AUB_H = "aub_h"
    AUB_CI1 = "aub_ci1"
    CUB_N1 = "cub_n1"
    N_UB_L0C_FACTOR = "n_ub_l0c_factor"
    M_L0 = "m_l0"
    K_L0 = "k_l0"
    M_AL1_FACTOR = "m_al1_factor"
    N_BL1_FACTOR = "n_bl1_factor"
    KAL1_16 = "kal1_16"
    KBL1_16 = "kbl1_16"
    KAL1_FACTOR = "kal1_factor"
    KBL1_FACTOR = "kbl1_factor"
    DILATION_H = "dilation_h"
    DILATION_W = "dilation_w"
    STRIDE_H = "stride_h"
    STRIDE_W = "stride_w"
    C_IN = "c_in"
    C_OUT = "c_out"
    K_H = "k_h"
    K_W = "k_w"
    PAD_TOP = "pad_top"
    PAD_BOTTOM = "pad_bottom"
    PAD_LEFT = "pad_left"
    PAD_RIGHT = "pad_right"
    BATCH_N = "batch_n"
    FMAP_H = "fmap_h"
    FMAP_W = "fmap_w"
    HO = "ho"
    WO = "wo"


TILINGDATA_KEY_MAP = {
    TilingDataIdx.IDX_BATCH_SINGLE_CORE: TilingDataKey.BATCH_SINGLE_CORE,
    TilingDataIdx.IDX_N_SINGLE_CORE: TilingDataKey.N_SINGLE_CORE,
    TilingDataIdx.IDX_BATCH_DIM: TilingDataKey.BATCH_DIM,
    TilingDataIdx.IDX_N_DIM: TilingDataKey.N_DIM,
    TilingDataIdx.IDX_M_DIM: TilingDataKey.M_DIM,
    TilingDataIdx.IDX_GROUP_DIM: TilingDataKey.GROUP_DIM,
    TilingDataIdx.IDX_AUB_H: TilingDataKey.AUB_H,
    TilingDataIdx.IDX_AUB_CI1: TilingDataKey.AUB_CI1,
    TilingDataIdx.IDX_CUB_N1: TilingDataKey.CUB_N1,
    TilingDataIdx.IDX_N_UB_L0C_FACTOR: TilingDataKey.N_UB_L0C_FACTOR,
    TilingDataIdx.IDX_M_L0: TilingDataKey.M_L0,
    TilingDataIdx.IDX_K_L0: TilingDataKey.K_L0,
    TilingDataIdx.IDX_M_AL1_FACTOR: TilingDataKey.M_AL1_FACTOR,
    TilingDataIdx.IDX_N_BL1_FACTOR: TilingDataKey.N_BL1_FACTOR,
    TilingDataIdx.IDX_KAL1_16: TilingDataKey.KAL1_16,
    TilingDataIdx.IDX_KBL1_16: TilingDataKey.KBL1_16,
    TilingDataIdx.IDX_KAL1_FACTOR: TilingDataKey.KAL1_FACTOR,
    TilingDataIdx.IDX_KBL1_FACTOR: TilingDataKey.KBL1_FACTOR,
    TilingDataIdx.IDX_DILATION_H: TilingDataKey.DILATION_H,
    TilingDataIdx.IDX_DILATION_W: TilingDataKey.DILATION_W,
    TilingDataIdx.IDX_STRIDE_H: TilingDataKey.STRIDE_H,
    TilingDataIdx.IDX_STRIDE_W: TilingDataKey.STRIDE_W,
    TilingDataIdx.IDX_C_IN: TilingDataKey.C_IN,
    TilingDataIdx.IDX_C_OUT: TilingDataKey.C_OUT,
    TilingDataIdx.IDX_K_H: TilingDataKey.K_H,
    TilingDataIdx.IDX_K_W: TilingDataKey.K_W,
    TilingDataIdx.IDX_PAD_TOP: TilingDataKey.PAD_TOP,
    TilingDataIdx.IDX_PAD_BOTTOM: TilingDataKey.PAD_BOTTOM,
    TilingDataIdx.IDX_PAD_LEFT: TilingDataKey.PAD_LEFT,
    TilingDataIdx.IDX_PAD_RIGHT: TilingDataKey.PAD_RIGHT,
    TilingDataIdx.IDX_BATCH_N: TilingDataKey.BATCH_N,
    TilingDataIdx.IDX_FMAP_H: TilingDataKey.FMAP_H,
    TilingDataIdx.IDX_FMAP_W: TilingDataKey.FMAP_W,
    TilingDataIdx.IDX_HO: TilingDataKey.HO,
    TilingDataIdx.IDX_WO: TilingDataKey.WO,
}


TILINGDATA_KEY_RANGE_MAP = {
    TilingDataKey.BATCH_SINGLE_CORE: [1, 64],
    TilingDataKey.N_SINGLE_CORE: [1, 64],
    TilingDataKey.BATCH_DIM: [1, TILING_DIM_MAX],
    TilingDataKey.N_DIM: [1, TILING_DIM_MAX],
    TilingDataKey.M_DIM: [1, TILING_DIM_MAX],
    TilingDataKey.GROUP_DIM: [1, TILING_DIM_MAX],
    TilingDataKey.AUB_H: [1, 16],
    TilingDataKey.AUB_CI1: [1, 16],
    TilingDataKey.CUB_N1: [1, 128],
    TilingDataKey.N_UB_L0C_FACTOR: [1, 64],
    TilingDataKey.M_L0: [1, 128],
    TilingDataKey.K_L0: [1, 128],
    TilingDataKey.M_AL1_FACTOR: [1, 1024],
    TilingDataKey.N_BL1_FACTOR: [1, 1024],
    TilingDataKey.KAL1_16: [1, 64],
    TilingDataKey.KBL1_16: [1, 64],
    TilingDataKey.KAL1_FACTOR: [1, 64],
    TilingDataKey.KBL1_FACTOR: [1, 64],
    TilingDataKey.DILATION_H: [1, CONV2D_DILATION_MAX],
    TilingDataKey.DILATION_W: [1, CONV2D_DILATION_MAX],
    TilingDataKey.STRIDE_H: [1, CONV2D_STRIDE_MAX],
    TilingDataKey.STRIDE_W: [1, CONV2D_STRIDE_MAX],
    TilingDataKey.C_IN: [1, 1024],
    TilingDataKey.C_OUT: [1, 1024],
    TilingDataKey.K_H: [1, CONV2D_KERNEL_MAX],
    TilingDataKey.K_W: [1, CONV2D_KERNEL_MAX],
    TilingDataKey.PAD_TOP: [1, CONV2D_PAD_MAX],
    TilingDataKey.PAD_BOTTOM: [1, CONV2D_PAD_MAX],
    TilingDataKey.PAD_LEFT: [1, CONV2D_PAD_MAX],
    TilingDataKey.PAD_RIGHT: [1, CONV2D_PAD_MAX],
    TilingDataKey.FMAP_H: [1, None],
    TilingDataKey.FMAP_W: [1, None],
    TilingDataKey.HO: [1, None],
    TilingDataKey.WO: [1, None],
    TilingDataKey.BATCH_N: [1, None]
}


class BinaryTilingKey(object):
    """
    binary attach key
    """
    BL0_ATTACH_FLAG = "bl0_attach_flag"
    AL1_ATTACH_FLAG = "al1_attach_flag"
    BL1_ATTACH_FLAG = "bl1_attach_flag"


class KernelIdKeyOffset(object):
    """
    kernel id key bit offset
    """
    OFFSET_AL1 = 0
    OFFSET_BL1 = 3
    OFFSET_BL0 = 6
    OFFSET_BATCH_SPLIT = 9
    OFFSET_GROUP_SPLIT = 11
    OFFSET_CUB_CHANNEL_WISE = 13
    OFFSET_LOAD2D = 15 # 15-16


# tiling template support list
TILING_ATTACH_SUPPORT_LIST = [
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_FULL_LOAD,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_PASS,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_FULL_LOAD,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_RES,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_CL0,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_PASS,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_RES,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_PASS,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_RES,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_RES,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_CL0,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_RES,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_CL0,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_FULL_LOAD,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_RES,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_FULL_LOAD,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_CL0_FOR_BL0},
    {BinaryTilingKey.AL1_ATTACH_FLAG: AttachMode.ATTACH_RES,
     BinaryTilingKey.BL1_ATTACH_FLAG: AttachMode.ATTACH_PASS,
     BinaryTilingKey.BL0_ATTACH_FLAG: AttachMode.ATTACH_FULL_LOAD},
]


class BinaryInfoKey(object):
    """
    binary compile feature flag.
    """
    LOAD2D_FLAG = "load2d_flag"


def get_binary_infos():
    """
    get binary compile feature infos.
    """
    conv2d_binary_infos_tmp = {}
    op_infos = get_context().get_op_info(None)
    if len(op_infos) != 0:
        extra_params = op_infos[0].extra_params
        if len(extra_params):
            conv2d_binary_infos_tmp = extra_params.get(BINARY_CONFIG_KEY, {})
    conv2d_binary_infos = {}
    conv2d_binary_infos[BinaryInfoKey.LOAD2D_FLAG] = conv2d_binary_infos_tmp.get(
        BinaryInfoKey.LOAD2D_FLAG, False)
    return conv2d_binary_infos


def show_class_var(class_obj):
    """
    get all tensor_name defined
    """
    tensor_name_vars = [attr for attr in dir(class_obj) if
                        not callable(getattr(class_obj, attr)) and not attr.startswith("__")]
    log.debug("show class variable of class [{}]".format(class_obj.__name__))
    for var in tensor_name_vars:
        log.debug("[{}.{}]:{}".format(class_obj.__name__, var, class_obj.__dict__.get(var)))


def is_support_fixpipe():
    """
    Check fixpipe support.
    """
    return tbe.common.platform.platform_info.intrinsic_check_support("Intrinsic_fix_pipe_unit_list")


def get_cur_soc():
    """
    get soc version
    """
    return tbe.common.platform.platform_info.get_soc_spec("SOC_VERSION")
