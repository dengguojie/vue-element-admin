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
unify schedule
"""
from . import elewise_schedule, elewise_tilingcase
from . import unify_broadcast
from . import reduce_schedule
from . import reduce_tilingcase
from . import norm_schedule
from . import norm_tilingcase
from . import softmax_schedule, softmax_tilingcase
from . import softmax_cross_entropy_with_logits_schedule, softmax_cross_entropy_with_logits_tilingcase
from . import layer_norm_x_backprop_tilingcase, layer_norm_x_backprop_schedule
from . import layer_norm_x_backprop_v2_tilingcase, layer_norm_x_backprop_v2_schedule
from . import layer_norm_beta_gamma_backprop_schedule, layer_norm_beta_gamma_backprop_tilingcase
from . import layer_norm_beta_gamma_backprop_v2_schedule, layer_norm_beta_gamma_backprop_v2_tilingcase
from .constants import Pattern, INSN_MAPPING, SUPPORT_SCALAR_INSNS, BROADCAST_INSNS, \
    DTYPE_BYTE_MAPPING, FAKE_NODE_TAG, NEED_TEMP_SPACE_INSNS, CompileInfo, VSEL_INSNS, \
    VCMPSEL_INSNS, NEED_SPACE_WITH_DIFF_TYPE, TERNARY_INSNS, NEED_EXTENT_NODE_INSNS, VCMP_INSNS
from .unify_auto_schedule import build
from .unify_auto_schedule import schedule_cce

from . import layer_norm_tilingcase, layer_norm_normal_schedule, layer_norm_workspace_schedule
from . import bn_update_grad_schedule, bn_training_update_grad_tilingcase
from . import confusion_softmax_grad_schedule, confusion_softmax_grad_tilingcase

# quant
from . import ascend_quant_schedule, ascend_quant_tilingcase
from . import ascend_anti_quant_schedule, ascend_anti_quant_tilingcase

# cube
from . import conv2d_bp_input_schedule
from . import conv2d_bp_input_tilingcase
from . import conv2d_bp_filter_tilingcase
from . import conv3d_schedule
from . import conv3d_tilingcase
from . import conv3d_bp_filter_tilingcase
from . import conv3d_bp_input_schedule
from . import conv3d_bp_input_tilingcase
from . import cube_schedule
from . import gemm_tilingcase
from . import conv2d_schedule
from . import conv2d_tilingcase
