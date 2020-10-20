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
TVM cce runtime
"""
from __future__ import absolute_import as _abs

if __name__ == "platform":
    import sys
    import os

    # te warning: using python build-in 'platform'
    TP = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    BAK_PATH = sys.path[:]
    for item in BAK_PATH:
        if (item == '' or os.path.realpath(item) == TP) and item in sys.path:
            sys.path.remove(item)

    sys.modules.pop('platform')
    import platform

    sys.path.insert(0, '')
    sys.path.append(TP)
else:
    from .cce_buffer import cur_cce_product_params
    from .cce_params import scope_cbuf
    from .cce_params import scope_ubuf
    from .cce_params import scope_ca
    from .cce_params import scope_cb
    from .cce_params import scope_cc
    from .cce_params import scope_reg
    from .cce_params import scope_aicpu
    from .cce_params import dma_copy
    from .cce_params import dma_copy_global
    from .cce_params import CUBE_MKN
    from .cce_params import CONV_MODE
    from .cce_params import scope_gm
    from .cce_params import scope_cbuf_fusion
    from .cce_params import CCE_AXIS
    from .cce_params import WGT_WIDTH
    from .cce_params import INP_WIDTH
    from .cce_params import OUT_WIDTH
    from .cce_params import BLOCK_IN
    from .cce_params import BLOCK_OUT
    from .cce_params import BLOCK_REDUCE
    from .cce_params import INP_ELEM_BYTES
    from .cce_params import WGT_ELEM_BYTES
    from .cce_params import OUT_ELEM_BYTES
    from .cce_params import GLB_ELEM_BYTES
    from .cce_params import conv_buffer_ex
    from .cce_params import C0_SIZE
    from .cce_params import ELEMENTS_VECTOR_OP_FP16
    from .cce_params import BLOCK_REDUCE_INT8
    from .cce_params import VECTOR_INST_BLOCK_WIDTH
    from .cce_params import VECTOR_INST_BLOCK_NUM
    from .cce_params import HI3796CV300ES
    from .cce_params import HI3796CV300CS
    from .cce_params import ASCEND_310
    from .cce_params import ASCEND_910
    from .cce_params import ASCEND_610
    from .cce_params import ASCEND_710
    from . import cce_intrin
    from .cce_intrin import get_bit_len
    from . import cce_intrin_md
    from .cce_build import get_pass_list
    from .cce_build import build_config
    from .cce_build import build_config_update
    from .cce_conf import getValue
    from .cce_conf import CceProductParams
    from .cce_conf import set_status_check
    from .cce_conf import get_soc_spec
    from .cce_conf import intrinsic_check_support
    from .cce_conf import te_set_version
    from .cce_conf import api_check_support
    from .cce_conf import SOC_VERSION
    from .cce_conf import AICORE_TYPE
    from .cce_conf import CORE_NUM
    from .cce_conf import UB_SIZE
    from .cce_conf import L1_SIZE
    from .cce_conf import L0A_SIZE
    from .cce_conf import is_lhisi_version
    from .cce_conf import L0A_SIZE
    from .cce_emitinsn_params import CceEmitInsnParams
    from .cce_policy import set_L1_info
    from .cce_policy import get_L1_info
    from .cce_policy import disableL2
    from . import cce_runtime
    from .operation import register_fusion_compute
    from .operation import register_operator
    from .operation import compute
    from .operation import add_compile_info
    from .operation import var
    from .operation import add_exclude_bound_var
    from .operation import register_tiling_case
    from .operation import register_schedule
    from .operation import get_te_var
    from .shape_classifier import classify
    from .shape_classifier import Mode
    from .insn_cmd import DMA_COPY
    from .insn_cmd import DATA_MOV
    from .insn_cmd import PHONY_INSN
    from .insn_cmd import IM2COL
    from .insn_cmd import CAST
    from .insn_cmd import SELECT
    from .insn_cmd import MUL
    from .insn_cmd import MAD
    from .insn_cmd import DUP
    from .insn_cmd import SET_FMATRIX
    from .insn_cmd import REDUCE_SUM
    from .insn_cmd import DMA_PADDING
    from .log_util import except_msg
    from .fusion_manager import get_fusion_build_cfg
