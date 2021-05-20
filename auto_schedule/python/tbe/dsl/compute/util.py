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
util
"""
# pylint: disable=import-error
from decorator import decorator
from tbe import tvm
from tbe.common.platform import ASCEND_310
from tbe.common.platform import ASCEND_610
from tbe.common.platform import ASCEND_615
from tbe.common.platform import ASCEND_710
from tbe.common.platform import ASCEND_910
from tbe.common.platform import HI3796CV300CS
from tbe.common.platform import HI3796CV300ES
from tbe.common.platform import SD3403
from tbe.common.platform import ASCEND_920A
from tbe.common.platform import SOC_VERSION
from tbe.common.platform import intrinsic_check_support
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils import shape_util
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation

ASCEND_SHISI = "smallhisi"

# Save op's output dtype, when first call the template api,we will save the dtype.
# Before auto scheduling,get the dtype and convert the res tensor to this dtype,
# and set the dtype to None.

DTYPE_MAP = {
    "float32": "f32",
    "float16": "f16",
    "int8": "s8",
    "uint8": "u8",
    "int32": "s32",
}

DSL_SAME_API_MAP = {
    "sum": "reduce_sum",
    "round_to": "clip",
    "round_d": "round_half_up"
}

DSL_CHECK_SUPPORT_MAP = {
    "broadcast": {
        "AllSoc": ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
        ASCEND_310: ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
        ASCEND_910: ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
        ASCEND_920A: ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
        ASCEND_710: ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
        ASCEND_610: ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
        ASCEND_615: ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
        ASCEND_SHISI: ("float16", "float32", "int32", "uint32", "int16", 
        "uint16", "int8", "uint8"),
    },

    # segment
    "unsorted_segment_sum": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },
    "unsorted_segment_mean": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },
    "unsorted_segment_prod": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32", "int16"),
        ASCEND_610: ("float16", "float32", "int32", "int16"),
        ASCEND_615: ("float16", "float32", "int32", "int16"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },
    "unsorted_segment_min": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32", "int16"),
        ASCEND_610: ("float16", "float32", "int32", "int16"),
        ASCEND_615: ("float16", "float32", "int32", "int16"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },
    "unsorted_segment_max": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32", "int16"),
        ASCEND_610: ("float16", "float32", "int32", "int16"),
        ASCEND_615: ("float16", "float32", "int32", "int16"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },

    # inplace
    "inplace_add": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },
    "inplace_sub": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },
    "inplace_update": {
        "AllSoc": ("float16", "float32", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "float32", "int32"),
    },

    # ceil/floor/round/trunc
    "ceil": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "floor": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "round": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "trunc": {
        "AllSoc": ("float16",),
        ASCEND_310: (),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "round_half_up": {
        "AllSoc": ("float16",),
        ASCEND_310: (),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },

    # reduce
    "reduce_sum": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),  # int32: nlst support, last not
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16",),
    },
    "reduce_max": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),  # fp32:last need priority_flag
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"), # v200 int32: nlst support, last not
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "reduce_min": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),  # fp32:last need priority_flag
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "reduce_prod": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),  # int32: nlst/last support
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16",),
    },

    # elewise
    "vadd": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vsub": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vmul": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32", "int16"),
        ASCEND_610: ("float16", "float32", "int32", "int16"),
        ASCEND_615: ("float16", "float32", "int32", "int16"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vdiv": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32",),
        ASCEND_910: ("float16", "float32",),
        ASCEND_920A: ("float16", "float32",),
        ASCEND_710: ("float16", "float32",),
        ASCEND_610: ("float16", "float32",),
        ASCEND_615: ("float16", "float32",),
        ASCEND_SHISI: ("float16",),
    },
    "vmod": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vmin": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vmax": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vadds": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vmins": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vmaxs": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vmuls": {
        "AllSoc": ("float16", "int32"),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16", "int32"),
    },
    "vnot": {
        "AllSoc": ("int16", "uint16"),
        ASCEND_310: ("int16", "uint16"),
        ASCEND_910: ("int16", "uint16"),
        ASCEND_920A: ("int16", "uint16"),
        ASCEND_710: ("int16", "uint16"),
        ASCEND_610: ("int16", "uint16"),
        ASCEND_615: ("int16", "uint16"),
        ASCEND_SHISI: ("int16", "uint16"),
    },
    "vor": {
        "AllSoc": ("int16", "uint16"),
        ASCEND_310: ("int16", "uint16"),
        ASCEND_910: ("int16", "uint16"),
        ASCEND_920A: ("int16", "uint16"),
        ASCEND_710: ("int16", "uint16"),
        ASCEND_610: ("int16", "uint16"),
        ASCEND_615: ("int16", "uint16"),
        ASCEND_SHISI: ("int16", "uint16"),
    },
    "vand": {
        "AllSoc": ("int16", "uint16"),
        ASCEND_310: ("int16", "uint16"),
        ASCEND_910: ("int16", "uint16"),
        ASCEND_920A: ("int16", "uint16"),
        ASCEND_710: ("int16", "uint16"),
        ASCEND_610: ("int16", "uint16"),
        ASCEND_615: ("int16", "uint16"),
        ASCEND_SHISI: ("int16", "uint16"),
    },
    "vcmp": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vlogic": {
        "AllSoc": ("bool",),
        ASCEND_310: ("bool",),
        ASCEND_910: ("bool",),
        ASCEND_920A: ("bool",),
        ASCEND_710: ("bool",),
        ASCEND_610: ("bool",),
        ASCEND_615: ("bool",),
        ASCEND_SHISI: ("bool",),
    },
    "vsel": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vcmpsel": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vlog": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vexp": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vabs": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vrec": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vrelu": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16",),
        ASCEND_920A: ("float16",),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vsqrt": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vrsqrt": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vaxpy": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vmla": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vmadd": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vmaddrelu": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32"),
        ASCEND_910: ("float16", "float32"),
        ASCEND_920A: ("float16", "float32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vlrelu": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "vaddrelu": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16",),
        ASCEND_920A: ("float16",),
        ASCEND_710: ("int16", "float16", "float32"),
        ASCEND_610: ("int16", "float16", "float32"),
        ASCEND_615: ("int16", "float16", "float32"),
        ASCEND_SHISI: ("int16", "float16",),
    },
    "vsubrelu": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16",),
        ASCEND_920A: ("float16",),
        ASCEND_710: ("int16", "float16", "float32"),
        ASCEND_610: ("int16", "float16", "float32"),
        ASCEND_615: ("int16", "float16", "float32"),
        ASCEND_SHISI: ("int16", "float16",),
    },

    # common
    "clip": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32"),  # int32: schedule not support
        ASCEND_610: ("float16", "float32"),  # int32: schedule not support
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),  # int32: schedule not support
    },
    "cast_to": {
        "AllSoc": ("f162f32", "f162s8", "f162u8", "f162s32",
                   "s82f16", "s82u8", "u82f16", "u82s8",
                   "s322f16", "s322s8", "s322u8", "s322f32"),
        ASCEND_310: ("f322f16", "f322s8", "f322u8", "f322s32",
                     "f162f32", "f162s8", "f162u8", "f162s32",
                     "s82f16", "s82u8", "u82f16", "u82s8",
                     "s322f16", "s322s8", "s322u8", "s322f32"),
        ASCEND_910: ("f322f16", "f322s8", "f322u8", "f322s32",
                     "f162f32", "f162s8", "f162u8", "f162s32",
                     "s82f16", "s82u8", "u82f16", "u82s8",
                     "s322f16", "s322s8", "s322u8", "s322f32"),
        ASCEND_920A: ("f322f16", "f322s8", "f322u8", "f322s32",
                      "f162f32", "f162s8", "f162u8", "f162s32",
                      "s82f16", "s82u8", "u82f16", "u82s8",
                      "s322f16", "s322s8", "s322u8", "s322f32"),
        ASCEND_710: ("f322f16", "f322s8", "f322u8", "f322s32",
                     "f162f32", "f162s8", "f162u8", "f162s32",
                     "s82f16", "s82u8", "u82f16", "u82s8",
                     "s322f16", "s322s8", "s322u8", "s322f32"),
        ASCEND_610: ("f322f16", "f322s8", "f322u8", "f322s32",
                     "f162f32", "f162s8", "f162u8", "f162s32",
                     "s82f16", "s82u8", "u82f16", "u82s8",
                     "s322f16", "s322s8", "s322u8", "s322f32"),
        ASCEND_615: ("f322f16", "f322s8", "f322u8", "f322s32",
                     "f162f32", "f162s8", "f162u8", "f162s32",
                     "s82f16", "s82u8", "u82f16", "u82s8",
                     "s322f16", "s322s8", "s322u8", "s322f32"),
        ASCEND_SHISI: ("f162f32", "f162s8", "f162u8", "f162s32",
                       "s82f16", "s82u8", "u82f16", "u82s8",
                       "s322f16", "s322s8", "s322u8", "s322f32"),
    },

    "conv": {
        "AllSoc": ("float16",),
    },
    "compute_four2five": {
        "AllSoc": ("float16",),
    },
    "compute_five2four": {
        "AllSoc": ("float16",),
    },
    "matmul": {
        "AllSoc": ("float16", "f162f16", "f162f32"),
    },
    "pooling2d": {
        "AllSoc": ("float16",),
    },
    "concat": {
        "AllSoc": ("int8", "int16", "int32", "int64",
                   "uint8", "uint16", "uint32", "uint64",
                   "float16", "float32"),
    },
}

UNIFY_DSL_CHECK_SUPPORT_MAP = {
    "reduce_prod": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16",),
    },
    "vsel": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16",),
        ASCEND_910: ("float16",),
        ASCEND_920A: ("float16",),
        ASCEND_710: ("float16", "float32"),
        ASCEND_610: ("float16", "float32"),
        ASCEND_615: ("float16", "float32"),
        ASCEND_SHISI: ("float16",),
    },
    "reduce_sum": {
        "AllSoc": ("float16",),
        ASCEND_310: ("float16", "float32", "int32"),
        ASCEND_910: ("float16", "float32", "int32"),
        ASCEND_920A: ("float16", "float32", "int32"),
        ASCEND_710: ("float16", "float32", "int32"),
        ASCEND_610: ("float16", "float32", "int32"),
        ASCEND_615: ("float16", "float32", "int32"),
        ASCEND_SHISI: ("float16",),
    },
}


def dsl_support_dtype(dsl_name):
    """
    dsl_support_dtype
    """
    if not isinstance(dsl_name, str):
        return []

    dsl_name = DSL_SAME_API_MAP.get(dsl_name, dsl_name)

    if in_dynamic_and_static_unify() and dsl_name in UNIFY_DSL_CHECK_SUPPORT_MAP:
        all_support_dtype = UNIFY_DSL_CHECK_SUPPORT_MAP.get(dsl_name)
    else:
        all_support_dtype = DSL_CHECK_SUPPORT_MAP.get(dsl_name)

    if all_support_dtype is None:
        return []

    soc_ver = get_soc_spec(SOC_VERSION)
    if soc_ver in (HI3796CV300CS, HI3796CV300ES, SD3403):
        soc_ver = ASCEND_SHISI
    soc_support_dtype = all_support_dtype.get(soc_ver)
    if soc_support_dtype is None:
        soc_support_dtype = all_support_dtype.get("AllSoc")
        if soc_support_dtype is None:
            return []

    return list(soc_support_dtype)


def dsl_check_support(dsl_api, dtype=None):
    """
    dsl_check_support
    """
    if not dsl_api.startswith("te.lang.cce.") and not dsl_api.startswith("tbe.dsl."):
        return False
    if (dtype is not None) and (not isinstance(dtype, str)):
        return False

    dsl_name = dsl_api.split(".")[-1]

    dsl_name = DSL_SAME_API_MAP.get(dsl_name, dsl_name)

    if in_dynamic_and_static_unify() and dsl_name in UNIFY_DSL_CHECK_SUPPORT_MAP:
        all_support_dtype = UNIFY_DSL_CHECK_SUPPORT_MAP.get(dsl_name)
    else:
        all_support_dtype = DSL_CHECK_SUPPORT_MAP.get(dsl_name)

    if all_support_dtype is None:
        return False

    soc_ver = get_soc_spec(SOC_VERSION)
    if soc_ver in (HI3796CV300CS, HI3796CV300ES, SD3403):
        soc_ver = ASCEND_SHISI
    soc_support_dtype = all_support_dtype.get(soc_ver)
    if soc_support_dtype is None:
        soc_support_dtype = all_support_dtype.get("AllSoc")
        if soc_support_dtype is None:
            return False

    if (dtype not in (None, "")) and (dtype not in soc_support_dtype):
        return False

    return True


# pylint: disable=too-many-branches
@decorator
def dtype_check_decorator(func, *args, **kwargs):
    """
    dtype_check_decorator
    """
    func_name = func.__name__
    if func_name == "broadcast":
        if isinstance(args[0], int):
            judge_dtype = "int32"
        elif isinstance(args[0], float):
            judge_dtype = "float16"
        else:
            judge_dtype = args[0].dtype
    elif func_name == "concat":
        if not isinstance(args[0], list):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The first input type must be list," \
                                          " while type is [%s]" % type(args[0])
            raise RuntimeError(dict_args, get_error_message(dict_args))
        if not isinstance(args[0][0], tvm.tensor.Tensor):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The first input type must be list" \
                                          " of tvm.tensor, while type is [%s]" % type(args[0][0])
            raise RuntimeError(dict_args, get_error_message(dict_args))
        judge_dtype = args[0][0].dtype
    else:
        if not isinstance(args[0], tvm.tensor.Tensor):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The first input type must be " \
                                          "tvm.tensor, while type is [%s]" % type(args[0])
            raise RuntimeError(dict_args, get_error_message(dict_args))
        judge_dtype = args[0].dtype

    # skip dtype check condition
    def _is_skip_dtype_check(func_name, args):
        if in_dynamic_and_static_unify():
            # dynamic vsel skip two scalar inputs
            return func_name == "vsel" and \
                   not isinstance(args[1], tvm.tensor.Tensor) and \
                   not isinstance(args[2], tvm.tensor.Tensor)

        # not dynamic skip vcmp vsel vcmpsel
        return func_name in ("vcmp", "vsel", "vcmpsel")

    if _is_skip_dtype_check(func_name, args):
        return func(*args, **kwargs)

    # get vsel lhs or rhs dtype if tensor
    def _get_vsel_dtype(lhs, rhs):
        tmp_type = None
        if isinstance(lhs, tvm.tensor.Tensor):
            tmp_type = lhs.dtype
        elif isinstance(rhs, tvm.tensor.Tensor):
            tmp_type = rhs.dtype
        return tmp_type

    # handle vsel
    if func_name == "vsel":
        judge_dtype = _get_vsel_dtype(args[1], args[2])

    if not dsl_check_support("tbe.dsl."+func_name, judge_dtype):
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "tbe.dsl.%s is not supported %s!" \
                                      % (func_name, judge_dtype)
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return func(*args, **kwargs)


def str_to_tuple(str_local):
    """
    str_to_tuple
    """
    if str_local:
        return str_local.split(",")
    return []


def is_cast_support(src_type, dst_type):
    """
    is_cast_support
    """
    if src_type == dst_type:
        return True

    cast_type = get_cast_type(src_type, dst_type)

    if intrinsic_check_support("Intrinsic_vconv", cast_type):
        return True
    else:
        # Default round mode set as 'z'
        return intrinsic_check_support("Intrinsic_vconv", cast_type + "z")


def get_cast_type(src_type, dst_type):
    """
    get cast type string for vconv_xxxxx
    """
    if src_type not in DTYPE_MAP:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The dtype must be f16, f32, u8, s8 or " \
                                      "s32, [%s] is unsupported dtype!" % src_type
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if dst_type not in DTYPE_MAP:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The dtype must be f16, f32, u8, s8 or " \
                                      "s32, [%s] is unsupported dtype!" % dst_type
        raise RuntimeError(dict_args, get_error_message(dict_args))

    cast_type = DTYPE_MAP[src_type] + "2" + DTYPE_MAP[dst_type]

    if cast_type == "s322f16":
        cast_type = "deq"
    return cast_type


def judge_var(num):
    """
    judge var if a tvm.var, tvm.const or python data type
    """
    var_dict = {
        "python_const": [int, float],
        "tvm_const": [tvm.expr.IntImm, tvm.expr.UIntImm, tvm.expr.FloatImm],
        "tvm_var": [tvm.expr.Var]
    }
    num_type = type(num)
    for i in var_dict:
        if num_type in var_dict[i]:
            return i
    dict_args = dict()
    dict_args["errCode"] = "E90001"
    dict_args["detailed_cause"] = "The input var type must be int, float, " \
                                  "tvm.expr.IntImm, tvm.expr.UIntImm, " \
                                  "tvm.expr.FloatImm or tvm.expr.Var, " \
                                  "while type is [%s]" % num_type
    raise RuntimeError(dict_args, get_error_message(dict_args))


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    return shape_util.shape_to_list(shape)


def int_ceil_div(num_a, num_b):
    """
    upper division
    """
    if num_b == 0:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "division by zero"
        raise RuntimeError(dict_args, get_error_message(dict_args))
    return (num_a + num_b - 1) // num_b


def align(x_1, x_2):
    """
    do align

    """
    if x_2 == 0:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "division by zero"
        raise RuntimeError(dict_args, get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2 * x_2


def get_and_res(flag_a, flag_b):
    """
    logical AND
    """
    return flag_a and flag_b


def get_or_res(flag_a, flag_b):
    """
    logical OR
    """
    return flag_a or flag_b


def refine_axis(axis, shape):
    """
    refine_axis
    """
    if isinstance(axis, (tuple, list)):
        local_axis = axis
    else:
        local_axis = [axis]
    res_axis = []
    shape_len = len(shape)
    for i in local_axis:
        if i < 0:
            laxis = shape_len + i
        else:
            laxis = i
        if (laxis >= shape_len) or (laxis < 0):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "laxis [%s] must less than " \
                                          "shape_len [%s] and bigger than zero!" \
                                          % (laxis, shape_len)
            raise RuntimeError(dict_args, get_error_message(dict_args))
        res_axis.append(laxis)
    return sorted(res_axis)


def _check(bool_res, append_str):
    if not bool_res:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = append_str
        raise RuntimeError(dict_args, get_error_message(dict_args))


def auto_cast_tensor(tensor, intr, supported_types=None, is_auto_cast=True):
    """
    auto_cast_tensor
    """
    from .cast import _cast
    if isinstance(tensor, tvm.tensor.Tensor):
        dtype = tensor.dtype
        intr_is_support_dtype = False
        intr_is_support_fp32 = False
        if supported_types is None:
            intrinsic = "Intrinsic_" + intr
            intr_is_support_dtype = intrinsic_check_support(intrinsic, dtype)
            intr_is_support_fp32 = intrinsic_check_support(intrinsic,
                                                           "float32")
        else:
            intr_is_support_dtype = (dtype in supported_types)
            intr_is_support_fp32 = ("float32" in supported_types)

        if not intr_is_support_dtype:
            if intr_is_support_fp32 and is_cast_support(dtype, "float32"):
                tensor = _cast(tensor, "float32", is_auto_cast)
            else:
                tensor = _cast(tensor, "float16", is_auto_cast)

    return tensor


def get_tvm_scalar(scalar, dtype):
    """
    get_tvm_scalar
    """
    scalar_type = judge_var(scalar)
    if scalar_type == "tvm_const" and scalar.dtype != dtype:
        scalar = tvm.const(scalar.value, dtype=dtype)

    if scalar_type == "python_const":
        scalar = tvm.const(scalar, dtype=dtype)

    return scalar


def check_input_tensor_shape(tensor_shape):
    """
    check_tensor_shape
    """
    shape = tensor_shape
    if isinstance(tensor_shape, tvm.tensor.Tensor):
        shape = shape_util.shape_to_list(tensor_shape.shape)

    in_dynamic = operation.in_dynamic()
    for val in shape:
        if in_dynamic:
            if isinstance(val, int) and val < 0:
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The dynamic input shape value " \
                                              "can not be negative when is a " \
                                              "integer while val is [%s]" % val
                raise RuntimeError(dict_args, get_error_message(dict_args))
        else:
            if isinstance(val, int) is False or val <= 0:
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The static input shape value " \
                                              "must be a positive integer while val is [%s]" % val
                raise RuntimeError(dict_args, get_error_message(dict_args))


def _axis_value_type_check(shape_len, value):
    """
    Check the value of the axis
    """
    if not isinstance(value, int):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "type of axis value should be int, " \
                                      "while axis's type is [%s]" % type(value)
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if value >= shape_len or value < -shape_len:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "input axis [%s] is out of range, " \
                                      "axis value can be from [%s] to [%s]" \
                                      % (value, -shape_len, shape_len - 1)
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if value < 0:
        value = shape_len + value
    return value


def reduce_axis_check(shape_len, axis):
    """
    Check the value of axis and return the sorted axis
    """
    axis = list(axis)
    if not hasattr(axis, 'index'):
        axis = _axis_value_type_check(shape_len, axis)
        return axis
    # pylint: disable=consider-using-enumerate
    for i in range(len(axis)):
        axis[i] = _axis_value_type_check(shape_len, axis[i])

    axis = list(set(axis))
    axis.sort()
    return axis


def util_astype(scalar, dtype):
    """
    :param scalar:
    :param dtype:
    :return:
    """
    if isinstance(scalar, int):
        return tvm.const(scalar, "int").astype(dtype)
    if isinstance(scalar, float):
        return tvm.const(scalar, "float").astype(dtype)
    if isinstance(scalar, (tvm.expr.IntImm, tvm.expr.UIntImm,
                           tvm.expr.FloatImm)):
        return scalar.astype(dtype)
    if isinstance(scalar, tvm.expr.Var):
        return scalar.astype(dtype)
    if isinstance(scalar, tvm.tensor.TensorSlice):
        return scalar
    dict_args = dict()
    dict_args["errCode"] = "E90001"
    dict_args["detailed_cause"] = "Scalar must be simple type, but now is [%s]" % type(scalar)
    raise RuntimeError(dict_args, get_error_message(dict_args))


def _get_priority_flag_value(priority_flag):
    if isinstance(priority_flag, (int, float)):
        return priority_flag
    return priority_flag.value


def in_dynamic_and_static_unify():
    """
    determine whether to perform the unification of dynamic and static shape
    """
    context = operation.get_context()
    return context is not None and context.get_mode() in ("dynamic", "static")
