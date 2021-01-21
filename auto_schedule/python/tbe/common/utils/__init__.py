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
tbe utils
"""
from . import const

from . import log

from .para_check import NONE_TYPE
from .para_check import KERNEL_NAME
from .para_check import REQUIRED_INPUT
from .para_check import OPTION_INPUT
from .para_check import DYNAMIC_INPUT
from .para_check import REQUIRED_OUTPUT
from .para_check import OPTION_OUTPUT
from .para_check import DYNAMIC_OUTPUT
from .para_check import REQUIRED_ATTR_INT
from .para_check import REQUIRED_ATTR_FLOAT
from .para_check import REQUIRED_ATTR_STR
from .para_check import REQUIRED_ATTR_BOOL
from .para_check import REQUIRED_ATTR_TYPE
from .para_check import REQUIRED_ATTR_LIST_INT
from .para_check import REQUIRED_ATTR_LIST_FLOAT
from .para_check import REQUIRED_ATTR_LIST_BOOL
from .para_check import REQUIRED_ATTR_LIST_LIST_INT
from .para_check import OPTION_ATTR_INT
from .para_check import OPTION_ATTR_FLOAT
from .para_check import OPTION_ATTR_STR
from .para_check import OPTION_ATTR_BOOL
from .para_check import OPTION_ATTR_TYPE
from .para_check import OPTION_ATTR_LIST_INT
from .para_check import OPTION_ATTR_LIST_FLOAT
from .para_check import OPTION_ATTR_LIST_BOOL
from .para_check import OPTION_ATTR_LIST_LIST_INT
from .para_check import OP_ERROR_CODE_000
from .para_check import OP_ERROR_CODE_001
from .para_check import OP_ERROR_CODE_002
from .para_check import OP_ERROR_CODE_003
from .para_check import OP_ERROR_CODE_004
from .para_check import OP_ERROR_CODE_005
from .para_check import OP_ERROR_CODE_006
from .para_check import OP_ERROR_CODE_007
from .para_check import OP_ERROR_CODE_008
from .para_check import OP_ERROR_CODE_009
from .para_check import OP_ERROR_CODE_010
from .para_check import OP_ERROR_CODE_011
from .para_check import OP_ERROR_CODE_012
from .para_check import OP_ERROR_CODE_013
from .para_check import OP_ERROR_CODE_014
from .para_check import OP_ERROR_CODE_015
from .para_check import OP_ERROR_CODE_016
from .para_check import OP_ERROR_CODE_017
from .para_check import OP_ERROR_CODE_018
from .para_check import OP_ERROR_CODE_019
from .para_check import OP_ERROR_CODE_020
from .para_check import OP_ERROR_CODE_021
from .para_check import OP_ERROR_CODE_022
from .para_check import OP_ERROR_CODE_023
from .para_check import OP_ERROR_CODE_024
from .para_check import OP_ERROR_CODE_025
from .para_check import OP_ERROR_CODE_026
from .para_check import OP_ERROR_CODE_027
from .para_check import ALL_DTYPE_LIST
from .para_check import check_op_params
from .para_check import check_shape
from .para_check import check_dtype
from .para_check import check_format
from .para_check import check_elewise_shape_range
from .para_check import check_input_type
from .para_check import check_dtype_rule
from .para_check import check_shape_rule
from .para_check import check_kernel_name
from .para_check import check_and_init_5hdc_reduce_support
from .para_check import is_scalar
from .para_check import check_shape_size

from .shape_util import squeeze_shape
from .shape_util import wrap_axes_to_positive
from .shape_util import refine_shape_axes
from .shape_util import broadcast_shapes
from .shape_util import refine_shapes_for_broadcast
from .shape_util import variable_shape
from .shape_util import simplify_axis_shape
from .shape_util import shape_refine
from .shape_util import refine_axis
from .shape_util import axis_check
from .shape_util import produce_shapes
from .shape_util import check_reduce_need_refine
from .shape_util import scalar2tensor_one
from .shape_util import axis_transform_5d
from .shape_util import compare_tensor_dict_key
from .shape_util import get_shape_size
from .shape_util import cast
from .shape_util import shape_to_list
