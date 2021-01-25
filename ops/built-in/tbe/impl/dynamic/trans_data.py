# Copyright 2020 Huawei Technologies Co., Ltd
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
trans_data
"""
from __future__ import absolute_import
import te.lang.dynamic
from te.utils import para_check
from impl.dynamic.trans_data_rnn import trans_data_rnn
from . import trans_data_negative_target_tc
from impl.util.platform_adapter import register_operator

@register_operator("TransData")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_STR,
                            para_check.REQUIRED_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def trans_data(src, dst, src_format, dst_format, group=1, kernel_name="trans_data"):
    """
    format transform for rnn
    """
    src_format = src_format.upper()
    dst_format = dst_format.upper()

    if (src_format == "NC1HWC0" and dst_format == "NHWC") or (src_format == "FRACTAL_NZ" and dst_format == "ND") \
        or (src_format == "FRACTAL_Z_3D" and dst_format == "NDHWC"):
        trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
    else:
        return trans_data_rnn(src, dst, src_format, dst_format, 0, 0, kernel_name)
