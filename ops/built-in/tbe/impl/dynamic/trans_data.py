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
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.dynamic.trans_data_rnn import trans_data_rnn
from impl.dynamic.transpose import Transpose
from . import trans_data_negative_target_tc
from . import trans_data_negative_target_ch
from . import trans_data_positive_source_nct
from . import trans_data_positive_source_tc

TILING_MAX_SIZE_GM = 2048 # 16KB
MAX_INT64_VALUE = 2 ** 64 - 1


def is_do_with_transpose_formats(src_format, dst_format):
    """
    judge src_format and dst_format in the list: ["NCHW", "NHWC", "HWCN", "CHWN"]
    """
    format_list = ["NCHW", "NHWC", "HWCN", "CHWN"]
    if src_format in format_list and dst_format in format_list and src_format != dst_format:
        return True
    else:
        return False


# pylint: disable=unused-argument
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
    elif (src_format == "NC1HWC0" and dst_format == "NCHW"):
        trans_data_negative_target_ch.trans_data_negative_target_ch(src, dst, src_format, dst_format, kernel_name)
    elif (src_format == "NCHW" and dst_format == "NC1HWC0") or (src_format == "NCDHW" and dst_format == "NDC1HWC0") \
          or ((src_format == "HWCN") and (dst_format == "FRACTAL_Z" or dst_format == "FRACTAL_ZN")) \
          or (src_format == "DHWCN" and dst_format == "FRACTAL_Z_3D"):
        trans_data_positive_source_nct.trans_data_positive_source_nct(src, dst, src_format, dst_format, kernel_name)
    elif is_do_with_transpose_formats(src_format, dst_format):
        x_dtype = src.get("dtype").lower()
        y_dtype = dst.get("dtype").lower()
        tik_inst = tik.Tik()
        data_in = tik_inst.Tensor(x_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_in")
        data_out = tik_inst.Tensor(y_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_out")
        data_workspace = tik_inst.Tensor(y_dtype, (1024, ), tik.scope_gm, "data_workspace", is_workspace=True)
        data_tiling = tik_inst.Tensor("int64", (TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
        tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
        input_list = [data_in]
        transpose_instance = Transpose(tik_inst, x_dtype, tensor_list, kernel_name)
        return transpose_instance.compute(input_list)
    else:
        trans_data_positive_source_tc.trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name)
