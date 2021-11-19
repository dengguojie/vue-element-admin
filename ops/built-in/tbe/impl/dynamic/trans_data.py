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
from impl.dynamic.transpose import Transpose
from . import trans_data_positive_source_tc
from . import trans_data_negative_target_ntc
from . import trans_data_positive_source_ntc
from . import trans_data_negative_target_tc


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    TILING_MAX_SIZE_GM = 2048  # 16KB
    MAX_INT64_VALUE = 2 ** 64 - 1


def is_do_with_transpose_formats(src_format, dst_format):
    """
    judge src_format and dst_format in the list: ["NCHW", "NHWC", "HWCN", "CHWN"]
    """
    format_list = ["NCHW", "NHWC", "HWCN", "CHWN"]
    if src_format in format_list and dst_format in format_list and src_format != dst_format:
        return True
    return False


def is_do_with_positive_source_ntc_100(src_format, dst_format):
    """
    judge src_format and dst_format in the dict:
    {"NCDHW":"NDC1HWC0", "NCHW":"NC1HWC0", "HWCN":"FRACTAL_Z", "HWCN":"FRACTAL_ZN", "DHWCN":"FRACTAL_Z_3D",
    "ND":"FRACTAL_Z", "ND":"FRACTAL_ZN", "NCHW":"FRACTAL_Z", "NCHW":"FRACTAL_ZN", "NCDHW":"FRACTAL_Z_3D"}
    """
    support_src_dst_formats = {"NCDHW": ["NDC1HWC0", "FRACTAL_Z_3D"], "HWCN": ["FRACTAL_Z", "FRACTAL_ZN"],
                               "DHWCN": ["FRACTAL_Z_3D"], "ND": ["FRACTAL_Z", "FRACTAL_ZN"],
                               "NCHW": ["FRACTAL_Z", "FRACTAL_ZN", "NC1HWC0"]}
    if src_format in support_src_dst_formats and dst_format in support_src_dst_formats.get(src_format):
        return True
    return False


# 'pylint: disable=unused-argument, too-many-arguments, too-many-locals, too-many-boolean-expressions
# 'pylint: disable=inconsistent-return-statements
@register_operator("TransData")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def trans_data(src, dst, src_format=None, dst_format=None, group=1, kernel_name="trans_data"):
    """
    format transform for rnn
    """
    if src_format is None:
        src_format = src.get("format").upper().split(":")[0]
    else:
        src_format = src_format.upper()

    if dst_format is None:
        dst_format = dst.get("format").upper().split(":")[0]
    else:
        dst_format = dst_format.upper()

    if ((src_format == "NC1HWC0" and dst_format == "NHWC") or
        (src_format == "FRACTAL_NZ" and dst_format in ("ND", "NHWC", "NCHW", "NC1HWC0")) or
        (src_format == "FRACTAL_Z_3D" and dst_format == "NDHWC") or
        (src_format == "NDC1HWC0" and dst_format == "NDHWC")):
        trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
    elif (((src_format == "NC1HWC0" and dst_format == "NCHW") or
           (src_format == "FRACTAL_Z_3D" and dst_format == "NCDHW") or
           (src_format == "NDC1HWC0" and dst_format == "NCDHW") or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "HWCN")) or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "NCHW")) or
           ((src_format in ("FRACTAL_Z", "FRACTAL_ZN")) and (dst_format == "ND")) or
           (src_format == "FRACTAL_Z_3D" and dst_format == "DHWCN"))):
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
    elif is_do_with_transpose_formats(src_format, dst_format):
        x_dtype = src.get("dtype").lower()
        y_dtype = dst.get("dtype").lower()
        tik_inst = tik.Tik()
        data_in = tik_inst.Tensor(x_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_in")
        data_out = tik_inst.Tensor(y_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_out")
        data_workspace = tik_inst.Tensor(y_dtype, (1024, ), tik.scope_gm, "data_workspace", is_workspace=True)
        data_tiling = tik_inst.Tensor("int64", (Constant.TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
        tensor_list = [data_in, None, data_out, data_workspace, data_tiling]
        input_list = [data_in]
        transpose_instance = Transpose(tik_inst, x_dtype, tensor_list, kernel_name)
        return transpose_instance.compute(input_list)
    elif is_do_with_positive_source_ntc_100(src_format, dst_format):
        trans_data_positive_source_ntc.trans_data_positive_source_ntc(src, dst, src_format, dst_format, kernel_name)
    else:
        trans_data_positive_source_tc.trans_data_positive_source_tc(src, dst, src_format, dst_format, kernel_name)
