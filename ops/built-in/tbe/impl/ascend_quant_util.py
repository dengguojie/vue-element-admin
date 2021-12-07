# Copyright 2019 Huawei Technologies Co., Ltd
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
ascend_quant_util
"""
from te import tvm
from impl.util import util_select_op_base


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    FP16_BLOCK_VALUE = 16


def is_matmul_fuse(tensor):
    """
    check is matmul fuse
    """
    if not isinstance(tensor, tvm.tensor.Tensor):
        return False
    is_matmul_fuse_flag = False
    stack = [tensor]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                if hasattr(in_tensor.op, "tag") and "matmul" in in_tensor.op.tag:
                    is_matmul_fuse_flag = True
                    break
    return is_matmul_fuse_flag


def is_nz_format(tensor, is_quant=False):
    """
    check is nz format
    """
    if "matmul" in tensor.op.tag or "gemm" in tensor.op.tag:
        return True

    tensor_format = "NC1HWC0"
    if tensor.op.attrs:
        if "format" in tensor.op.attrs:
            tensor_format = tensor.op.attrs["format"]

    if tensor_format == "FRACTAL_NZ":
        return True

    if is_quant:
        if is_matmul_fuse(tensor):
            return True

    return False


# 'pylint: disable = unused-argument
def get_quant_support_info(x, x1=None, dual_output=False, l1_fusion_enable=0):
    """
    obtains the split information of the quantization operator
    """
    dim_x = len(x.get("shape"))
    format_x = x.get("format")

    # C1 C0  can not split
    not_cut_dim = [1, 4]
    if format_x == "FRACTAL_NZ":
        not_cut_dim = [dim_x - 4, dim_x - 1]

    if format_x in ["NC1HWC0", "FRACTAL_NZ"]:
        axis_split_list = []
        for i in range(dim_x):
            if i not in not_cut_dim:
                if x1 is not None:
                    split_in = util_select_op_base.SplitInput([0, [i], [-1], [-1]],
                                                              [2, [i], [-1], [-1]])
                else:
                    split_in = util_select_op_base.SplitInput([0, [i], [-1], [-1]])
                if dual_output:
                    split_out = util_select_op_base.SplitOutput([0, [i]],
                                                                [1, [i]])
                else:
                    split_out = util_select_op_base.SplitOutput([0, [i]])
                axis_split_list.append([split_in, split_out])
    else:
        axis_split_list = None

    axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_list, axis_reduce_list, l1_fusion_enable)
    return op_cal_info_in_json
