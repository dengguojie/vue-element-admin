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

# 32B to the number of fp16
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
    if "matmul" in tensor.op.tag:
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
