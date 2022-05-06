# Copyright 2021 Huawei Technologies Co., Ltd
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
test_confusion_transpose_d_golden.py
"""
import numpy as np


# pylint: disable=unused-argument,invalid-name
def confusion_transpose_d_by_np(x, ori_format, perm, shape, transpose_first):
    """
    confusion_transpose_d_by_np
    """
    if ori_format == "FRACTAL_NZ":
        input_shape = x.shape
        input_remain_len = len(input_shape) - 4
        nz_nd_perm = [1+input_remain_len, 2+input_remain_len, 0+input_remain_len, 3+input_remain_len]
        add_perm = [i for i in range(input_remain_len)]
        last_nz_nd_perm = add_perm + nz_nd_perm
        reshape_shape = [input_shape[i] for i in range(input_remain_len)]
        last_reshape_shape = reshape_shape + [input_shape[-3]*input_shape[-2],input_shape[-4]*input_shape[-1]]
        x = x.transpose(last_nz_nd_perm).reshape(last_reshape_shape)

    if transpose_first:
        result = x.transpose(perm).reshape(shape)
    else:
        result = x.reshape(shape).transpose(perm)

    if ori_format == "FRACTAL_NZ":
        result_shape = result.shape
        out_remain_len = len(result_shape) - 2
        reshape_shape = [result_shape[i] for i in range(out_remain_len)]
        last_reshape_shape = reshape_shape + [result_shape[-2]//16,16,result_shape[-1]//16,16]
        add_perm = [i for i in range(out_remain_len)]
        last_nd_nz_perm = add_perm + [2+out_remain_len,0+out_remain_len,1+out_remain_len,3+out_remain_len]
        result = result.reshape(last_reshape_shape).transpose(last_nd_nz_perm)

    return result


def calc_expect_func(x, y, perm, shape, transpose_first):
    """
    calc_expect_func
    """
    res = confusion_transpose_d_by_np(x["value"], x["format"], perm, shape, transpose_first)
    return [res]