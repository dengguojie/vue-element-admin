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
gen_case_util.py
"""


def get_special_shape(ori_shape, ori_format, dst_format, align_num=16):
    """
    get_special_shape
    """
    def _ceil_div(dim):
        return (dim + align_num - 1) // align_num

    dst_shape = []
    if dst_format in ("FRACTAL_NZ",):
        dst_shape = ori_shape[:-2] + [
            _ceil_div(ori_shape[-1]),
            _ceil_div(ori_shape[-2]), align_num, align_num
        ]
    if dst_format in ("NC1HWC0",):
        foramt_dim = dict(zip(list(ori_format), ori_shape))
        dst_shape = [foramt_dim["N"],
                     _ceil_div(foramt_dim["C"]),
                     foramt_dim["H"],
                     foramt_dim["W"],
                     align_num]

    dst_shape_len = len(dst_shape)

    return dst_shape if dst_shape_len != 0 else ori_shape


def tensor_dict(tensor_ori_shape, tensor_ori_format, tensor_type, tensor_format=None, is_output=False):
    """
    return a dict
    """
    if tensor_format is None:
        tensor_format = tensor_ori_format
    tensor_shape = get_special_shape(tensor_ori_shape, tensor_ori_format, tensor_format)

    gen_dict = dict()
    gen_dict["ori_shape"] = tensor_ori_shape
    gen_dict["ori_format"] = tensor_ori_format
    gen_dict["dtype"] = tensor_type
    gen_dict["shape"] = tensor_shape
    gen_dict["format"] = tensor_format
    gen_dict["range"] = [(1, 100000)] * len(tensor_shape)
    param_type = "output" if is_output else "input"
    gen_dict["param_type"] = param_type
    return gen_dict
