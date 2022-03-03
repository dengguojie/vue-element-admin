# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
"""
ut for resize
"""
import json
from op_test_frame.ut import OpUT


ut_case = OpUT("L2Loss", "impl.dynamic.l2_loss", "l2_loss")


# pylint: disable=unused-argument
def get_special_shape(ori_shape, ori_format, dst_format, align_num=16):
    """
    get_special_shape
    """
    def _ceil_div(dim):
        return (dim + align_num - 1) // align_num

    dst_shape = []
    if dst_format in ("FRACTAL_NZ",):
        dst_shape = ori_shape[:-2] + [_ceil_div(ori_shape[-1]), _ceil_div(ori_shape[-2]), align_num, align_num]
    dst_shape_len = len(dst_shape)
    return dst_shape if dst_shape_len != 0 else ori_shape


def tensor_dict(tensor_ori_shape, tensor_ori_format, tensor_type, tensor_format=None):
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

    return gen_dict


case1 = {"params": [tensor_dict([-1, -1, -1], "ND", "float16"),
                    tensor_dict([], "ND", "float16")],
         "case_name": "dynamic_l2_loss_dynamic_fp16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


case2 = {"params": [tensor_dict([-1, -1, -1], "ND", "float32"),
                    tensor_dict([], "ND", "float32")],
         "case_name": "dynamic_l2_loss_dynamic_fp32",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


case3 = {"params": [tensor_dict([-2], "ND", "float32"),
                    tensor_dict([], "ND", "float32")],
         "case_name": "dynamic_l2_loss_unknown_rank_fp32",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


def test_import_lib(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.util.reduce_pattern_adapter"))


ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_cust_test_func(test_func=test_import_lib)


if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
