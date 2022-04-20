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


def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.dynamic.l2_loss import op_select_format
    op_select_format(
        {"shape": (1024, 1024, 256, 256), "dtype": "float", "format": "NCHW", "ori_shape": (1024, 1024, 256, 256),
         "ori_format": "NCHW"},
        {"shape": (1024, 1024, 256, 256), "dtype": "float", "format": "NCHW", "ori_shape": (1024, 1024, 256, 256),
         "ori_format": "NCHW"},
        kernel_name="test_l2_loss_op_select_format_1")
    op_select_format(
        {"shape": (14, 10, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (160, 224),
         "ori_format": "NCHW"},
        {"shape": (1), "dtype": "float32", "format": "ND", "ori_shape": (1), "ori_format": "NCHW"},
        kernel_name="test_l2_loss_op_select_format_2")
    op_select_format(
        {"shape": (16, 1, 16, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 16, 16),
         "ori_format": "NCHW"},
        {"shape": (1), "dtype": "float16", "format": "ND", "ori_shape": (1), "ori_format": "ND"},
        kernel_name="test_l2_loss_op_select_format_3")
    op_select_format(
        {"shape": (1, 5, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 5, 16, 16),
         "ori_format": "NCHW"},
        {"shape": (1), "dtype": "float32", "format": "ND", "ori_shape": (1), "ori_format": "NCHW"},
        kernel_name="test_l2_loss_op_select_format_4")


ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_cust_test_func(test_func=test_import_lib)
ut_case.add_cust_test_func(test_func=test_op_select_format)


if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
