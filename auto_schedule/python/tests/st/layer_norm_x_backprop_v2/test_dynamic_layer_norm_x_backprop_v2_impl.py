# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe

ut_case = OpUT("LayerNormXBackpropV2", "impl.dynamic.layer_norm_x_backprop_v2", "layer_norm_x_backprop_v2")


case3 = {"params": [{"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case4 = {"params": [{"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}

def test_generalization(args):
    from impl.dynamic.layer_norm_x_backprop_v2 import layer_norm_x_backprop_generalization
    layer_norm_x_backprop_generalization(
        {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
        {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
        {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
        {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
        {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
        {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
        {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
        None, None)
        

compile_case_list = [
    case3,
    case4
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A"], case=item)
if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
