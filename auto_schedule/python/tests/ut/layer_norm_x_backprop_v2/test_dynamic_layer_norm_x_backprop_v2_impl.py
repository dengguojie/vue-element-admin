# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe

ut_case = OpUT("LayerNormXBackpropV2", "impl.dynamic.layer_norm_x_backprop_v2", "layer_norm_x_backprop_v2")


case1 = {"params": [{"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (13, 32, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case4 = {"params": [{"shape": (-1, -1, 30), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 30), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND","range": ((0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (30,), "dtype": "float16", "format": "ND", "ori_shape": (30,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (-1, -1, 30), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (-1, -1, 30), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 30),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
         "expect": "success",
         "support_expect": True}
case5 = {"params": [{"shape": (64, -1, -1, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 114, 16, 16),"ori_format": "FRACTAL_NZ", "range": ((0, 1), (0, 1), (0, 1), (0, 1))},
                    {"shape": (64, -1, -1, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 114, 16, 16),"ori_format": "FRACTAL_NZ","range": ((0, 1), (0, 1), (0, 1), (0, 1))},
                    {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1824, 1),"ori_format": "ND", "range": ((0, 5), (0, 5))},
                    {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1824, 1),"ori_format": "ND", "range": ((0, 5), (0, 5))},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND", "range": ((0, 1),)},
                    {"shape": (64, -1, 16), "dtype": "float16", "format": "ND", "ori_shape": (64, 1824, 16),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},
                    {"shape": (64, -1, 16), "dtype": "float32", "format": "ND", "ori_shape": (64, 1824, 16),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))}],
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
    case1,
    case2,
    case3,
    case4,
    case5
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A"], case=item)
if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
