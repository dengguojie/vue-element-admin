# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe

ut_case = OpUT("LayerNormXBackpropV2", "impl.dynamic.layer_norm_x_backprop_v2", "layer_norm_x_backprop_v2")


case1 = {"params": [{"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512), "run_shape":(30, 496, 512), "ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1)),"param_type":"input"},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"run_shape":(30, 496, 512), "ori_format": "ND","range": ((0, 1), (0, 1), (0, 1)),"param_type":"input"},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"run_shape":(30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5)),"param_type":"input"},
                    {"shape": (-1, -1, 1), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 1),"run_shape":(30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5)),"param_type":"input"},
                    {"shape": (512,), "dtype": "float32", "format": "ND", "ori_shape": (512,),"run_shape":(512,),"ori_format": "ND", "range": ((0, 1),),"param_type":"input"},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"run_shape":(30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5)),"param_type":"output"},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"run_shape":(30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5))},"param_type":"output"],
         "case_name":"layer_norm_x_backprop_v2_1",
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"run_shape":(30, 496, 512), "ori_format": "ND", "range": ((0, 1), (0, 1), (0, 1)),"param_type":"input"},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"run_shape":(30, 496, 512), "ori_format": "ND","range": ((0, 1), (0, 1), (0, 1)),"param_type":"input"},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"run_shape":(30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5)),"param_type":"input"},
                    {"shape": (-1, -1, 1), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 1),"run_shape":(30, 496, 1),"ori_format": "ND", "range": ((0, 5), (0, 5), (0, 5)),"param_type":"input"},
                    {"shape": (512,), "dtype": "float16", "format": "ND", "ori_shape": (512,),"run_shape":(512,),"ori_format": "ND", "range": ((0, 1),),"param_type":"input"},
                    {"shape": (-1, -1, 512), "dtype": "float16", "format": "ND", "ori_shape": (30, 496, 512),"run_shape":(30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5)),"param_type":"output"},
                    {"shape": (-1, -1, 512), "dtype": "float32", "format": "ND", "ori_shape": (30, 496, 512),"run_shape":(30, 496, 512),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5)),"param_type":"output"}],
         "case_name":"layer_norm_x_backprop_v2_2",
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (64, -1, -1, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 114, 16, 16),"run_shape":(64, 114, 16, 16), "ori_format": "FRACTAL_NZ", "range": ((0, 1), (0, 1), (0, 1), (0, 1)),"param_type":"input"},
                    {"shape": (64, -1, -1, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 114, 16, 16),"run_shape":(64, 114, 16, 16)"ori_format": "FRACTAL_NZ","range": ((0, 1), (0, 1), (0, 1), (0, 1)),"param_type":"input"},
                    {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1824, 1),"run_shape":(1824, 1),"ori_format": "ND", "range": ((0, 5), (0, 5)),"param_type":"input"},
                    {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1824, 1),"run_shape":(1824, 1),"ori_format": "ND", "range": ((0, 5), (0, 5)),"param_type":"input"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"run_shape":(1024),"ori_format": "ND", "range": ((0, 1),),"param_type":"input"},
                    {"shape": (64, -1, 16), "dtype": "float16", "format": "ND", "ori_shape": (64, 1824, 16),"run_shape":(64, 1824, 16),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5)),"param_type":"output"},
                    {"shape": (64, -1, 16), "dtype": "float32", "format": "ND", "ori_shape": (64, 1824, 16),"run_shape":(64, 1824, 16),"ori_format": "ND","range": ((0, 5), (0, 5), (0, 5)),"param_type":"output"}],
         "case_name":"layer_norm_x_backprop_v2_3",
         "expect": "success",
         "support_expect": True}

compile_case_list = [
    case1,
    case2,
    case3,
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A"], case=item)
if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
