# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe

ut_case = OpUT("LayerNormBetaGammaBackpropV2", "impl.dynamic.layer_norm_beta_gamma_backprop_v2", "layer_norm_beta_gamma_backprop_v2")

case1 = {"params": [{"shape": (-1,-1,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1,-1,512), "dtype": "float16", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2,3,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (2,3,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2,3,-1), "dtype": "float16", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (2,3,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (-1,-1,-1), "dtype": "float16", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,-1,-1), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (-1,3,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,3,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [2,3,512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (-1,3,512), "dtype": "float16", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (-1,3,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [2,3,512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_8",
         "expect": "success",
         "support_expect": True}


def test_generalization(args):
    from impl.dynamic.layer_norm_beta_gamma_backprop_v2 import layer_norm_beta_gamma_backprop_v2_generalization
    layer_norm_beta_gamma_backprop_v2_generalization(
        {"shape": (2,3,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
        {"shape": (2,3,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
        {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
        {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
        [512], None, None)
        
compile_case_list = [
    case1,
    case2,
    case3,
    case4,
    case5,
    case6,
    case7,
    case8,
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A"], case=item)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")