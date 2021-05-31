#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("LayerNormBetaGammaBackpropV2", "impl.dynamic.layer_norm_beta_gamma_backprop_v2", "layer_norm_beta_gamma_backprop_v2")

case1 = {"params": [{"shape": (2,3,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (2,3,512), "dtype": "float32", "format": "ND", "ori_shape": (2,3,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    {"shape": (1,1,512), "dtype": "float32", "format": "ND", "ori_shape": (1,1,512),"ori_format": "ND","range":[(1,10),(1,10),(1,1024)]},
                    [512]
                    ],
         "case_name": "layer_norm_beta_gamma_backprop_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(["Ascend910A"])
