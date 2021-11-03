# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe


ut_case = OpUT("LayerNorm", "impl.dynamic.layer_norm", "layer_norm")

case1 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1), "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    2, 2],
         "case_name": "layer_norm_1",
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1), "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1,), "dtype": "float16", "format": "NCHW", "ori_shape": (-1,),
                     "ori_format": "NCHW", "range": ((1, None), )},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    {"shape": (-1, -1, -1), "dtype": "float16", "format": "NCHW", "ori_shape": (-1, -1, -1),
                     "ori_format": "NCHW", "range": ((1, None), (1, None), (1, None))},
                    0, 2],
         "case_name": "layer_norm_2",
         "expect": "success",
         "support_expect": True}

compile_case_list = [
    case1,
    case2
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A", "Ascend310"], case=item)
    
if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
