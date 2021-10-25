# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe


ut_case = OpUT("SoftmaxCrossEntropyWithLogits", "impl.dynamic.softmax_cross_entropy_with_logits", "softmax_cross_entropy_with_logits")

case2 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1), "ori_format": "ND",
                     "range": ((1, None), (1, None))},
                    {"shape": (-1, 3), "dtype": "float16", "format": "ND", "ori_shape": (-1, 3), "ori_format": "ND",
                     "range": ((1, None), (3, 3))},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,), "ori_format": "ND",
                     "range": ((1, None),)},
                    {"shape": (-1, 3), "dtype": "float16", "format": "ND", "ori_shape": (-1, 3), "ori_format": "ND",
                     "range": ((1, None), (3, 3))}],
         "case_name": "softmax_cross_entropy_with_logits_2",
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (-1, 2, 3), "dtype": "float32", "format": "ND", "ori_shape": (-1, 2, 3),"ori_format": "ND",
                     "range": ((1, None), (2, 2), (3, 3))},
                    {"shape": (-1, 2), "dtype": "float32", "format": "ND", "ori_shape": (-1, 2),"ori_format": "ND",
                     "range": ((1, None), (2, 2))},
                    {"shape": (-1, ), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND",
                     "range": ((1, None),)},
                    {"shape": (-1, 2), "dtype": "float32", "format": "ND", "ori_shape": (-1, 2),"ori_format": "ND",
                     "range": ((1, None), (2, 2))}],
         "case_name": "softmax_cross_entropy_with_logits_3",
         "expect": ValueError,
         "support_expect": True}
case4 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND",
                     "range": ((1, None), (1, None))},
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND",
                     "range": ((1, None), (1, None))},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),"ori_format": "ND",
                     "range": ((1, None),)},
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND",
                     "range": ((1, None), (1, None))}],
         "case_name": "softmax_cross_entropy_with_logits_4",
         "expect": "success",
         "support_expect": True}
case5 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND",
                     "range": ((1, None), (1, None))},
                    {"shape": (2, -1), "dtype": "float16", "format": "ND", "ori_shape": (2, -1),"ori_format": "ND",
                     "range": ((2, 2), (1, None))},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND",
                     "range": ((2, 2),)},
                    {"shape": (2, -1), "dtype": "float16", "format": "ND", "ori_shape": (2, -1),"ori_format": "ND",
                     "range": ((2, 2), (1, None))}],
         "case_name": "softmax_cross_entropy_with_logits_5",
         "expect": "success",
         "support_expect": True}
case6 = {"params": [{"shape": (2, 24), "dtype": "float16", "format": "ND", "ori_shape": (2, 24),"ori_format": "ND",
                     "range": ((2, 2), (24, 24))},
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, -1),"ori_format": "ND",
                     "range": ((2, None), (22, None))},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND",
                     "range": ((2, 2),)},
                    {"shape": (2, 24), "dtype": "float16", "format": "ND", "ori_shape": (2, 24),"ori_format": "ND",
                     "range": ((2, 2), (24, 24))}],
         "case_name": "softmax_cross_entropy_with_logits_6",
         "expect": "success",
         "support_expect": True}
         
compile_case_list = [
    case2,
    case3,
    case4,
    case5,
    case6,
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A"], case=item)
    
if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")