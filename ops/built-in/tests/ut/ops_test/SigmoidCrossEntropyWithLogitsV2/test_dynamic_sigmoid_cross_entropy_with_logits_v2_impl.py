'''
test code
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SigmoidCrossEntropyWithLogitsV2", "impl.dynamic.sigmoid_cross_entropy_with_logits_v2",
               "sigmoid_cross_entropy_with_logits_v2")

case1 = {
    "params": [
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
        {
            "shape": (-1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (2,),
            "ori_format": "ND",
            "range": [(1, 100)]
        },
    ],
    "case_name": "sigmoid_cross_entropy_with_logits_v2_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        None,
        None,
        {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
    ],
    "case_name": "sigmoid_cross_entropy_with_logits_v2_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {
            "shape": (2, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (2, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        None,
        {
            "shape": (2, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
        {
            "shape": (2, -1, -1),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (2, 4, 4),
            "ori_format": "ND",
            "range": [(1, 100), (1, 100), (1, 100)]
        },
    ],
    "case_name": "sigmoid_cross_entropy_with_logits_v2_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
