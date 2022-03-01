#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SoftmaxCrossEntropyWithLogits", "impl.dynamic.softmax_cross_entropy_with_logits", "softmax_cross_entropy_with_logits")

case1 = {"params": [{"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1), "ori_format": "ND",
                     "range": ((1, None), (1, None))},
                    {"shape": (-1, 3), "dtype": "float32", "format": "ND", "ori_shape": (-1, 3), "ori_format": "ND",
                     "range": ((1, None), (3, 3))},
                    {"shape": (-1, ), "dtype": "float32", "format": "ND", "ori_shape": (-1, ), "ori_format": "ND",
                     "range": ((1, None),)},
                    {"shape": (-1, 3), "dtype": "float32", "format": "ND", "ori_shape": (-1, 3), "ori_format": "ND",
                     "range": ((1, None), (3, 3))}],
         "case_name": "softmax_cross_entropy_with_logits_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
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
         "format_expect": [],
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
         "format_expect": [],
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
         "format_expect": [],
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
         "format_expect": [],
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
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A", "Ascend710"], case6)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
