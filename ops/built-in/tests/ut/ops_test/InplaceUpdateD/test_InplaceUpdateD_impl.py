#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import tensorflow as tf
from tensorflow.python.ops import gen_array_ops

ut_case = OpUT("InplaceUpdateD", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    [1]],
         "case_name": "InplaceUpdateD_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                    [16]],
         "case_name": "InplaceUpdateD_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (1, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (1, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 4, 16),"ori_format": "ND"},
                    [32]],
         "case_name": "InplaceUpdateD_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (2, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (2, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 4, 16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "InplaceUpdateD_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [1]],
         "case_name": "InplaceUpdateD_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)

def calc_expect_func(x, y, output, indices):
    out = gen_array_ops.inplace_update(x['value'],indices,y['value'])
    sess = tf.Session()
    out = sess.run(out)
    return out

precision_case1 = {"params": [{"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND","param_type":"input"},
                              {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND","param_type":"input"},
                              {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND","param_type":"output"},
                              [0, 1]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case2 = {"params": [{"shape": (4,4,5), "dtype": "float32", "format": "ND", "ori_shape": (4,4,5),"ori_format": "ND","param_type":"input"},
                              {"shape": (4,4,5), "dtype": "float32", "format": "ND", "ori_shape": (4,4,5),"ori_format": "ND","param_type":"input"},
                              {"shape": (4,4,5), "dtype": "float32", "format": "ND", "ori_shape": (4,4,5),"ori_format": "ND","param_type":"output"},
                              [0, -10, 6, 3]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}

def test_check_support(test_arg):
    """
    test_check_support
    """
    from impl.inplace_update_d import check_supported
    res, reason = check_supported({"shape": (30001, 3), "dtype": "float16", "format": "ND", "ori_shape": (30001, 3), "ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3), "ori_format": "ND"},
                    {"shape": (30001, 3), "dtype": "float16", "format": "ND", "ori_shape": (30001, 3), "ori_format": "ND"},
                    [-1],"inplace_update_d_check_support_case_001")
    assert not res

ut_case.add_cust_test_func(test_func=test_check_support)

ut_case.add_precision_case("Ascend910",precision_case1)
ut_case.add_precision_case("Ascend910",precision_case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
