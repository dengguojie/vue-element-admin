#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf

ut_case = BroadcastOpUT("Xlogy", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910A","Ascend310","Ascend710"], ["float16", "float32"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910A","Ascend310","Ascend710"], ["float16", "float32"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910A","Ascend310","Ascend710"], ["float16", "float32"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================
def calc_expect_func(inputA, inputB, output):
    with tf.Session() as sess:
        outputArr = sess.run(tf.math.xlogy(inputA['value'], inputB['value']))
    return outputArr

precision_case1 = {"params": [{"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input"},
                              {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input"},
                              {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910A",precision_case1)
ut_case.add_precision_case("Ascend910A",precision_case2)
ut_case.add_precision_case("Ascend910A",precision_case3)
if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend710"])
    exit(0)
