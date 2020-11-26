#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import tensorflow as tf
from tensorflow.python.ops import gen_array_ops

ut_case = OpUT("InplaceAddD", None, None)

case1 = {"params": [{"shape": (4,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (4,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (4,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (4,4,32,2),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (4,4,32), "dtype": "float32", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,4,32,3), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,3),"ori_format": "ND"},
                    {"shape": (2,4,32,2), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,2),"ori_format": "ND"},
                    {"shape": (2,4,32,3), "dtype": "float32", "format": "ND", "ori_shape": (2,4,32,3),"ori_format": "ND"},
                    [0, 1]],
         "case_name": "inplace_add_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

def calc_expect_func(x, y, output, indices):
    out = gen_array_ops.inplace_add(x['value'],indices,y['value'])
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
                              [0, -6, 2, 5]],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}

ut_case.add_precision_case("Ascend910",precision_case1)
ut_case.add_precision_case("Ascend910",precision_case2)


if __name__ == '__main__':
    ut_case.run("Ascend910")
