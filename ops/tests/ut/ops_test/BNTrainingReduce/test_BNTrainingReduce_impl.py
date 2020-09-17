#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BnTrainingReduce", None, None)

def gen_BNTrainingReduce_case(shape_x, shape_sum, shape_square, dtype, case_name_val):
    return {"params": [{"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": "NC1HWC0", "format": "NC1HWC0"},
                       {"shape": shape_sum, "dtype": dtype, "ori_shape": shape_sum, "ori_format": "NC1HWC0", "format": "NC1HWC0"},
                       {"shape": shape_square, "dtype": dtype, "ori_shape": shape_square, "ori_format": "NC1HWC0", "format": "NC1HWC0"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_BNTrainingReduce_case((2,2,2,2,16), (1,2,1,1,16), (1,2,1,1,16), "float16", "bn_training_reduce_1")
case2 = gen_BNTrainingReduce_case((2,4,384,576,16), (1,4,1,1,16), (1,4,1,1,16), "float32", "bn_training_reduce_2")
case3 = gen_BNTrainingReduce_case((2, 4, 96, 144, 16), (1,4,1,1,16), (1,4,1,1,16), "float16", "bn_training_reduce_3")
case4 = gen_BNTrainingReduce_case((2, 4, 96, 144, 16), (1,4,1,1,16), (1,4,1,1,16), "float32", "bn_training_reduce_4")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")