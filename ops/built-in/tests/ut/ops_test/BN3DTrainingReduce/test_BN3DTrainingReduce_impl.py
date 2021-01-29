#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
ut_case = OpUT("Bn3dTrainingReduce", None, None)

def gen_BNTrainingReduce_case(shape_x, shape_sum, shape_square, dtype, format, case_name_val):
    return {"params": [{"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": format, "format": format},
                       {"shape": shape_sum, "dtype": "float32", "ori_shape": shape_sum, "ori_format": format, "format": format},
                       {"shape": shape_square, "dtype": "float32", "ori_shape": shape_square, "ori_format": format, "format": format}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1= gen_BNTrainingReduce_case((2,2,2,4,4,16),(1,1,2,1,1,16),(1,1,2,1,1,16), "float16", "NDC1HWC0", "bn_training_reduce_1")
case2= gen_BNTrainingReduce_case((2,2,2,4,4,16),(1,1,2,1,1,16),(1,1,2,1,1,16), "float32", "NDC1HWC0", "bn_training_reduce_2")

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
