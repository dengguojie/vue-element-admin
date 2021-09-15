#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SoftmaxV2", "impl.dynamic.softmax_v2", "softmax_v2")

def gen_softmaxv2_case(dynamic_input_shapes, ori_input_shapes, dtype, axis,
                    case_name_val, expect, input_format="ND"):
    inputs = (
        {"shape": dynamic_input_shapes,
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
    )
    outputs = (
        {"shape": [-1],
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * 1},
    )

    return {"params": [inputs[0],
                       outputs[0],
                       axis],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"],
                 gen_softmaxv2_case((-1, -1, -1),
                                    (16, 16, 16),
                                    "float16", -1, "dynamic_softmax_v2_1", "success"))

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"],
                 gen_softmaxv2_case((-1, -1, -1),
                                    (16, 16, 16),
                                    "float32", -1, "dynamic_softmax_v2_3", "success"))


from impl.dynamic.softmax_v2 import op_select_format

def test_op_select_format(test_arg):
    op_select_format({"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                     {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                     -1)

ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
