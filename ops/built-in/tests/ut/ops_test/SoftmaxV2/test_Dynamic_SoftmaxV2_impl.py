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


ut_case.add_case(["Ascend910A"],
                 gen_softmaxv2_case((-1, -1, -1),
                                    (16, 16, 16),
                                    "float16", -1, "dynamic_softmax_v2_1", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_softmaxv2_case((-1, -1, -1),
                                    (16, 16, 16),
                                    "float32", -1, "dynamic_softmax_v2_3", "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
