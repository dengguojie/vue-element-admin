#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("SigmoidCrossEntropyWithLogits", "impl.dynamic.sigmoid_cross_entropy_with_logits", "sigmoid_cross_entropy_with_logits")

case1 = {"params": [{"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "sigmoid_cross_entropy_with_logits_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


def gen_dynamic_case(shape_x, shape_y, range_x, dtype_val,
                     kernel_name_val, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": shape_y, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x}
        ],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True}


ut_case.add_case("all", gen_dynamic_case((-1, -1), (-1, -1), ((1, None), (1, None)),
                                         "float16",
                                         "dynamic_sigmoid_cross_entropy_with_logits_1",
                                         "success"))

ut_case.add_case("all", gen_dynamic_case((5, -1), (-1, -1), ((5, 50), (1, None)),
                                         "float32",
                                         "dynamic_sigmoid_cross_entropy_with_logits_2",
                                         "success"))

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
