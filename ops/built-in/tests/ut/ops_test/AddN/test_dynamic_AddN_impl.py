#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("AddN", "impl.dynamic.add_n", "add_n")


def gen_dynamic_add_n_case(shape_x, range_x, dtype_val, tensor_num,
                           kernel_name_val, expect):
    return {"params": [[{"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
                         "format": "ND", "dtype": dtype_val, "range": range_x},
                        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
                         "format": "ND", "dtype": dtype_val, "range": range_x}],
                       {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
                        "format": "ND", "dtype": dtype_val, "range": range_x},
                       tensor_num],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True}


ut_case.add_case("all",
                 gen_dynamic_add_n_case((-1,), ((1, None),), "float16", 2,
                                        "dynamic_add_n_fp16_ND", "success"))

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
