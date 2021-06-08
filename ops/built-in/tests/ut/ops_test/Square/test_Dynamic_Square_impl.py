#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Square", "impl.dynamic.square", "square")

def gen_dynamic_square_case(shape_x, range_x, dtype_val, Format,
                            ori_shape_x, kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": Format,
                        "ori_shape": ori_shape_x, "ori_format": Format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": Format,
                        "ori_shape": ori_shape_x, "ori_format": Format}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_dynamic_square_case((-1,),
                                         [(1, None)],
                                         "float32", "ND",
                                         (-1,),
                                         "dynamic_square_float32_ND",
                                         "success"))

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
