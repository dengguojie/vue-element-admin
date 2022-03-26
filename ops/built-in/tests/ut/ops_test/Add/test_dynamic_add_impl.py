#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Add", "impl.dynamic.add", "add")


def gen_dynamic_add_case(shape_x, shape_y, range_x, range_y,
                         dtype_val, format_x, format_y, ori_shape_x, ori_shape_y, ori_format_x, ori_format_y,
                         kernel_name_val, expect, op_imply_type):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format_x,
                        "ori_shape": ori_shape_x, "ori_format": ori_format_x},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": format_y,
                        "ori_shape": ori_shape_y, "ori_format": ori_format_y},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": format_x,
                        "ori_shape": ori_shape_y, "ori_format": ori_format_x}],
            "case_name": kernel_name_val,
            "expect": expect,
            "op_imply_type": op_imply_type,
            "format_expect": [],
            "support_expect": True}


# dynamic
ut_case.add_case(["Ascend910A"],
                 gen_dynamic_add_case((-1,), (-1,),
                                      [(2, 10)],
                                      [(2, 10)],
                                      "float32", "ND", "ND", (-1,
                                                              ), (-1,), "ND", "ND",
                                      "dynamic_mul_fp32_ND", "success", "dynamic"))
# static
ut_case.add_case(["Ascend910A"],
                 gen_dynamic_add_case((2, 2, 16, 16), (32,),
                                      [(2, 2), (2, 2), (16, 16), (16, 16)],
                                      [(32, 32), ],
                                      "float32", "FRACTAL_NZ", "ND", (
                                          32, 32), (32,), "ND", "ND",
                                      "dynamic_mul_fp32_NZ_ND", "success", "static"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
