#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("NotEqual", "impl.dynamic.not_equal", "not_equal")


def gen_dynamic_not_equal_case(shape_x, shape_y, range_x, range_y,
                               dtype_val, kernel_name_val, expect):
    return {
        "params": [{
            "ori_shape": shape_x,
            "shape": shape_x,
            "ori_format": "ND",
            "format": "ND",
            "dtype": dtype_val,
            "range": range_x
        }, {
            "ori_shape": shape_y,
            "shape": shape_y,
            "ori_format": "ND",
            "format": "ND",
            "dtype": dtype_val,
            "range": range_y
        }, {
            "ori_shape": shape_y,
            "shape": shape_y,
            "ori_format": "ND",
            "format": "ND",
            "dtype": "int8",
            "range": range_y
        }],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }


case1 = gen_dynamic_not_equal_case((-1,), (1,), ((2, 16),), ((1, 1),), "float16", "not_equal_fp16_ND_case1", "success")
case2 = gen_dynamic_not_equal_case((-1,), (1,), ((2, 16),), ((1, 1),), "int64", "not_equal_int64_ND_case2", "success")
case3 = gen_dynamic_not_equal_case((2, 128), (1,), ((2, 2), (128, 128)), ((1, 1),), "int64", "not_equal_int64_ND_case3", "success")
case4 = gen_dynamic_not_equal_case((-1, 128), (128,), ((2, 16), (128, 128)), ((128, 128),), "int64", "not_equal_int64_ND_case4", "success")

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
