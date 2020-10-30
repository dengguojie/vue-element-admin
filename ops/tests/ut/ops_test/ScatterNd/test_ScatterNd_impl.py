#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ScatterNdD", None, None)


def gen_data(ind_shape, updates_shape, output_y_shape, shape_out,
               update_dtype, case_name,
               out_dtype, ind_dtype="int32"):
    indices = {"shape": ind_shape, "ori_shape": ind_shape, "format": "NHWC", "ori_format": "NHWC", "dtype": ind_dtype}
    updates = {"shape": updates_shape, "ori_shape": updates_shape, "format": "NHWC", "ori_format": "NHWC", "dtype": update_dtype}
    output_y = {"shape": output_y_shape, "ori_shape": output_y_shape, "format": "NHWC", "ori_format": "NHWC", "dtype": out_dtype}
    shape = shape_out

    return {"params": [indices, updates, output_y, shape],
            "case_name":case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_data((10241, 3), (10241, 3, 3), (3, 3, 3), (3, 3, 3), "float32",
                   "scatter_nd_1", "float32")
case2 = gen_data((4, ), (4,), (4,), (4,), "float32",
                   "scatter_nd_2", "float32")
case3 = gen_data((2, 3), (2, 3, 3), (3, 3, 3), (3, 3, 3), "float32",
                   "scatter_nd_3", "float32")
case4 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "float32",
                   "scatter_nd_4", "float32")
case5 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "float16",
                   "scatter_nd_5", "float16")
case6 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "int32",
                   "scatter_nd_6", "int32")
case7 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "int8",
                   "scatter_nd_7", "int8")
case8 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "uint8",
                   "scatter_nd_8", "uint8")

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)
ut_case.add_case("Ascend910", case3)
ut_case.add_case("Ascend910", case4)
ut_case.add_case("Ascend910", case5)
ut_case.add_case("Ascend910", case6)
ut_case.add_case("Ascend910", case7)
ut_case.add_case("Ascend910", case8)

if __name__ == '__main__':
    ut_case.run("Ascend910")
