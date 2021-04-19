#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("UnsortedSegmentSum", "impl.dynamic.unsorted_segment_sum", "unsorted_segment_sum")


def test_op_select_format(test_arg):
    from impl.dynamic.unsorted_segment_sum import op_select_format
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (28, 16, 16), "dtype": "int32", "format": "ND", "ori_shape": (28, 16, 16),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                     {"shape": (200, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (200, 28, 16, 16),"ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "int8", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (28, 16, 16), "dtype": "int32", "format": "ND", "ori_shape": (28, 16, 16),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                     {"shape": (200, 28, 16, 16), "dtype": "int8", "format": "NCHW", "ori_shape": (200, 28, 16, 16),"ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "uint8", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (20,), "dtype": "int32", "format": "ND", "ori_shape": (20,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                     {"shape": (200, 28, 16, 16), "dtype": "uint8", "format": "NCHW", "ori_shape": (200, 28, 16, 16),"ori_format": "NCHW"})


def test_check_supported(test_arg):
    from impl.dynamic.unsorted_segment_sum import check_supported
    check_supported({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (28, 16, 16), "dtype": "int32", "format": "ND", "ori_shape": (28, 16, 16),"ori_format": "ND"},
                     {"shape": (200, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (200,28, 16, 16),"ori_format": "NCHW"},
                    {"shape": (200, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (200,28, 16, 16),"ori_format": "NCHW"})

    check_supported({"shape": (25, 28, 16, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (25, 28, 16, 1),"ori_format": "NCHW"},
                     {"shape": (25,), "dtype": "int32", "format": "ND", "ori_shape": (25,),"ori_format": "ND"},
                     {"shape": (300, 28, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (300, 28, 16, 16),"ori_format": "NCHW"},
                    {"shape": (25, 28, 16, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (25, 28, 16, 1),"ori_format": "NCHW"})

    check_supported({"shape": (25, 28, -1, -1), "dtype": "float32", "format": "NCHW", "ori_shape": (25, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (25,), "dtype": "int32", "format": "ND", "ori_shape": (25,),"ori_format": "ND"},
                     {"shape": (300, 28, 16, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (300, 28, 16, 16),"ori_format": "NCHW"},
                    {"shape": (25, 28, -1, -1), "dtype": "float32", "format": "NCHW", "ori_shape": (25, 28, 16, 16),"ori_format": "NCHW"})

    check_supported({"shape": (25, 28), "dtype": "uint8", "format": "ND", "ori_shape": (25, 28),"ori_format": "ND"},
                     {"shape": (25,), "dtype": "int64", "format": "ND", "ori_shape": (25,),"ori_format": "ND"},
                     {"shape": (300, 28), "dtype": "uint8", "format": "ND", "ori_shape": (300, 28),"ori_format": "ND"},
                    {"shape": (25, 28), "dtype": "uint8", "format": "ND", "ori_shape": (25, 28),"ori_format": "ND"},)

    check_supported({"shape": (25, 28), "dtype": "int32", "format": "ND", "ori_shape": (25, 28),"ori_format": "ND"},
                     {"shape": (25,), "dtype": "int64", "format": "ND", "ori_shape": (25,),"ori_format": "ND"},
                     {"shape": (300, 28), "dtype": "int32", "format": "ND", "ori_shape": (300, 28),"ori_format": "ND"},
                    {"shape": (25, 28), "dtype": "int32", "format": "ND", "ori_shape": (25, 28),"ori_format": "ND"})

def gen_dynamic_floormod_case(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": "float32", "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "int32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y},
                       {"shape": shape_x, "dtype": "int32", "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "float32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def gen_dynamic_floormod_case_int32(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": "int32", "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "int32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y},
                       {"shape": shape_x, "dtype": "int32", "ori_shape":shape_x,"ori_format":"ND", "format":"ND","range": range_x},
                       {"shape": shape_y, "dtype": "int32", "ori_shape":shape_y,"ori_format":"ND", "format":"ND","range": range_y}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("Ascend910A",
                 gen_dynamic_floormod_case((-1,), (1,),
                                           ((1, None),), ((1, 1),),
                                           "float32", "dynamic_unsorted_segment_sum_case", "success"))
ut_case.add_case("Ascend910A",
                 gen_dynamic_floormod_case_int32((-1,), (1,),
                                           ((1, None),), ((1, 1),),
                                           "int32", "dynamic_unsorted_segment_sum_case", "success"))
ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_check_supported)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
