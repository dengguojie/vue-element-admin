#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
InplaceUpdate ut testcase
"""
from op_test_frame.ut import OpUT
from impl.dynamic.inplace_update import check_supported

ut_case = OpUT("InplaceUpdate", "impl.dynamic.inplace_update", "inplace_update")


def gen_dynamic_floormod_case(shape_x, shape_y, range_x, range_y,
                              dtype_val, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": shape_x,
                        "ori_format": "ND", "format": "ND", "range": range_x},
                       {"shape": shape_y, "dtype": "int32", "ori_shape": shape_y,
                        "ori_format": "ND", "format": "ND", "range": range_y},
                       {"shape": shape_x, "dtype": dtype_val, "ori_shape": shape_x,
                        "ori_format": "ND", "format": "ND", "range": range_x},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape": shape_y,
                        "ori_format": "ND", "format": "ND", "range": range_y}],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_floormod_case((-1,), (1,),
                                           ((1, None),), ((1, 1),),
                                           "float32", "dynamic_inplace_update_case", "success"))
def test_check_support(test_arg):
    """
    test_check_support
    """
    res, reason = check_supported({"shape": (-1, 50), "dtype": "float16", "format": "ND", "ori_shape": (50, 50), "ori_format": "ND"},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (10,), "ori_format": "ND"},
                    {"shape": (-1, 50), "dtype": "float16", "format": "ND", "ori_shape": (10, 50), "ori_format": "ND"},
                    "inplace_update_check_support_case_001")
    assert res
    res, reason = check_supported({"shape": (-1, 50), "dtype": "float16", "format": "ND", "ori_shape": (50, 50), "ori_format": "ND"},
                    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (10, 10), "ori_format": "ND"},
                    {"shape": (-1, 50), "dtype": "float16", "format": "ND", "ori_shape": (10, 50), "ori_format": "ND"},
                    "inplace_update_check_support_case_002")
    assert not res
    res, reason = check_supported({"shape": (-1, 50), "dtype": "int8", "format": "ND", "ori_shape": (50, 50), "ori_format": "ND"},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (10,), "ori_format": "ND"},
                    {"shape": (-1, 50), "dtype": "int8", "format": "ND", "ori_shape": (10, 50), "ori_format": "ND"},
                    "inplace_update_check_support_case_003")
    assert not res
    res, reason = check_supported({"shape": (-1, 50), "dtype": "float16", "format": "ND", "ori_shape": (50, 50), "ori_format": "ND"},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (10,), "ori_format": "ND"},
                    {"shape": (-1, 50), "dtype": "int8", "format": "ND", "ori_shape": (10, 50), "ori_format": "ND"},
                    "inplace_update_check_support_case_004")
    assert not res

ut_case.add_cust_test_func(test_func=test_check_support)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
