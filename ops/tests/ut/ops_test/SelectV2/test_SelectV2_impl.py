#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SelectV2", None, None)

def gen_select_case(shape_var, dtype, expect, case_name_val,  bool_dtype="int8"):
    return {"params": [{"shape": shape_var, "dtype": bool_dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": shape_var, "dtype": dtype, "ori_shape": shape_var, "ori_format": "NCHW", "format": "NCHW"}],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_select_case((21,), "float16", "success", "select_v2_1")
case2 = gen_select_case((21,), "int32", "success", "select_v2_2")
case3 = gen_select_case((21,), "float32", "success", "select_v2_3")
case4 = gen_select_case((100000,), "float32", "success", "select_v2_4")
case5 = gen_select_case((100000, 2147), "float16", "success", "select_v2_5")
case6 = gen_select_case((20, 5, 5, 2, 2147, 20, 5, 10), "int32", "success", "select_v2_6")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)


if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)