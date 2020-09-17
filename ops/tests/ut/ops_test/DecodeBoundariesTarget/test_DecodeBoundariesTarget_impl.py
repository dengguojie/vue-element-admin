#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("DecodeBoundariesTarget", None, None)

def gen_decode_boundaries_target_case(shape_x, shape_y, dtype, case_name_val):
    return {"params": [{"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_y, "dtype": dtype, "ori_shape": shape_y, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_x, "dtype": dtype, "ori_shape": shape_x, "ori_format": "ND", "format": "ND"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_decode_boundaries_target_case((1, 1), (1, 4), "float16", "decode_boundaries_target_1")
case2 = gen_decode_boundaries_target_case((16, 1), (16, 4), "float16", "decode_boundaries_target_2")
case3 = gen_decode_boundaries_target_case((100, 1), (100, 4), "float16", "decode_boundaries_target_3")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)