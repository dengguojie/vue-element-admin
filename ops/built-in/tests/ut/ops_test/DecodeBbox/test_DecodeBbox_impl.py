#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("DecodeBbox", None, None)

def gen_decode_bbox_case(shape_x, shape_y, dtype, decode_clip, case_name_val):
    return {"params": [{"shape": shape_x, "dtype": dtype, "format": "NCHW"},
                       {"shape": shape_x, "dtype": dtype, "format": "NCHW"},
                       {"shape": shape_y, "dtype": dtype, "format": "ND"},
                       decode_clip],
            "case_name": case_name_val,
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

case1 = gen_decode_bbox_case((4, 1, 1, 16), (4, 16), "float16", 8, "decode_bbox_1")
case2 = gen_decode_bbox_case((6, 16, 4), (16*6, 4), "float16", 8, "decode_bbox_2")


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)