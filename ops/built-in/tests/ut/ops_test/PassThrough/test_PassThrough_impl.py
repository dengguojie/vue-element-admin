#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("PassThrough", None, None)

def do_pass_through(shape, dtype, dfmt, stride, reverse, case_name):

    if reverse is True:
        out_shape = (shape[0], shape[1] * stride, shape[2] * stride,
                     shape[3]//(stride*stride))
    else:
        out_shape = (shape[0], shape[1] // stride, shape[2] // stride,
                     shape[3] * (stride * stride))

    in_dic = {'shape': shape, 'dtype': dtype, 'format': dfmt, 'ori_shape':shape, "ori_format":"NHWC"}
    w_dic = {'shape': shape, 'dtype': "float16", 'format': "NHWC", 'ori_shape':shape, "ori_format":"NHWC"}
    out_dic = {'shape': out_shape, 'dtype': dtype, 'format': dfmt, 'ori_shape':out_shape, "ori_format":"NHWC"}

    return {"params": [in_dic, w_dic, out_dic, stride, reverse],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = do_pass_through((100, 87, 870, 11), "float32", "NHWC", 87, False,
                        "pass_through_1")
case2 = do_pass_through((1, 20, 20, 1), "int16", "NHWC", 2, False,
                        "pass_through_2")
case3 = do_pass_through((2, 2, 2, 70001), "uint16", "NHWC", 2, False,
                        "pass_through_3")
case4 = do_pass_through((2, 200, 20, 160), "int8", "NHWC", 20, False,
                        "pass_through_4")
case5 = do_pass_through((1, 1, 1, 4), "float32", "NHWC", 2, True,
                        "pass_through_5")
case6 = do_pass_through((1, 1, 1, 720004), "float32", "NHWC", 2, True,
                        "pass_through_6")



ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)

