#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("SppPooling", None, None)

POOLING_CEIL = 0
POOLING_FLOOR = 1
POOLING_ROUND = 2
MAX_POOLING = 0
AVG_POOLING = 1
OTHER_POOLING = 2

def call_spp_pooling(shape, dtype, global_pooling, mode, window,
                     pad, stride, ceil_mode, case_name):
    n, c1, h, w, c0 = shape[0], shape[1], shape[2], shape[3], shape[4]
    if global_pooling is True:
        out_h = 1
        out_w = 1
    else:
        stride0 = stride[0] if stride[0] != 0 else window[0]
        stride1 = stride[1] if stride[1] != 0 else window[1]
        if ceil_mode == "CEIL":
            out_h = (h+2*pad[0]-window[0]+stride0-1)//stride0+1
            out_w = (h+2*pad[2]-window[1]+stride1-1)//stride1+1
        else:
            out_h = (h+2*pad[0]-window[0])//stride0+1
            out_w = (h+2*pad[2]-window[1])//stride1+1
        if pad[0] > 0 or pad[2] > 0:
            if (out_h - 1) * stride0 >= h + pad[0]:
                out_h = out_h - 1
            if (out_w - 1) * stride1 >= w + pad[2]:
                out_w = out_w - 1

    in_shape = (n, c1, h, w, c0)
    out_shape = (n, c1, out_h, out_w, c0)

    in_dic = {'shape': in_shape, 'dtype': dtype, 'format': "NC1HWC0", "ori_shape": in_shape, "ori_format": "NC1HWC0"}
    out_dic = {'shape': out_shape, 'dtype': dtype, 'format': "NC1HWC0", "ori_shape": out_shape, "ori_format": "NC1HWC0"}

    pooling_mode = MAX_POOLING if mode == "MAX" else (AVG_POOLING if mode == "AVG" else OTHER_POOLING)
    pooling_ceil_mode = POOLING_CEIL if ceil_mode == "CEIL" else (POOLING_FLOOR if ceil_mode == "FLOOR" else POOLING_ROUND)

    return {"params": [in_dic, out_dic, global_pooling, pooling_mode, window,
                       pad, stride, pooling_ceil_mode],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = call_spp_pooling((1, 1, 2, 2, 16), "float16", True, "MAX",
                         (2, 2), (0, 0, 0, 0), (1, 1), "FLOOR","spp_pooling_1")
case2 = call_spp_pooling((1, 2, 299, 299, 16), "float16", False, "AVG",
                         (20, 20), (1, 1, 1, 1), (20, 20), "CEIL","spp_pooling_2")
case3 = call_spp_pooling((1, 1, 224, 224, 16), "float16", False, "MAX",
                         (14, 14), (0, 0, 0, 0), (14, 14), "CEIL","spp_pooling_3")
case4 = call_spp_pooling((2, 2, 32, 32, 16), "float32", False, "MAX",
                         (16, 16), (0, 0, 0, 0), (16, 16), "CEIL","spp_pooling_4")
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


