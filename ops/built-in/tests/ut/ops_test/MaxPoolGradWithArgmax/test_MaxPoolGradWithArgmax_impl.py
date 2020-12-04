
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolGradWithArgmax", None, None)

case1 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888,),"ori_format": "NHWC"},
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "VALID"],
         "case_name": "max_pool_grad_with_arxmax_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,70,514,16),"ori_format": "NHWC"},
                    {"shape": (32,1,14,86,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,14,86,16),"ori_format": "NHWC"},
                    {"shape": (1576960,), "dtype": "uint16", "format": "NHWC", "ori_shape": (1576960,),"ori_format": "NHWC"},
                    {"shape": (32,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,70,514,16),"ori_format": "NHWC"},
                    [1, 5, 8, 1],
                    [1, 5, 6, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,70,514,16),"ori_format": "NHWC"},
                    {"shape": (1,1,14,86,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,14,86,16),"ori_format": "NHWC"},
                    {"shape": (49280,), "dtype": "uint16", "format": "NHWC", "ori_shape": (49280,),"ori_format": "NHWC"},
                    {"shape": (1,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,70,514,16),"ori_format": "NHWC"},
                    [1, 5, 8, 1],
                    [1, 5, 6, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888,),"ori_format": "NHWC"},
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2,32,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,32,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888*16,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888*16,),"ori_format": "NHWC"},
                    {"shape": (2,32,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,96,144,16),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend710", "Ascend910"], case5)


if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
