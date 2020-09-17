#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", None, None)

#ut_case.add_test_cfg_cov_case("all")
# TODO add you test case

err1 = {"params": [{"shape":(24,1,16,16),"dtype":"float16",
                    "ori_shape":(24,1,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
                   {"shape":(16,2,3,4,16),"dtype":"int8",
                    "ori_shape":(16,2,3,4,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
                   "FRACTAL_NZ","NC1HWC0"],
         "expect": RuntimeError,
         "format_expect": ["NC1HWC0"],
         "support_expect": False}

# err2 = {"params": [{"shape":(24,1,16,17),"dtype":"float16",
#                     "ori_shape":(24,1,16,17),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                    {"shape":(16,2,3,4,16),"dtype":"float16",
#                     "ori_shape":(16,2,3,4,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                    "FRACTAL_NZ","NC1HWC0"],
#          "expect": RuntimeError,
#          "format_expect": ["NC1HWC0"],
#          "support_expect": False}

# err3 = {"params": [{"shape":(24,1,17,16),"dtype":"float16",
#                     "ori_shape":(24,1,17,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                    {"shape":(16,2,3,4,16),"dtype":"float16",
#                     "ori_shape":(16,2,3,4,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                    "FRACTAL_NZ","NC1HWC0"],
#          "expect": RuntimeError,
#          "format_expect": ["NC1HWC0"],
#          "support_expect": False}

# err4 = {"params": [{"shape":(24,1,16,16),"dtype":"float16",
#                     "ori_shape":(24,1,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                    {"shape":(16,2,3,4,17),"dtype":"float16",
#                     "ori_shape":(16,2,3,4,17),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                    "FRACTAL_NZ","NC1HWC0"],
#          "expect": RuntimeError,
#          "format_expect": ["NC1HWC0"],
#          "support_expect": False}

# err5 = {"params": [{"shape":(24,2,16,16),"dtype":"float16",
#                     "ori_shape":(24,2,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                    {"shape":(16,2,3,4,16),"dtype":"float16",
#                     "ori_shape":(16,2,3,4,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                    "FRACTAL_NZ","NC1HWC0"],
#          "expect": RuntimeError,
#          "format_expect": ["NC1HWC0"],
#          "support_expect": False}

# err6 = {"params": [{"shape":(25,1,16,16),"dtype":"float16",
#                     "ori_shape":(25,1,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                    {"shape":(16,2,3,4,16),"dtype":"float16",
#                     "ori_shape":(16,2,3,4,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                    "FRACTAL_NZ","NC1HWC0"],
#          "expect": RuntimeError,
#          "format_expect": ["NC1HWC0"],
#          "support_expect": False}

# case1 = {"params": [{"shape":(24,1,16,16),"dtype":"float16",
#                      "ori_shape":(24,1,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(16,2,3,4,16),"dtype":"float16",
#                      "ori_shape":(16,2,3,4,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case2 = {"params": [{"shape":(32,8,16,16),"dtype":"float16",
#                      "ori_shape":(32,8,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(113,2,1,16,16),"dtype":"float16",
#                      "ori_shape":(113,2,1,16,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case3 = {"params": [{"shape":(128,2,16,16),"dtype":"float16",
#                      "ori_shape":(128,2,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(17,4,1,32,16),"dtype":"float16",
#                      "ori_shape":(17,4,1,32,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case4 = {"params": [{"shape":(4082,248,16,16),"dtype":"float16",
#                      "ori_shape":(4082,248,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(3953,2,1,2041,16),"dtype":"float16",
#                      "ori_shape":(3953,2,1,2041,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case5 = {"params": [{"shape":(4082,256,16,16),"dtype":"float16",
#                      "ori_shape":(4082,256,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(4081,2,1,2041,16),"dtype":"float16",
#                      "ori_shape":(4081,2,1,2041,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case6 = {"params": [{"shape":(24,1,16,32),"dtype":"int8",
#                      "ori_shape":(24,1,16,32),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(16,2,3,4,32),"dtype":"int8",
#                      "ori_shape":(16,2,3,4,32),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case7 = {"params": [{"shape":(32,8,16,32),"dtype":"int8",
#                      "ori_shape":(32,8,16,32),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(113,2,1,16,32),"dtype":"int8",
#                      "ori_shape":(113,2,1,16,32),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case8 = {"params": [{"shape":(128,2,16,32),"dtype":"int8",
#                      "ori_shape":(128,2,16,32),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(17,4,1,32,32),"dtype":"int8",
#                      "ori_shape":(17,4,1,32,32),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

# case9 = {"params": [{"shape":(4082,248,16,32),"dtype":"int8",
#                      "ori_shape":(4082,248,16,32),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
#                     {"shape":(3953,2,1,2041,32),"dtype":"int8",
#                      "ori_shape":(3953,2,1,2041,32),"format":"NC1HWC0","ori_format":"NC1HWC0"},
#                     "FRACTAL_NZ","NC1HWC0"],
#          "expect": "success",
#          "format_expect": ["NC1HWC0"],
#          "support_expect": True}

case10 = {"params": [{"shape":(4082,256,16,32),"dtype":"int8",
                     "ori_shape":(4082,256,16,32),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
                     {"shape":(4081,2,1,2041,32),"dtype":"int8",
                      "ori_shape":(4081,2,1,2041,32),"format":"NC1HWC0","ori_format":"NC1HWC0"},
                      "FRACTAL_NZ","NC1HWC0"],
          "expect": "success",
          "format_expect": ["NC1HWC0"],
          "support_expect": True}

case11 = {"params": [{"shape":(32,13,16,16),"dtype":"float16",
                     "ori_shape":(32,13,16,16),"format":"FRACTAL_NZ","ori_format":"FRACTAL_NZ"},
                     {"shape":(200,2,1,16,16),"dtype":"float16",
                      "ori_shape":(200,2,1,16,16),"format":"NC1HWC0","ori_format":"NC1HWC0"},
                      "FRACTAL_NZ","NC1HWC0"],
          "expect": "success",
          "format_expect": ["NC1HWC0"],
          "support_expect": True}


ut_case.add_case(["Ascend910"], err1)
# ut_case.add_case(["Ascend910"], err2)
# ut_case.add_case(["Ascend910"], err3)
# ut_case.add_case(["Ascend910"], err4)
# ut_case.add_case(["Ascend910"], err5)
# ut_case.add_case(["Ascend910"], err6)
# ut_case.add_case(["Ascend910"], case1)
# ut_case.add_case(["Ascend910"], case2)
# ut_case.add_case(["Ascend910"], case3)
# ut_case.add_case(["Ascend910"], case4)
# ut_case.add_case(["Ascend910"], case5)
# ut_case.add_case(["Ascend910"], case6)
# ut_case.add_case(["Ascend910"], case7)
# ut_case.add_case(["Ascend910"], case8)
# ut_case.add_case(["Ascend910"], case9)
ut_case.add_case(["Ascend910"], case10)
ut_case.add_case(["Ascend910"], case11)


if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
