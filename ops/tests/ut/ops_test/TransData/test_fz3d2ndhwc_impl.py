#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", None, None)

#ut_case.add_test_cfg_cov_case("all")
# TODO add you test case

err1 = {"params": [{"shape":(120,4,16,16),"dtype":"float16",
                    "ori_shape":(120,4,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                   {"shape":(64,3,4,5,32),"dtype":"float32",
                    "ori_shape":(64,3,4,5,32),"format":"NDHWC","ori_format":"NDHWC"},
                   "FRACTAL_Z_3D","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect":False}

err2 = {"params": [{"shape":(120,4,32,16),"dtype":"float16",
                    "ori_shape":(120,4,32,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                   {"shape":(64,3,4,5,32),"dtype":"float16",
                    "ori_shape":(64,3,4,5,32),"format":"NDHWC","ori_format":"NDHWC"},
                   "FRACTAL_Z_3D","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect":False}

err3 = {"params": [{"shape":(120,4,16,32),"dtype":"float16",
                    "ori_shape":(120,4,16,32),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                   {"shape":(64,3,4,5,32),"dtype":"float16",
                    "ori_shape":(64,3,4,5,32),"format":"NDHWC","ori_format":"NDHWC"},
                   "FRACTAL_Z_3D","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect":False}

err4 = {"params": [{"shape":(120,3,16,16),"dtype":"float16",
                    "ori_shape":(120,3,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                   {"shape":(64,3,4,5,32),"dtype":"float16",
                    "ori_shape":(64,3,4,5,32),"format":"NDHWC","ori_format":"NDHWC"},
                   "FRACTAL_Z_3D","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect":False}

err5 = {"params": [{"shape":(100,4,16,16),"dtype":"float16",
                    "ori_shape":(100,4,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                   {"shape":(64,3,4,5,32),"dtype":"float16",
                    "ori_shape":(64,3,4,5,32),"format":"NDHWC","ori_format":"NDHWC"},
                   "FRACTAL_Z_3D","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect":False}

case1 = {"params": [{"shape":(27,1,16,16),"dtype":"float16",
                     "ori_shape":(27,1,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(16,3,3,3,5),"dtype":"float16",
                     "ori_shape":(16,3,3,3,5),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case2 = {"params": [{"shape":(27,1,16,16),"dtype":"float16",
                     "ori_shape":(27,1,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(16,3,3,3,16),"dtype":"float16",
                     "ori_shape":(16,3,3,3,16),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case3 = {"params": [{"shape":(27,2,16,16),"dtype":"float16",
                     "ori_shape":(27,2,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(32,3,3,3,16),"dtype":"float16",
                     "ori_shape":(32,3,3,3,16),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case4 = {"params": [{"shape":(216,8,16,16),"dtype":"float16",
                     "ori_shape":(216,8,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(128,3,3,3,128),"dtype":"float16",
                     "ori_shape":(128,3,3,3,128),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case5 = {"params": [{"shape":(24,8,16,16),"dtype":"float16",
                     "ori_shape":(24,8,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(128,1,1,3,128),"dtype":"float16",
                     "ori_shape":(128,1,1,3,128),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case6 = {"params": [{"shape":(27,1,16,16),"dtype":"float32",
                     "ori_shape":(27,1,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(16,3,3,3,5),"dtype":"float32",
                     "ori_shape":(16,3,3,3,5),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case7 = {"params": [{"shape":(27,1,16,16),"dtype":"float32",
                     "ori_shape":(27,1,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(16,3,3,3,16),"dtype":"float32",
                     "ori_shape":(16,3,3,3,16),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case8 = {"params": [{"shape":(27,2,16,16),"dtype":"float32",
                     "ori_shape":(27,2,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(32,3,3,3,16),"dtype":"float32",
                     "ori_shape":(32,3,3,3,16),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case9 = {"params": [{"shape":(216,8,16,16),"dtype":"float32",
                     "ori_shape":(216,8,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(128,3,3,3,128),"dtype":"float32",
                     "ori_shape":(128,3,3,3,128),"format":"NDHWC","ori_format":"NDHWC"},
                    "FRACTAL_Z_3D","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect":True}

case10 = {"params": [{"shape":(24,8,16,16),"dtype":"float32",
                     "ori_shape":(24,8,16,16),"format":"FRACTAL_Z_3D","ori_format":"FRACTAL_Z_3D"},
                    {"shape":(128,1,1,3,128),"dtype":"float32",
                     "ori_shape":(128,1,1,3,128),"format":"NDHWC","ori_format":"NDHWC"},
                     "FRACTAL_Z_3D","NDHWC"],
          "expect": "success",
          "format_expect": ["NDHWC"],
          "support_expect":True}

ut_case.add_case(["Ascend910"], err1)
# ut_case.add_case(["Ascend910"], err2)
# ut_case.add_case(["Ascend910"], err3)
# ut_case.add_case(["Ascend910"], err4)
# ut_case.add_case(["Ascend910"], err5)
# ut_case.add_case(["Ascend910"], case1)
# ut_case.add_case(["Ascend910"], case2)
# ut_case.add_case(["Ascend910"], case3)
# ut_case.add_case(["Ascend910"], case4)
# ut_case.add_case(["Ascend910"], case5)
# ut_case.add_case(["Ascend910"], case6)
# ut_case.add_case(["Ascend910"], case7)
# ut_case.add_case(["Ascend910"], case8)
ut_case.add_case(["Ascend910"], case9)
ut_case.add_case(["Ascend910"], case10)


if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
