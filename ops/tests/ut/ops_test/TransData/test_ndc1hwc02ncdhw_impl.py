#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")

#ut_case.add_test_cfg_cov_case("all")
# TODO add you test case

err1 = {"params": [{"shape":(1,1,16,1,1,16),"dtype":"float16",
                    "ori_shape":(1,1,16,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,256,1,1,1),"dtype":"float32",
                    "ori_shape":(1,256,1,1,1),"format":"NCDHW","ori_format":"NCDHW"},
                   "NDC1HWC0","NCDHW"],
         "expect": RuntimeError,
         "format_expect": ["NCDHW"],
         "support_expect": False}

err2 = {"params": [{"shape":(1,1,16,1,1,17),"dtype":"float16",
                    "ori_shape":(1,1,16,1,1,17),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,256,1,1,1),"dtype":"float16",
                    "ori_shape":(1,256,1,1,1),"format":"NCDHW","ori_format":"NCDHW"},
                   "NDC1HWC0","NCDHW"],
         "expect": RuntimeError,
         "format_expect": ["NCDHW"],
         "support_expect": False}

err3 = {"params": [{"shape":(2,1,16,1,1,16),"dtype":"float16",
                    "ori_shape":(2,1,16,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,256,1,1,1),"dtype":"float16",
                    "ori_shape":(1,256,1,1,1),"format":"NCDHW","ori_format":"NCDHW"},
                   "NDC1HWC0","NCDHW"],
         "expect": RuntimeError,
         "format_expect": ["NCDHW"],
         "support_expect": False}

err4 = {"params": [{"shape":(1,1,15,1,1,16),"dtype":"float16",
                    "ori_shape":(1,1,15,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,256,1,1,1),"dtype":"float16",
                    "ori_shape":(1,256,1,1,1),"format":"NCDHW","ori_format":"NCDHW"},
                   "NDC1HWC0","NCDHW"],
         "expect": RuntimeError,
         "format_expect": ["NCDHW"],
         "support_expect": False}

err5 = {"params": [{"shape":(1,1,16,1,1,16),"dtype":"float16",
                    "ori_shape":(1,1,16,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,256,1,1,1),"dtype":"float16",
                    "ori_shape":(1,256,1,1,1),"format":"NCDHW","ori_format":"NCDHW"},
                   "NDC1HWC0","NCDHW"],
         "expect": RuntimeError,
         "format_expect": ["NCDHW"],
         "support_expect": False}

case1 = {"params": [{"shape":(1,2,8,126,126,16),"dtype":"float16",
                     "ori_shape":(1,2,8,126,126,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(1,128,2,126,126),"dtype":"float16",
                     "ori_shape":(1,128,2,126,126),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

case2 = {"params": [{"shape":(2,3,2,16,3,16),"dtype":"float16",
                     "ori_shape":(2,3,2,16,3,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,32,3,16,3),"dtype":"float16",
                     "ori_shape":(2,32,3,16,3),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

case3 = {"params": [{"shape":(2,3,4096,16,2,16),"dtype":"float16",
                     "ori_shape":(2,3,4096,16,2,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,65536,3,16,2),"dtype":"float16",
                     "ori_shape":(2,65536,3,16,2),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

case4 = {"params": [{"shape":(2,3,2,1,5120,16),"dtype":"float16",
                     "ori_shape":(2,3,2,1,5120,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,32,3,1,5120),"dtype":"float16",
                     "ori_shape":(2,32,3,1,5120),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

# split big
case5 = {"params": [{"shape":(2,3,4375,1,5120,16),"dtype":"float16",
                     "ori_shape":(2,3,4375,1,5120,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,70000,3,1,5120),"dtype":"float16",
                     "ori_shape":(2,70000,3,1,5120),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

case6 = {"params": [{"shape":(2,3,4094,1,3960,16),"dtype":"float16",
                     "ori_shape":(2,3,4094,1,3960,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,65500,3,1,3960),"dtype":"float16",
                     "ori_shape":(2,65500,3,1,3960),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

# hwc0_core
case7 = {"params": [{"shape":(2,3,3952,1,3952,16),"dtype":"float16",
                     "ori_shape":(2,3,3952,1,3952,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,63232,3,1,3952),"dtype":"float16",
                     "ori_shape":(2,63232,3,1,3952),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

case8 = {"params": [{"shape":(2,3,3952,1,3951,16),"dtype":"float16",
                     "ori_shape":(2,3,3952,1,3951,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,63232,3,1,3951),"dtype":"float16",
                     "ori_shape":(2,63232,3,1,3951),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

case9 = {"params": [{"shape":(2,3,4096,15,2,16),"dtype":"float16",
                     "ori_shape":(2,3,4096,15,2,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,65536,3,15,2),"dtype":"float16",
                     "ori_shape":(2,65536,3,15,2),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

case10 = {"params": [{"shape":(2,3,2,15,3,16),"dtype":"float16",
                     "ori_shape":(2,3,2,15,3,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(2,32,3,15,3),"dtype":"float16",
                     "ori_shape":(2,32,3,15,3),"format":"NCDHW","ori_format":"NCDHW"},
                    "NDC1HWC0","NCDHW"],
         "expect": "success",
         "format_expect": ["NCDHW"],
         "support_expect": True}

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
