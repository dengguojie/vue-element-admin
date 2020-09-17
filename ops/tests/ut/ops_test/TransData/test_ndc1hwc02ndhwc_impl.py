#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")

#ut_case.add_test_cfg_cov_case("all")
# TODO add you test case

err1 = {"params": [{"shape":(1,1,16,1,1,16),"dtype":"float16",
                    "ori_shape":(1,1,16,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,1,1,1,256),"dtype":"float32",
                    "ori_shape":(1,1,1,1,256),"format":"NDHWC","ori_format":"NDHWC"},
                   "NDC1HWC0","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect": False}

err2 = {"params": [{"shape":(1,1,16,1,1,17),"dtype":"float16",
                    "ori_shape":(1,1,16,1,1,17),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,1,1,1,256),"dtype":"float16",
                    "ori_shape":(1,1,1,1,256),"format":"NDHWC","ori_format":"NDHWC"},
                   "NDC1HWC0","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect": False}

err3 = {"params": [{"shape":(2,1,16,1,1,16),"dtype":"float16",
                    "ori_shape":(2,1,16,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,1,1,1,256),"dtype":"float16",
                    "ori_shape":(1,1,1,1,256),"format":"NDHWC","ori_format":"NDHWC"},
                   "NDC1HWC0","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect": False}

err4 = {"params": [{"shape":(1,1,15,1,1,16),"dtype":"float16",
                    "ori_shape":(1,1,15,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                   {"shape":(1,1,1,1,256),"dtype":"float16",
                    "ori_shape":(1,1,1,1,256),"format":"NDHWC","ori_format":"NDHWC"},
                   "NDC1HWC0","NDHWC"],
         "expect": RuntimeError,
         "format_expect": ["NDHWC"],
         "support_expect": False}

case1 = {"params": [{"shape":(1,1,16,1,1,16),"dtype":"float16",
                     "ori_shape":(1,1,16,1,1,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(1,1,1,1,256),"dtype":"float16",
                     "ori_shape":(1,1,1,1,256),"format":"NDHWC","ori_format":"NDHWC"},
                    "NDC1HWC0","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect": True}

case2 = {"params": [{"shape":(4,5,8,26,26,16),"dtype":"float16",
                     "ori_shape":(4,5,8,26,26,16),"format":"NDC1HWC0","ori_format":"NDC1HWC0"},
                    {"shape":(4,5,26,26,128),"dtype":"float16",
                     "ori_shape":(4,5,26,26,128),"format":"NDHWC","ori_format":"NDHWC"},
                    "NDC1HWC0","NDHWC"],
         "expect": "success",
         "format_expect": ["NDHWC"],
         "support_expect": True}



ut_case.add_case(["Ascend910"], err1)
# ut_case.add_case(["Ascend910"], err2)
# ut_case.add_case(["Ascend910"], err3)
# ut_case.add_case(["Ascend910"], err4)
# ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)


if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
