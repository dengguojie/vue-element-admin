#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("BnTrainingUpdateV2", None, None)

case1 = {"params": [{"shape": (2,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,2,2,2,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2,2,2,2,16),"ori_format": "NC1HWC0"},
                    0.0001],
         "case_name": "BNTrainingUpdateV2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    {"shape": (1,2,2,2,2,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,2,16),"ori_format": "NDC1HWC0"},
                    0.0001],
         "case_name": "BNTrainingUpdateV2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    {"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    {"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    {"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    {"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    {"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    {"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    {"shape": (1,16,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,16,1,16),"ori_format": "NCHW"},
                    0.0001],
         "case_name": "BNTrainingUpdateV2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
def test_op_select_format(test_arg):

    from impl.bn_training_update_v2 import op_select_format
    op_select_format({"shape":(2,1,2,5,5,16), "ori_shape": (2,1,2,5,5,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape": (1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape": (1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},0.001)
    op_select_format({"shape":(1,2,1,32), "ori_shape": (1,2,1,32), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,12), "ori_shape": (1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,1), "ori_shape": (1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,1), "ori_shape":(1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,1), "ori_shape":(1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,1), "ori_shape":(1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,1), "ori_shape":(1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,1), "ori_shape":(1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},0.001)
    op_select_format({"shape":(-1,-1,-1,-1,-1), "ori_shape":(-1,-1,-1,-1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(1,-1,1,1,-1), "ori_shape":(1,-1,1,1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(1,-1,1,1,-1), "ori_shape":(1,-1,1,1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(1,-1,1,1,-1), "ori_shape":(1,-1,1,1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(1,-1,1,1,-1), "ori_shape":(1,-1,1,1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(-1,-1,-1,-1,-1), "ori_shape":(-1,-1,-1,-1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(1,-1,1,1,-1), "ori_shape":(1,-1,1,1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(1,-1,1,1,-1), "ori_shape":(1,-1,1,1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},0.001)

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_cust_test_func(test_func=test_op_select_format)
#if __name__ == '__main__':
#    ut_case.run("Ascend910A")
#    # ut_case.run()
#    exit(0)
