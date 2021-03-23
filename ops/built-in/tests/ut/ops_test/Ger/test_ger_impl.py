#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("Ger", None, None)

# common cases
case0 = {"params": [{"shape":(1,),    "dtype":"float32", "format":"NCHW", "ori_shape":(1,),    "ori_format":"NCHW"},
                    {"shape":(200,),  "dtype":"float32", "format":"NCHW", "ori_shape":(200,),  "ori_format":"NCHW"},
                    {"shape":(1,200), "dtype":"float32", "format":"NCHW", "ori_shape":(1,200), "ori_format":"NCHW"}],
            "case_name": "ger_0",
            "expect": "success",
            "support_expect": True}

case1 = {"params": [{"shape":(1,),  "dtype":"float32", "format":"NCHW", "ori_shape":(1,),  "ori_format":"NCHW"},
                    {"shape":(1,),  "dtype":"float32", "format":"NCHW", "ori_shape":(1,),  "ori_format":"NCHW"},
                    {"shape":(1,1), "dtype":"float32", "format":"NCHW", "ori_shape":(1,1), "ori_format":"NCHW"}],
            "case_name": "ger_1",
            "expect": "success",
            "support_expect": True}

case2 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"NCHW", "ori_shape":(100,),    "ori_format":"NCHW"},
                    {"shape":(200,),    "dtype":"float32", "format":"NCHW", "ori_shape":(200,),    "ori_format":"NCHW"},
                    {"shape":(100,200), "dtype":"float32", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"}],
            "case_name": "ger_2",
            "expect": "success",
            "support_expect": True}

case3 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"NHWC", "ori_shape":(100,),    "ori_format":"NHWC"},
                    {"shape":(200,),    "dtype":"float32", "format":"NHWC", "ori_shape":(200,),    "ori_format":"NHWC"},
                    {"shape":(100,200), "dtype":"float32", "format":"NHWC", "ori_shape":(100,200), "ori_format":"NHWC"}],
            "case_name": "ger_3",
            "expect": "success",
            "support_expect": True}

case4 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"NC1HWC0", "ori_shape":(100,),    "ori_format":"NC1HWC0"},
                    {"shape":(200,),    "dtype":"float32", "format":"NC1HWC0", "ori_shape":(200,),    "ori_format":"NC1HWC0"},
                    {"shape":(100,200), "dtype":"float32", "format":"NC1HWC0", "ori_shape":(100,200), "ori_format":"NC1HWC0"}],
            "case_name": "ger_4",
            "expect": "success",
            "support_expect": True}

case5 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"ND", "ori_shape":(100,),    "ori_format":"ND"},
                    {"shape":(200,),    "dtype":"float32", "format":"ND", "ori_shape":(200,),    "ori_format":"ND"},
                    {"shape":(100,200), "dtype":"float32", "format":"ND", "ori_shape":(100,200), "ori_format":"ND"}],
            "case_name": "ger_5",
            "expect": "success",
            "support_expect": True}

case6 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"NCHW", "ori_shape":(100,),    "ori_format":"NCHW"},
                    {"shape":(200,),    "dtype":"float16", "format":"NCHW", "ori_shape":(200,),    "ori_format":"NCHW"},
                    {"shape":(100,200), "dtype":"float16", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"}],
            "case_name": "ger_6",
            "expect": "success",
            "support_expect": True}

case7 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"NHWC", "ori_shape":(100,),    "ori_format":"NHWC"},
                    {"shape":(200,),    "dtype":"float16", "format":"NHWC", "ori_shape":(200,),    "ori_format":"NHWC"},
                    {"shape":(100,200), "dtype":"float16", "format":"NHWC", "ori_shape":(100,200), "ori_format":"NHWC"}],
            "case_name": "ger_7",
            "expect": "success",
            "support_expect": True}

case8 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"NC1HWC0", "ori_shape":(100,),    "ori_format":"NC1HWC0"},
                    {"shape":(200,),    "dtype":"float16", "format":"NC1HWC0", "ori_shape":(200,),    "ori_format":"NC1HWC0"},
                    {"shape":(100,200), "dtype":"float16", "format":"NC1HWC0", "ori_shape":(100,200), "ori_format":"NC1HWC0"}],
            "case_name": "ger_8",
            "expect": "success",
            "support_expect": True}

case9 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"ND", "ori_shape":(100,),    "ori_format":"ND"},
                    {"shape":(200,),    "dtype":"float16", "format":"ND", "ori_shape":(200,),    "ori_format":"ND"},
                    {"shape":(100,200), "dtype":"float16", "format":"ND", "ori_shape":(100,200), "ori_format":"ND"}],
            "case_name": "ger_9",
            "expect": "success",
            "support_expect": True}

case10 = {"params": [{"shape":(100,),    "dtype":"int32", "format":"NCHW", "ori_shape":(100,),    "ori_format":"NCHW"},
                    {"shape":(200,),    "dtype":"int32", "format":"NCHW", "ori_shape":(200,),    "ori_format":"NCHW"},
                    {"shape":(100,200), "dtype":"int32", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"}],
            "case_name": "ger_10",
            "expect": RuntimeError,
            "support_expect": True}

case11 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"NCHW", "ori_shape":(100,),    "ori_format":"NCHW"},
                    {"shape":(200,),    "dtype":"float16", "format":"NCHW", "ori_shape":(200,),    "ori_format":"NCHW"},
                    {"shape":(100,200), "dtype":"float16", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"}],
            "case_name": "ger_11",
            "expect": RuntimeError,
            "support_expect": True}

case12 = {"params": [{"shape":(100,200), "dtype":"float32", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"},
                    {"shape":(200,200), "dtype":"float32", "format":"NCHW", "ori_shape":(200,200), "ori_format":"NCHW"},
                    {"shape":(100,200), "dtype":"float32", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"}],
            "case_name": "ger_12",
            "expect": RuntimeError,
            "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend910"], case0)
ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend910"], case12)

#precision cases
# pylint: disable=unused-argument
def calc_expect_func(x, vec2, y):
    input_x_value = x["value"]
    input_vec2_value = vec2["value"]
    res = np.outer(input_x_value, input_vec2_value)
    res = res.astype(x["dtype"])
    return res

case13 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"NCHW", "ori_shape":(100,),    "ori_format":"NCHW"},
                     {"shape":(200,),    "dtype":"float32", "format":"NCHW", "ori_shape":(200,),    "ori_format":"NCHW"},
                     {"shape":(100,200), "dtype":"float32", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"}],
            "case_name": "ger_13",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
            "support_expect": True}

case14 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"NHWC", "ori_shape":(100,),    "ori_format":"NHWC"},
                     {"shape":(200,),    "dtype":"float32", "format":"NHWC", "ori_shape":(200,),    "ori_format":"NHWC"},
                     {"shape":(100,200), "dtype":"float32", "format":"NHWC", "ori_shape":(100,200), "ori_format":"NHWC"}],
            "case_name": "ger_14",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
            "support_expect": True}

case15 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"NC1HWC0", "ori_shape":(100,),    "ori_format":"NC1HWC0"},
                     {"shape":(200,),    "dtype":"float32", "format":"NC1HWC0", "ori_shape":(200,),    "ori_format":"NC1HWC0"},
                     {"shape":(100,200), "dtype":"float32", "format":"NC1HWC0", "ori_shape":(100,200), "ori_format":"NC1HWC0"}],
            "case_name": "ger_15",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
            "support_expect": True}

case16 = {"params": [{"shape":(100,),    "dtype":"float32", "format":"ND", "ori_shape":(100,),    "ori_format":"ND"},
                     {"shape":(200,),    "dtype":"float32", "format":"ND", "ori_shape":(200,),    "ori_format":"ND"},
                     {"shape":(100,200), "dtype":"float32", "format":"ND", "ori_shape":(100,200), "ori_format":"ND"}],
            "case_name": "ger_16",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
            "support_expect": True}

case17 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"NCHW", "ori_shape":(100,),    "ori_format":"NCHW"},
                     {"shape":(200,),    "dtype":"float16", "format":"NCHW", "ori_shape":(200,),    "ori_format":"NCHW"},
                     {"shape":(100,200), "dtype":"float16", "format":"NCHW", "ori_shape":(100,200), "ori_format":"NCHW"}],
            "case_name": "ger_17",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
            "support_expect": True}

case18 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"NHWC", "ori_shape":(100,),    "ori_format":"NHWC"},
                     {"shape":(200,),    "dtype":"float16", "format":"NHWC", "ori_shape":(200,),    "ori_format":"NHWC"},
                     {"shape":(100,200), "dtype":"float16", "format":"NHWC", "ori_shape":(100,200), "ori_format":"NHWC"}],
            "case_name": "ger_18",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
            "support_expect": True}

case19 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"NC1HWC0", "ori_shape":(100,),    "ori_format":"NC1HWC0"},
                     {"shape":(200,),    "dtype":"float16", "format":"NC1HWC0", "ori_shape":(200,),    "ori_format":"NC1HWC0"},
                     {"shape":(100,200), "dtype":"float16", "format":"NC1HWC0", "ori_shape":(100,200), "ori_format":"NC1HWC0"}],
            "case_name": "ger_19",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
            "support_expect": True}

case20 = {"params": [{"shape":(100,),    "dtype":"float16", "format":"ND", "ori_shape":(100,),    "ori_format":"ND"},
                     {"shape":(200,),    "dtype":"float16", "format":"ND", "ori_shape":(200,),    "ori_format":"ND"},
                     {"shape":(100,200), "dtype":"float16", "format":"ND", "ori_shape":(100,200), "ori_format":"ND"}],
            "case_name": "ger_20",
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
            "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend910"], case13)
ut_case.add_case(["Ascend310", "Ascend910"], case14)
ut_case.add_case(["Ascend310", "Ascend910"], case15)
ut_case.add_case(["Ascend310", "Ascend910"], case16)
ut_case.add_case(["Ascend310", "Ascend910"], case17)
ut_case.add_case(["Ascend310", "Ascend910"], case18)
ut_case.add_case(["Ascend310", "Ascend910"], case19)
ut_case.add_case(["Ascend310", "Ascend910"], case20)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)