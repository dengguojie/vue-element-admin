#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Pooling", None, None)

case1 = {"params": [{"shape": (1,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,4,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (1,2,4,16,16),"ori_format": "FRACTAL_Z"},
                    {"shape": (1,2,4,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,2,4,16,16),"ori_format": "NCHW"},
                    {"shape": (1,2,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "pooling_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,4,16,16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1,2,4,16,16),"ori_format": "FRACTAL_Z"},
                    {"shape": (1,2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (1,2,4,16,16),"ori_format": "NCHW"},
                    {"shape": (1,2,4,16,16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1,2,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "pooling_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"}],
         "case_name": "pooling_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 1,4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 1,4, 16),"ori_format": "ND"}],
         "case_name": "pooling_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "pooling_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (13, 1, 6, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 13, 6, 768),"ori_format": "NCHW"},
                    {"shape": (9, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 3, 3),"ori_format": "NCHW"},
                    None,
                    {"shape": (13, 1, 6, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 6, 768, 13),"ori_format": "NCHW"},
                    (1, 1,), (1, 1), 0, 1],
         "case_name": "pooling_16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)

def test_pooling_get_op_support_info(test_arg):
    from impl.pooling import get_op_support_info
    get_op_support_info({"shape": (13, 1, 2, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 2, 768, 13),"ori_format": "NCHW"},
                        {"shape": (9, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 3, 3),"ori_format": "NCHW"},
                        None,
                        {"shape": (13, 1, 2, 768, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (13, 2, 768, 13),"ori_format": "NCHW"},
                        (1, 1,), (1, 1), 0, 1)

ut_case.add_cust_test_func(test_func=test_pooling_get_op_support_info)

def test_pooling_sd3403(test_arg):
    from impl.pooling import pooling
    from te import platform as cce_conf
    cce_conf.cce_conf.te_set_version("SD3403", core_type="AiCore")
    pooling({"shape": (1, 2, 64, 64, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, 64, 64),"ori_format": "NCHW"},
            None,
            None,
            {"shape": (1, 2, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, 1, 1),"ori_format": "NCHW"},
            (64, 64),
            (1, 1),
            0,
            1,
            (0, 0, 0, 0),
            True,
            0,
            (1, 1, 1, 1),
            "pooling_cce",
            "high_precision")
    cce_conf.cce_conf.te_set_version(test_arg)
ut_case.add_cust_test_func(test_func=test_pooling_sd3403)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
