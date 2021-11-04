
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("ReduceMeanVariance", "impl.reduce_mean_variance", "check_supported")

case1 = {"params": [{"shape": (2,260,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    {"shape": (2,2, 2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3, 2, 2,16),"ori_format": "NHWC"},
                    [257, 10], False],
         "case_name": "check_supported_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": False}

case2 = {"params": [{"shape": (4, 224, 224, 160, 32), "dtype": "float32", "format": "NDHWC",
                     "ori_shape": (4, 224, 224, 160, 32),"ori_format": "NDHWC"},
                    {"shape": (4, 1, 1, 1, 32), "dtype": "float32", "format": "NDHWC",
                     "ori_shape": (4, 1, 1, 1, 32),"ori_format": "NDHWC"},
                    {"shape": (4, 1, 1, 1, 32), "dtype": "float32", "format": "NDHWC",
                     "ori_shape": (4, 1, 1, 1, 32),"ori_format": "NDHWC"},
                    [1, 2, 3], True],
         "case_name": "check_supported_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": False}

case3 = {"params": [{"shape": (4, 224, 224, 160, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (4, 224, 224, 160, 32),"ori_format": "NHWC"},
                    {"shape": (4, 1, 1, 1, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (4, 1, 1, 1, 32),"ori_format": "NHWC"},
                    {"shape": (4, 1, 1, 1, 32), "dtype": "float32", "format": "NHWC",
                     "ori_shape": (4, 1, 1, 1, 32),"ori_format": "NHWC"},
                    [1, 2, 3], True],
         "case_name": "check_supported_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": False}

case4 = {"params": [{"shape": (4, 224, 224, 160), "dtype": "float32", "format": "NDHWC",
                     "ori_shape": (4, 224, 224, 160),"ori_format": "NDHWC"},
                    {"shape": (4, 1, 1, 1), "dtype": "float32", "format": "NDHWC",
                     "ori_shape": (4, 1, 1, 1),"ori_format": "NDHWC"},
                    {"shape": (4, 1, 1, 1), "dtype": "float32", "format": "NDHWC",
                     "ori_shape": (4, 1, 1, 1),"ori_format": "NDHWC"},
                    [1, 2, 3], True],
         "case_name": "check_supported_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": False}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
