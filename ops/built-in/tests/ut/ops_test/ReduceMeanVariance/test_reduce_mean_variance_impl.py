#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ReduceMeanVariance", None, None)

case1 = {"params": [{"shape": (4, 224, 2, 224, 160, 16), "dtype": "float16", "format": "NDC1HWC0",
                     "ori_shape": (4, 224, 224, 160, 32), "ori_format": "NDHWC"},
                    {"shape": (4, 1, 2, 1, 1, 16), "dtype": "float16", "format": "NDC1HWC0",
                     "ori_shape": (4, 1, 1, 1, 32), "ori_format": "NDHWC"},
                    {"shape": (4, 1, 2, 1, 1, 16), "dtype": "float16", "format": "NDC1HWC0",
                     "ori_shape": (4, 1, 1, 1, 32), "ori_format": "NDHWC"},
                    [1, 2, 3], True],
         "case_name": "reduce_mean_variance1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910A"], case1)

if __name__ == "__main__":
    ut_case.run("Ascend710")
