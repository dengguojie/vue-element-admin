#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DataFormatDimMap", "impl.dynamic.data_format_dim_map", "data_format_dim_map")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 4), "ori_format": "NCHW",
         "range": [(1, 4)]},
        {"shape": (-1,), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 4), "ori_format": "NCHW",
         "range": [(1, 4)]},
    ],
    "case_name": "DataFormatDimMap_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [
        {"shape": (-1, -1, -1), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0",
         "range": [(1, 4), (1, 4), (1, 4)]},
        {"shape": (-1, -1, -1), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0",
         "range": [(1, 4), (1, 4), (1, 4)]},
    ],
    "case_name": "DataFormatDimMap_2",
    "expect": "success",
    "support_expect": True
}
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case2)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend710", "Ascend310"])
