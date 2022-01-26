#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ZerosLike", "impl.dynamic.zeros_like", "zeros_like")

ut_case.add_case(
    ["Ascend910A"],
    {"params": 
        [
            {"ori_shape": [-2], "shape": [-2], "ori_format": "ND", "format": "ND", "dtype": "bool", "range": None},
            {"ori_shape": [-2], "shape": [-2], "ori_format": "ND", "format": "ND", "dtype": "bool", "range": None}
        ],
        "case_name": "zeros_like_000", "expect": "success", "format_expect": [], "support_expect": True
    }
)

if __name__ == '__main__':
    ut_case.run("Ascend910A")