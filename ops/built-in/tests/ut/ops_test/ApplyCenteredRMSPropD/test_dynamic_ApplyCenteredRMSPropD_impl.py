#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyCenteredRMSPropD", "impl.dynamic.apply_centered_rms_prop_d", "apply_centered_rms_prop_d")

case1 = {"params": [{"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100),]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100),]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100),]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,), "ori_format": "ND", "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16), "ori_format": "ND", "range": [(1, 100), (1, 100)]}],
         "case_name": "apply_centered_rms_prop_d_first",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("Ascend910A", case1)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
