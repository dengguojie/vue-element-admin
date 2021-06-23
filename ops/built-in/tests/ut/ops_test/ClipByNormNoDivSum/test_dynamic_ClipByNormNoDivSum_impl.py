#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ClipByNormNoDivSum", "impl.dynamic.clip_by_norm_no_div_sum", "clip_by_norm_no_div_sum")

case1 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]}],
         "case_name": "clip_by_norm_no_div_sum_compute_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range":[(1, 100)]}],
         "case_name": "clip_by_norm_no_div_sum_compute_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
