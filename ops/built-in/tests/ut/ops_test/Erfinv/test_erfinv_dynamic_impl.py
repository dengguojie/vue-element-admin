#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Erfinv", "impl.dynamic.erfinv", "erfinv")

ut_case.add_case(["all"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (-1, 2),
                "param_type": "input", "range": [(1, 100), (2, 2)]},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (-1, 2),
                "param_type": "output", "range": [(1, 100), (2, 2)]}],
    "case_name": "erfinv_success_dynamic_fp32_2d",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
})

ut_case.add_case(["all"], {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (-2,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (-2,),
                "param_type": "output"}],
    "case_name": "erfinv_success_dynamic_fp16_unknowndimensions",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
})

ut_case.add_case(["all"], {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 16, 32), "shape": (-1, -1),
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 16, 32), "shape": (-2,),
                "param_type": "output"}],
    "case_name": "erfinv_failed_unmatched_shape",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
})

ut_case.add_case(["all"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (4, 2),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 2), "shape": (4, 2),
                "param_type": "output"}],
    "case_name": "erfinv_success_static_fp32_2d",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
})

if __name__ == "__main__":
    ut_case.run(["Ascend310", "Ascend910A"])
