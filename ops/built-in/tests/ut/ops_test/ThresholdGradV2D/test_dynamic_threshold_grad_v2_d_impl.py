#/user/bin/env python
# -*- coding: UTF-8 -*-
'''
ut_test_threshold_grad_v2_d
'''
from op_test_frame.ut import OpUT

ut_case = OpUT("ThresholdGradV2D", "impl.dynamic.threshold_grad_v2_d", "threshold_grad_v2_d")

case1 = {
        "params": [
            {"shape": (2,), "ori_shape": (2,), "range": [(2, 2)],
             "ori_format": "ND", "format": "ND", "dtype": "float16"},
            {"shape": (2,), "ori_shape": (2,), "range": [(2, 2)],
             "ori_format": "ND", "format": "ND", "dtype": "float16"},
            {"shape": (1,), "ori_shape": (1,), "range": [(1, 100)],
             "ori_format": "ND", "format": "ND", "dtype": "float16"},
            {"shape": (2,), "ori_shape": (2,), "range": [(2, 3)],
             "ori_format": "ND", "format": "ND", "dtype": "float16"},
        ],
        "case_name": "threshold_grad_v2_d_dynamic_1",
        "except": "success",
        "format_except": [],
        "support_except": True
}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
