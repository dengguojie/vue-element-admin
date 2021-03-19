#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from op_test_frame.ut import OpUT

ut_case = OpUT("LarsV2Update", "impl.dynamic.lars_v2_update", "lars_v2_update")


case1 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 512, 218), "ori_format": "ND", "range": [(1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 512, 218), "ori_format": "ND", "range": [(1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 512, 218), "ori_format": "ND", "range": [(1, None), (1, None), (1, None), (1, None)]},
                    0.001,
                    1e-5,
                    False
                    ],
         "case_name": "lars_v2_update_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 512, 218), "ori_format": "FRACTAL_Z", "range": [(1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 512, 218), "ori_format": "FRACTAL_Z", "range": [(1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 512, 218), "ori_format": "FRACTAL_Z", "range": [(1, None), (1, None), (1, None), (1, None)]},
                    0.001,
                    1e-5,
                    False
                    ],
         "case_name": "lars_v2_update_dynamic_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1, -1, -1, -1), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 512, 218, 10), "ori_format": "NC1HWC0", "range": [(1, None), (1, None), (1, None), (1, None), (1, 16)]},
                    {"shape": (-1, -1, -1, -1, -1), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 512, 218, 10), "ori_format": "NC1HWC0", "range": [(1, None), (1, None), (1, None), (1, None), (1, 16)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1, -1, -1, -1, -1), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 512, 218, 10), "ori_format": "NC1HWC0", "range": [(1, None), (1, None), (1, None), (1, None), (1, 16)]},
                    0.001,
                    1e-5,
                    False
                    ],
         "case_name": "lars_v2_update_dynamic_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (-1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "C1HWNCoC0", "ori_shape": (1, 1, 512, 218, 10, 1), "ori_format": "C1HWNCoC0", "range": [(1, None), (1, None), (1, None), (1, None), (1, 16), (1, 1)]},
                    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "C1HWNCoC0", "ori_shape": (1, 1, 512, 218, 10, 1), "ori_format": "C1HWNCoC0", "range": [(1, None), (1, None), (1, None), (1, None), (1, 16), (1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND", "range": [(1, 1)]},
                    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "C1HWNCoC0", "ori_shape": (1, 1, 512, 218, 10, 1), "ori_format": "C1HWNCoC0", "range": [(1, None), (1, None), (1, None), (1, None), (1, 16), (1, 1)]},
                    0.001,
                    1e-5,
                    False
                    ],
         "case_name": "lars_v2_update_dynamic_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
