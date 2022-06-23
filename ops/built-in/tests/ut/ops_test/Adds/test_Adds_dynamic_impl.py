#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("Adds", "impl.dynamic.adds", "adds")

case1 = {
    "params": [
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
        {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (2, 4),"ori_format": "ND","range":[(1, 100)]},
        1.0
    ],
    "case_name": "Adds_1",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case("Ascend910A", case1)

def calc_expect_func(input_x, output_arr, hg):
    import numpy as np
    input_y = np.ones((1,)).astype(input_x["dtype"])
    input_y = input_y * hg
    output_arr = input_x.get("value") + input_y
    return output_arr

ut_case.add_precision_case("Ascend910A", {
    "params": [
        {"shape": (-1, -1), "dtype": "float16", "format": "ND", "range": [(1, 200), (1, 200)],
         "ori_shape": (125, 125), "ori_format": "ND", "param_type": "input", "run_shape":(125, 125)},
        {"shape": (-1, -1), "dtype": "float16", "format": "ND", "range": [(1, 200), (1, 200)],
         "ori_shape": (125, 125),"ori_format": "ND", "param_type": "output", "run_shape":(125, 125)},
        3.0
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run("Ascend910A", simulator_mode="pv",
                simulator_lib_path="/usr/local/Ascend/latest/toolkit/tools/simulator")
