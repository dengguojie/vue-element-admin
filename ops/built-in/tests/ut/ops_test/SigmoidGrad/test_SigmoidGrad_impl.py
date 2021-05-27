"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

SigmoidGrad ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("SigmoidGrad", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (10, 13), (10, 11, 12), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (10, 13, 1), (10, 13, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================
def calc_expect_func(input_x, output_y, output_z):
    res_mid = input_x["value"] * (1 - input_x["value"]).astype("float32")
    res = (res_mid * output_y["value"]).astype(output_z["dtype"])
    return res

#ut_case.add_precision_case("all", {
#    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
#                "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
#               {"dtype": "float32", "format": "ND", "ori_format": "ND",
#                "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
#               {"dtype": "float32", "format": "ND", "ori_format": "ND",
#                "ori_shape": (1, ), "shape": (1, ), "param_type": "output"}],
#    "calc_expect_func": calc_expect_func,
#    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND",
                "ori_shape": (11, 33), "shape": (11, 33), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 32), "shape": (16, 2, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 32), "shape": (16, 2, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 32), "shape": (16, 2, 32), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND",
                "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
