"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic FakeQuantWithMinMaxVars ut case
"""
from op_test_frame.ut import OpUT
from test_FakeQuantWithMinMaxVars_impl import calc_expect_func
from op_test_frame.common import precision_info

ut_case = OpUT("FakeQuantWithMinMaxVars",
               "impl.dynamic.fake_quant_with_min_max_vars",
               "fake_quant_with_min_max_vars")


def gen_dynamic_case(shape_x, range_x, dtype_x, ori_shape_x,
                     shape_min, range_min, dtype_min, ori_shape_min,
                     shape_max, range_max, dtype_max, ori_shape_max,
                     num_bits, narrow_range,
                     kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_x,
                        "range": range_x, "format": "ND",
                        "ori_shape": ori_shape_x, "ori_format": "ND"},  # x
                       {"shape": shape_min, "dtype": dtype_min,
                        "range": range_min, "format": "ND",
                        "ori_shape": ori_shape_min, "ori_format": "ND"},  # min
                       {"shape": shape_max, "dtype": dtype_max,
                        "range": range_max, "format": "ND",
                        "ori_shape": ori_shape_max, "ori_format": "ND"},  # max
                       {"shape": shape_x, "dtype": dtype_x,
                        "range": range_x, "format": "ND",
                        "ori_shape": ori_shape_x, "ori_format": "ND"},  # y
                       num_bits,
                       narrow_range
                       ],
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["all"],
                 gen_dynamic_case([3, -1, -1], [[3, 3], [1, None], [1, None]], "float32", [3, -1, -1],
                                  [1], [[1, 1]], "float32", [1],
                                  [1], [[1, 1]], "float32", [1],
                                  8, False, "FakeQuantWithMinMaxVars_success_1", "success"))
ut_case.add_case(["all"],
                 gen_dynamic_case([16, -1, -1, 7, 42, -1, 3], [[16, 16], [1, None], [1, None], [7, 7], [42, 42], [1, None], [3, 3]], "float32", [16, -1, -1, 7, 42, -1, 3],
                                  [1], [[1, 1]], "float32", [1],
                                  [1], [[1, 1]], "float32", [1],
                                  10, True, "FakeQuantWithMinMaxVars_success_2", "success"))

ut_case.add_case(["all"],
                 gen_dynamic_case([-1, -1, -1, 3], [[1, None], [1, None], [1, None], [3, 3]], "float32", [-1, -1, -1, 3],
                                  [1], [[1, 1]], "float16", [1],
                                  [1], [[1, 1]], "int8", [1],
                                  8, False, "FakeQuantWithMinMaxVars_failed_1", "failed"))
ut_case.add_case(["all"],
                 gen_dynamic_case([3, -1, -1], [[3, 3], [1, None], [1, None]], "float32", [3, -1, -1],
                                  [1, 1], [[1, 1], [1, 1]], "float32", [1, 1],
                                  [1], [[1, 1]], "float32", [1],
                                  8, False, "FakeQuantWithMinMaxVars_failed_2", "failed"))
ut_case.add_case(["all"],
                 gen_dynamic_case([3, -1, -1], [[3, 3], [1, None], [1, None]], "float32", [3, -1, -1],
                                  [1], [[1, 1]], "float32", [1],
                                  [1], [[1, 1]], "float32", [1],
                                  19, False, "FakeQuantWithMinMaxVars_failed_3", "failed"))
