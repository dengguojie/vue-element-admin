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

FusedMulApplyMomentumExtern ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("FusedMulApplyMomentumExtern",
               "impl.dynamic.fused_mul_apply_momentum_extern",
               "fused_mul_apply_momentum_extern")

# use nesterov
case1 = {"params": [{"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                     {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                     True],
         "case_name": "fused_mul_apply_momentum_extern_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
        }

ut_case.add_case(["Ascend910A"], case1)

# not use nesterov
case2 = {"params": [{"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                     {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                     False],
         "case_name": "fused_mul_apply_momentum_extern_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
        }

ut_case.add_case(["Ascend910A"], case2)

# accum dtype is float16
case3 = {"params": [{"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (1,), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1,), "range": [(1, 100),]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                     {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND",
                     "ori_format": "ND", "ori_shape": (-1, 16), "range": [(1, 100), (1, 100)]}],
         "case_name": "fused_mul_apply_momentum_extern_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
        }

ut_case.add_case(["Ascend910A"], case3)

if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
