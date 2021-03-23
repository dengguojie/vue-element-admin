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

ApplyProximalGradientDescent ut case
"""
import tbe

from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyProximalGradientDescent",
               "impl.dynamic.apply_proximal_gradient_descent", "apply_proximal_gradient_descent")

case1 = {"params": [{"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    ],
         "case_name": "apply_proximal_gradient_descent_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    {"shape": (-1, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 16),
                     "ori_format": "ND", "range": [(1, 100), (1, 100)]},
                    ],
         "case_name": "apply_proximal_gradient_descent_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (-1,),
                     "ori_format": "ND", "range": [(1, 100)]},
                    ],
         "case_name": "apply_proximal_gradient_descent_2",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case3)

# pylint: disable=consider-using-sys-exit
if __name__ == "__main__":
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
