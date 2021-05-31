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

BesselI0e ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BesselI0e", "impl.dynamic.bessel_i0e", "bessel_i0e")

case1 = {"params": [{"shape": (-1, 8, 375), "dtype": "float32",
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]}, #x
                    {"shape": (-1, 8, 375), "dtype": "float32",
                     "format": "ND", "ori_shape": (16, 8, 375),
                     "ori_format": "ND", "range": [(15, 16), (8, 8), (375, 375)]},
                    ],
         "case_name": "BesselI0e_1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend310"])
