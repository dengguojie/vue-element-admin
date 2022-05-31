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

ApplyFtrlD ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyFtrlD", "impl.dynamic.apply_ftrl_d", "apply_ftrl_d")

case1 = {"params": [{"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]}],
         "case_name": "ApplyFtrlD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-2, ), "dtype": "float32", "format": "ND", "ori_shape": (-2, ),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-2, ), "dtype": "float32", "format": "ND", "ori_shape": (-2, ),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-2, ), "dtype": "float32", "format": "ND", "ori_shape": (-2, ),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-2, ), "dtype": "float32", "format": "ND", "ori_shape": (-2, ),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,),"ori_format": "ND", "range":[(1,100),]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]},
                    {"shape": (-1, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 16),"ori_format": "ND", "range":[(1,100), (1,100)]}],
         "case_name": "ApplyFtrlD_2",
         "expect": "success",
         "support_expect": True}


ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
