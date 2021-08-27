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

KMeansCentroids ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("KMeansCentroids", "impl.k_means_centroids", "k_means_centroids")

case1 = {"params": [{"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                    {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                    {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    ],
         "case_name": "KMeansCentroids_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                    {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                    {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    True
                    ],
         "case_name": "KMeansCentroids_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                    {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                    {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                    {"shape": (128, 1), "dtype": "float32", "format": "ND", "ori_shape": (128, 1), "ori_format": "ND"},
                    {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                    {"shape": (128, 1), "dtype": "float32", "format": "ND", "ori_shape": (128, 1), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    ],
         "case_name": "KMeansCentroids_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (17680, 128), "dtype": "float32", "format": "ND", "ori_shape": (17680, 128), "ori_format": "ND"},
                    {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                    {"shape": (17680, 1), "dtype": "float32", "format": "ND", "ori_shape": (17680, 1), "ori_format": "ND"},
                    {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                    {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    ],
         "case_name": "KMeansCentroids_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend710"], case3)
ut_case.add_case(["Ascend910A", "Ascend710"], case4)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend710"])
