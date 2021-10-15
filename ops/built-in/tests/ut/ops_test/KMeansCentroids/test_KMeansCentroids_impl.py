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

case_k_128 = {"params": [{"shape": (65536, 128), "dtype": "float32", "format": "ND", "ori_shape": (65536, 128), "ori_format": "ND"},
                         {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                         {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                         {"shape": (65536, 1), "dtype": "float32", "format": "ND", "ori_shape": (65536, 1), "ori_format": "ND"},
                         {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                         {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                         {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                         False
                        ],
              "case_name": "KMeansCentroids_k_128",
              "expect": "success",
              "support_expect": True}

case_k_32 = {"params": [{"shape": (65536, 32), "dtype": "float32", "format": "ND", "ori_shape": (65536, 32), "ori_format": "ND"},
                        {"shape": (256, 32), "dtype": "float32", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"},
                        {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                        {"shape": (65536, 1), "dtype": "float32", "format": "ND", "ori_shape": (65536, 1), "ori_format": "ND"},
                        {"shape": (256, 32), "dtype": "float32", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"},
                        {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                        {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                        False
                       ],
             "case_name": "KMeansCentroids_k_32",
             "expect": "success",
             "support_expect": True}

case_k_128_T = {"params": [{"shape": (65536, 128), "dtype": "float32", "format": "ND", "ori_shape": (65536, 128), "ori_format": "ND"},
                           {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                           {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                           {"shape": (65536, 1), "dtype": "float32", "format": "ND", "ori_shape": (65536, 1), "ori_format": "ND"},
                           {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                           {"shape": (128, 1), "dtype": "float32", "format": "ND", "ori_shape": (128, 1), "ori_format": "ND"},
                           {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                           True
                          ],
                "case_name": "KMeansCentroids_k_128_T",
                "expect": "success",
                "support_expect": True}

case_k_32_T = {"params": [{"shape": (65536, 32), "dtype": "float32", "format": "ND", "ori_shape": (65536, 32), "ori_format": "ND"},
                          {"shape": (256, 32), "dtype": "float32", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"},
                          {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                          {"shape": (65536, 1), "dtype": "float32", "format": "ND", "ori_shape": (65536, 1), "ori_format": "ND"},
                          {"shape": (256, 32), "dtype": "float32", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"},
                          {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                          {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                          True
                         ],
               "case_name": "KMeansCentroids_k_32_T",
               "expect": "success",
               "support_expect": True}

case_k_128_HP = {"params": [{"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                            {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                            {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                            {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                            {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                            {"shape": (128, 1), "dtype": "float32", "format": "ND", "ori_shape": (128, 1), "ori_format": "ND"},
                            {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                            True
                           ],
                 "addition_params":{"impl_mode": "high_precision"},
                 "case_name": "KMeansCentroids_k_128_HP",
                 "expect": "success",
                 "support_expect": True}

case_m_tail = {"params": [{"shape": (17680, 128), "dtype": "float32", "format": "ND", "ori_shape": (17680, 128), "ori_format": "ND"},
                          {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                          {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                          {"shape": (17680, 1), "dtype": "float32", "format": "ND", "ori_shape": (17680, 1), "ori_format": "ND"},
                          {"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                          {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                          {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                          True
                         ],
               "addition_params":{"impl_mode": "high_precision"},
               "case_name": "KMeansCentroids_m_tail",
               "expect": "success",
               "support_expect": True}

case_n_tail = {"params": [{"shape": (256, 128), "dtype": "float32", "format": "ND", "ori_shape": (256, 128), "ori_format": "ND"},
                          {"shape": (5888, 128), "dtype": "float32", "format": "ND", "ori_shape": (5888, 128), "ori_format": "ND"},
                          {"shape": (1, 5888), "dtype": "float32", "format": "ND", "ori_shape": (1, 5888), "ori_format": "ND"},
                          {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                          {"shape": (5888, 128), "dtype": "float32", "format": "ND", "ori_shape": (5888, 128), "ori_format": "ND"},
                          {"shape": (5888, 1), "dtype": "float32", "format": "ND", "ori_shape": (5888, 1), "ori_format": "ND"},
                          {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                          True
                          ],
               "case_name": "KMeansCentroids_n_tail",
               "expect": "success",
               "support_expect": True}

case_not_aligned = {"params": [{"shape": (11, 128), "dtype": "float32", "format": "ND", "ori_shape": (11, 128), "ori_format": "ND"},
                               {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                               {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                               {"shape": (11, 1), "dtype": "float32", "format": "ND", "ori_shape": (11, 1), "ori_format": "ND"},
                               {"shape": (128, 128), "dtype": "float32", "format": "ND", "ori_shape": (128, 128), "ori_format": "ND"},
                               {"shape": (128, 1), "dtype": "float32", "format": "ND", "ori_shape": (128, 1), "ori_format": "ND"},
                               {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                               True
                              ],
                    "case_name": "KMeansCentroids_not_aligned",
                    "expect": RuntimeError,
                    "support_expect": True}

case_k_2048 = {"params": [{"shape": (256, 2048), "dtype": "float32", "format": "ND", "ori_shape": (256, 2048), "ori_format": "ND"},
                          {"shape": (256, 2048), "dtype": "float32", "format": "ND", "ori_shape": (256, 2048), "ori_format": "ND"},
                          {"shape": (1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 256), "ori_format": "ND"},
                          {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                          {"shape": (256, 2048), "dtype": "float32", "format": "ND", "ori_shape": (256, 2048), "ori_format": "ND"},
                          {"shape": (256, 1), "dtype": "float32", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"},
                          {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                          True
                          ],
               "case_name": "KMeansCentroids_k_2048",
               "expect": RuntimeError,
               "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend710"], case_k_128)
ut_case.add_case(["Ascend910A", "Ascend710"], case_k_32)
ut_case.add_case(["Ascend910A", "Ascend710"], case_k_128_T)
ut_case.add_case(["Ascend910A", "Ascend710"], case_k_32_T)
ut_case.add_case(["Ascend910A"], case_k_128_HP)
ut_case.add_case(["Ascend910A"], case_m_tail)
ut_case.add_case(["Ascend910A"], case_n_tail)
ut_case.add_case(["Ascend910A"], case_not_aligned)
ut_case.add_case(["Ascend910A"], case_k_2048)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend710"])
