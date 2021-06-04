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

StridedSliceGradD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("StridedSliceGradD", None, None)

case1 = {"params": [{"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND", "ori_shape": (80, 56, 8, 1023), "ori_format": "ND"},  # x
                    {"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND",
                     "ori_shape": (80, 56, 8, 1023), "ori_format": "ND"},
                    (80, 23, 8, 443), (1, 33, 0, 342),
                    (56, 51, 3, 785), (1, 1, 1, 1), 5, 7, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 512, 1024), "dtype": "float32", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},  # x
                    {"shape": (1, 512, 1024), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 1, 1024), (0, 0, 0), (1, 1, 1024), (1, 1, 1),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND", "ori_shape": (80, 56, 8, 1023), "ori_format": "ND"},  # x
                    {"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND",
                     "ori_shape": (80, 56, 8, 1023), "ori_format": "ND"},
                    (80, 23, 8, 783), (1, 33, 0, -1021),
                    (56, -7, 3, -238), (1, 1, 1, 1), 5, 7, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1, 512, 1024), "dtype": "float32", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},  # x
                    {"shape": (1, 512, 1024), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 1, 1024), (0, 0, 0), (1, 1, 1024), (1, 1, 1),
                    0, 0, 1, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_4",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},  # x
                    {"shape": (1, 512, 1024), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 1, 1024), (0, 0), (1, 1, 1024), (1, 1, 1),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (1, 1, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},  # x
                    {"shape": (1, 512, 1024), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 512, 1024), (0, -101, 0), (1, -100, 1024), (1, 1, 1),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (1, 512, 2048), "dtype": "int8", "format": "ND", "ori_shape": (1, 512, 2048), "ori_format": "ND"},  # x
                    {"shape": (1, 512, 2048), "dtype": "int8", "format": "ND",
                     "ori_shape": (1, 512, 2048), "ori_format": "ND"},
                    (1, 1, 1024), (0, 0, 0), (1, 1, 1024), (1, 1, 2),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_8",
         "expect": RuntimeError,
         "support_expect": True}

case9 = {"params": [{"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND", "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},  # x
                    {"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND",
                     "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                    (12, 512, 512, 3), [], [], [],
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_9",
         "expect": RuntimeError,
         "support_expect": True}

case10 = {"params": [{"shape": (12, 512, 512, 1), "dtype": "float32", "format": "ND", "ori_shape": (12, 512, 512, 1), "ori_format": "ND"},  # x
                     {"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND",
                      "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                     (12, 512, 512, 3), [0, 0, 0, 0], [12, 512, 512, 1], [1, 1, 1, 1],
                     0, 0, 0, 0, 0,
                     ],
          "case_name": "StridedSliceGradD_10",
          "expect": RuntimeError,
          "support_expect": True}


def test_check_supported(test_arg):
    from impl.strided_slice_grad_d import check_supported
    result = check_supported({"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND", "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             {"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND",
                              "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             (12, 512, 512, 3), [], [], [],
                             0, 0, 0, 0, 0)
    assert result[0]

    result = check_supported({"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND", "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             {"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND",
                              "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             (12, 512, 512, 3), [], [], [],
                             0, 0, 0, 2, 4)
    assert result[0]

    result = check_supported({"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND", "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             {"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND",
                              "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             (12, 512, 512, 3), [], [], [],
                             0, 0, 0, 0, 3)
    assert not result[0]

    result = check_supported({"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND", "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             {"shape": (12, 512, 512, 3), "dtype": "float32", "format": "ND",
                              "ori_shape": (12, 512, 512, 3), "ori_format": "ND"},
                             (), [], [], [],
                             0, 0, 0, 0, 0)
    assert not result[0]


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case3)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case4)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case6)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case7)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case8)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case9)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case10)
ut_case.add_cust_test_func(test_func=test_check_supported)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310", "Ascend710"])
    exit(0)
