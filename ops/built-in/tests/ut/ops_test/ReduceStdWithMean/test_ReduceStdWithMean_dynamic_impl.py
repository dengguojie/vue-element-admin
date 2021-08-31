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

Dot ut case
"""
# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceStdWithMean", "impl.dynamic.reduce_std_with_mean", "reduce_std_with_mean")

case1 = {"params": [{"shape":(-1, -1), "dtype":"float16", "format":"ND", "ori_shape":(3, 4), "ori_format":"ND", "range":[(3, 3), (4, 4)]},
                    {"shape":(-1, -1), "dtype":"float16", "format":"ND", "ori_shape":(3, 1), "ori_format":"ND", "range":[(3, 3), (1, 1)]},
                    {"shape":(-1, -1), "dtype":"float16", "format":"ND", "ori_shape":(3, 1), "ori_format":"ND", "range":[(3, 3), (1, 1)]},
                    [1, ],
                    True,
                    True
                    ],
         "case_name": "test_dynamic_reduce_std_mean_case_1",
         "expect": "success",
         "support_expect": True}


case2 = {"params": [{"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 5), "ori_format":"ND", "range":[(3, 3), (4, 4), (5, 5)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    [2, ],
                    True,
                    True
                    ],
         "case_name": "test_dynamic_reduce_std_mean_case_2",
         "expect": "success",
         "support_expect": True}


case3 = {"params": [{"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 5), "ori_format":"ND", "range":[(3, 3), (4, 4), (5, 5)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    [2, ],
                    False,
                    True
                    ],
         "case_name": "test_dynamic_reduce_std_mean_case_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 5), "ori_format":"ND", "range":[(3, 3), (4, 4), (5, 5)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    [2, ],
                    False,
                    True,
                    True,
                    0.001
                    ],
         "case_name": "test_dynamic_reduce_std_mean_case_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 5), "ori_format":"ND", "range":[(3, 3), (4, 4), (5, 5)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    {"shape":(-1, -1, -1), "dtype":"float32", "format":"ND", "ori_shape":(3, 4, 1), "ori_format":"ND", "range":[(3, 3), (4, 4), (1, 1)]},
                    [2, ],
                    True,
                    True,
                    True,
                    0.001
                    ],
         "case_name": "test_dynamic_reduce_std_mean_case_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape":(-1, -1), "dtype":"float16", "format":"ND", "ori_shape":(3, 4), "ori_format":"ND", "range":[(3, 3), (4, 4)]},
                    {"shape":(-1, -1), "dtype":"float16", "format":"ND", "ori_shape":(3, 1), "ori_format":"ND", "range":[(3, 3), (1, 1)]},
                    {"shape":(-1, -1), "dtype":"float16", "format":"ND", "ori_shape":(3, 1), "ori_format":"ND", "range":[(3, 3), (1, 1)]},
                    [1, ],
                    True,
                    True,
                    True,
                    0.001
                    ],
         "case_name": "test_dynamic_reduce_std_mean_case_6",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)