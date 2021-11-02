#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

Dynamic SyncBatchNormBackwardElemt ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("SyncBatchNormBackwardElemt", "impl.dynamic.sync_batch_norm_backward_elemt",
               "sync_batch_norm_backward_elemt")

case1 = {"params": [
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         {"shape": (-1, 2, 4, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4, 3), "ori_format": "ND",
          "range": [(3, 3), (2, 2), (4, 4), (3, 3)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         {"shape": (2, -1, 4, 3), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 4, 3), "ori_format": "ND",
          "range": [(2, 2), (1, 1), (4, 4), (3, 3)]},
         ],
         "case_name": "test_dynamic_sync_batch_norm_backward_elemt_case_2",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
