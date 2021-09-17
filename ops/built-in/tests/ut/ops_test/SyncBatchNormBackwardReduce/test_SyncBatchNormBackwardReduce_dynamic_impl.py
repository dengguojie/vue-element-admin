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

Dynamic SyncBatchNormBackwardReduce ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("SyncBatchNormBackwardReduce", "impl.dynamic.sync_batch_norm_backward_reduce", "sync_batch_norm_backward_reduce")

case1 = {"params": [{"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(1,), "ori_format":"ND", "range":[(1, 1)]},
                    {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(1,), "ori_format":"ND", "range":[(1, 1)]},
                    {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(1,), "ori_format":"ND", "range":[(1, 1)]},
                    {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(1,), "ori_format":"ND", "range":[(1, 1)]},
                    {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(1,), "ori_format":"ND", "range":[(1, 1)]},
                    {"shape":(-1,), "dtype":"float16", "format":"ND", "ori_shape":(1,), "ori_format":"ND", "range":[(1, 1)]},
                    ],
         "case_name": "test_dynamic_sync_batch_norm_backward_reduce_case_1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])