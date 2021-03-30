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

AvgPool1D ut case
"""
from op_test_frame.ut import OpUT
import tbe

ut_case = OpUT("AvgPool1DD", "impl.dynamic.avg_pool_1d", "avg_pool_1d")

ut_case.add_case(["Ascend910A"], {"params": [
    {'shape': (-1, -1, -1, -1, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (-1, -1, -1, -1, 16),"range":[(1,1),(1,1),(1,1),(1,1),(1,1)]},
    {'shape': (-1, -1, -1, -1, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (-1, -1, -1, -1, 16),"range":[(1,1),(1,1),(1,1),(1,1),(1,1)]},
    {'shape': (-1, -1, -1, -1, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (-1, -1, -1, -1, 16),"range":[(1,1),(1,1),(1,1),(1,1),(1,1)]},
    4,
    2,
    [0, 0],
    False,
    False],
    "expect": "success",
    "support_expect": True})

ut_case.add_case(["Ascend910A"], {"params": [
    {'shape': (-1, -1, -1, -1, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (-1, -1, -1, -1, 16),"range":[(1,1),(1,1),(1,1),(1,1),(1,1)]},
    {'shape': (-1, -1, -1, -1, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (-1, -1, -1, -1, 16),"range":[(1,1),(1,1),(1,1),(1,1),(1,1)]},
    {'shape': (-1, -1, -1, -1, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (-1, -1, -1, -1, 16),"range":[(1,1),(1,1),(1,1),(1,1),(1,1)]},
    1,
    2,
    [0, 0],
    False,
    False],
    "expect": "success",
    "support_expect": True})

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
