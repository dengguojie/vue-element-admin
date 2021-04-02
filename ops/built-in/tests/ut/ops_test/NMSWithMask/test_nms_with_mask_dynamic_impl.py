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

NMSWithMask dynamic ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("NMSWithMask", "impl.dynamic.nms_with_mask", "nms_with_mask")

ut_case.add_case(["Ascend910A"], {"params": [
    {'shape': (-1, 8), 'dtype': "float16", "format": "ND", "ori_format": "ND",
     "ori_shape": (-1, 8),"range":[(16,16),(8,8)]},
    {'shape': (-1, 5), 'dtype': "float16", "format": "ND", "ori_format": "ND",
     "ori_shape": (-1, 5),"range":[(16,16),(5,5)]},
    {'shape': (-1, ), 'dtype': "int32", "format": "ND", "ori_format": "ND",
     "ori_shape": (-1, ),"range":[(16,16)]},
     {'shape': (-1, ), 'dtype': "uint8", "format": "ND", "ori_format": "ND",
     "ori_shape": (-1, ),"range":[(16,16)]},
    0.7],
    "expect": "success",
    "support_expect": True,
    "case_name": "test_nms_with_mask_dynamic_001"})


if __name__ == '__main__':
    ut_case.run("Ascend910A")
