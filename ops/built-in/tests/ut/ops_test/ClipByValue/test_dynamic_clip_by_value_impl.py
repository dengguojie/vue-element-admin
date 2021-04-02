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

ApproximateEqual ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("ClipByValue", "impl.dynamic.clip_by_value", "clip_by_value")

case1 = {"params": [{"shape": (-1, -1), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 1),
                     "ori_format": "NHWC","range":[(1, 100), (1, 50)]},
                    {"shape": (-1,), "dtype": "float16", "format": "NHWC", "ori_shape": (1,),
                     "ori_format": "NHWC","range":[(1, 5)]},
                    {"shape": (-1,), "dtype": "float16", "format": "NHWC", "ori_shape": (1,),
                     "ori_format": "NHWC","range":[(1, 5)]},
                    {"shape": (-1, -1), "dtype": "float16", "format": "NHWC", "ori_shape": (5, 1),
                     "ori_format": "NHWC","range":[(1, 100), (1, 50)]}],
         "case_name": "broadcast_to_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
