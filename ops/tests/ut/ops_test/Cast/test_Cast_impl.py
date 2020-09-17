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

Cast ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("Cast", None, None)

case1 = {"params": [{"shape": (2,3,4), "dtype": "int32", "format": "NHWC", "ori_shape": (2,3,4),"ori_format": "NHWC"}, #x
                    {"shape": (2,3,4), "dtype": "int32", "format": "NHWC", "ori_shape":(2,3,4),"ori_format": "NHWC"},
                    2,
                    ],
         "case_name": "Cast_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,),"ori_format": "NHWC"}, #x
                    {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,),"ori_format": "NHWC"},
                    1,
                    ],
         "case_name": "Cast_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2,5,3,4,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,5,3,4,4),"ori_format": "NCHW"}, #x,
                    {"shape": (2,5,3,4,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,5,3,4,4),"ori_format": "NCHW"},
                    4,
                    ],
         "case_name": "Cast_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)


if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
