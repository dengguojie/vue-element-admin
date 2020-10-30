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

LrnGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("LrnGrad", None, None)

case1 = {"params": [{"shape": (32, 16, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"}, #x
                    {"shape": (32, 16, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"},
                    {"shape": (32, 16, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"},
                    {"shape": (32, 16, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"},
                    4,1.0,0.00011111111234640703,0.75,
                    ],
         "case_name": "LrnGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (32, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"}, #x
                    {"shape": (32, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"},
                    {"shape": (32, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"},
                    {"shape": (32, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 16, 64, 16),"ori_format": "NHWC"},
                    4,1.0,0.00011111111234640703,0.75,
                    ],
         "case_name": "LrnGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1, 2, 432000, 20), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 432000, 20),"ori_format": "NHWC"}, #x
                    {"shape": (1, 2, 432000, 20), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 432000, 20),"ori_format": "NHWC"},
                    {"shape": (1, 2, 432000, 20), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 432000, 20),"ori_format": "NHWC"},
                    {"shape": (1, 2, 432000, 20), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 432000, 20),"ori_format": "NHWC"},
                    4,1.0,0.00011111111234640703,0.75,
                    ],
         "case_name": "LrnGrad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"}, #x
                    {"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"},
                    {"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"},
                    {"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"},
                    4,1.0,0.00011111111234640703,0.75,
                    ],
         "case_name": "LrnGrad_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"}, #x
                    {"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"},
                    {"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"},
                    {"shape": (1, 1024, 3, 7), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 1024, 3, 7),"ori_format": "NHWC"},
                    -1,1.0,0.00011111111234640703,0.75,
                    ],
         "case_name": "LrnGrad_5",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
