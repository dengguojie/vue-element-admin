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

SpaceToDepth ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("SpaceToDepth", None, None)

case1 = {"params": [{"shape": (100, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (100, 87, 870, 11),"ori_format": "NHWC"}, #x
                    None,
                    {"shape": (100, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (100, 87, 870, 11),"ori_format": "NHWC"},
                    87, "NHWC",
                    ],
         "case_name": "SpaceToDepth_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2, 2, 2, 70031), "dtype": "uint16", "format": "NHWC", "ori_shape": (2, 2, 2, 70031),"ori_format": "NHWC"}, #x
                    None,
                    {"shape": (2, 2, 2, 70031), "dtype": "uint16", "format": "NHWC", "ori_shape": (2, 2, 2, 70031),"ori_format": "NHWC"},
                    2,"NHWC",
                    ],
         "case_name": "SpaceToDepth_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (2, 20, 20000, 1), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20000, 1),"ori_format": "NHWC"}, #x
                    None,
                    {"shape": (2, 20, 20000, 1), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20000, 1),"ori_format": "NHWC"},
                    20,"NHWC",
                    ],
         "case_name": "SpaceToDepth_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2, 2, 2), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 2, 2),"ori_format": "NHWC"}, #x
                    None,
                    {"shape": (2, 2, 2), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 2, 2),"ori_format": "NHWC"},
                    2,"NHWC",
                    ],
         "case_name": "SpaceToDepth_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (2, 2, 2, 3200), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 2, 2, 3200),"ori_format": "NHWC"}, #x
                    None,
                    {"shape": (2, 2, 2, 3200), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 2, 2, 3200),"ori_format": "NHWC"},
                    0, "NHWC",
                    ],
         "case_name": "SpaceToDepth_5",
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
    exit(0)
