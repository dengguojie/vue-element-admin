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
from op_test_frame.common import precision_info
import numpy as np
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

case4 = {"params": [{"shape": (2,5,3,4,4), "dtype": "float16", "format": "NCHW", "ori_shape": (2,5,3,4,4),"ori_format": "NCHW"}, #x,
                    {"shape": (2,5,3,4,4), "dtype": "bool", "format": "NCHW", "ori_shape": (2,5,3,4,4),"ori_format": "NCHW"},
                    12,
                    ],
         "case_name": "Cast_4",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)

def calc_expect_func(x, y, dst):
    if dst == 12:
        dst = 6
    dst_list = ["float32", "float16", "int8", "int32", "uint8", "uint64", "bool"]
    dst_type = dst_list[dst]
    input_A_Arr = x['value']
    if dst_type == 'uint8':
        outputArr = np.maximum(input_A_Arr,0).astype(dst_type)
    else:
        outputArr = input_A_Arr.astype(dst_type)
        if dst_type == 'bool':
            outputArr = outputArr.astype("int8")
    return outputArr

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              0],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 32), "dtype": "int8", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
                                              1],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 32), "dtype": "int8", "format": "ND", "ori_shape": (16, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "int32", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
                                              3],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "uint8", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
                                              4],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

#ut_case.add_precision_case("all", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 1),"ori_format": "ND", "param_type": "input"},
#                                              {"shape": (16, 32), "dtype": "bool", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
#                                              12],
#                                   "calc_expect_func": calc_expect_func,
#                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                   })
