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

DepthToSpace ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
import tensorflow as tf
ut_case = OpUT("DepthToSpace", None, None)

case1 = {"params": [{"shape": (1,1,1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,1,1,4),"ori_format": "ND"}, #x
                    {"shape": (1,1,1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,1,1,4),"ori_format": "ND"},
                    2, "NHWC",
                    ],
         "case_name": "DepthToSpace_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2,2,3,40000), "dtype": "float32", "format": "ND", "ori_shape": (2,2,3,40000),"ori_format": "ND"}, #x
                    {"shape": (2,40,60,100), "dtype": "float32", "format": "ND", "ori_shape": (2,40,60,100),"ori_format": "ND"},
                    20, "NHWC",
                    ],
        "case_name": "DepthToSpace_2",
        "expect": "success",
        "support_expect": True}

case3 = {"params": [{"shape": (1,1,3,90000), "dtype": "int16", "format": "ND", "ori_shape": (1,1,3,90000),"ori_format": "ND"}, #x
                    {"shape": (1,2,6,22500), "dtype": "int16", "format": "ND", "ori_shape": (1,2,6,22500),"ori_format": "ND"},
                    2, "NHWC",
                    ],
        "case_name": "DepthToSpace_3",
        "expect": "success",
        "support_expect": True}

case4 = {"params": [{"shape": (68,2,3,9), "dtype": "uint8", "format": "ND", "ori_shape": (68,2,3,9),"ori_format": "ND"}, #x
                    {"shape": (68,6,9,1), "dtype": "uint8", "format": "ND", "ori_shape": (68,6,9,1),"ori_format": "ND"},
                    3, "NHWC",
                    ],
        "case_name": "DepthToSpace_4",
        "expect": "success",
        "support_expect": True}

case5 = {"params": [{"shape": (3,2,3,909), "dtype": "int8", "format": "ND", "ori_shape": (3,2,3,909),"ori_format": "ND"}, #x
                    {"shape": (3,6,9,101), "dtype": "int8", "format": "ND", "ori_shape": (3,6,9,101),"ori_format": "ND"},
                    3, "NHWC",
                    ],
        "case_name": "DepthToSpace_5",
        "expect": "success",
        "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case5)

def calc_expect_func(x, y, block_size, data_format):
    out = tf.depth_to_space(x['value'], block_size, data_format=data_format)
    with tf.Session() as sess:
        res = sess.run(out)
    return res

#ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (2,64,80,64), "dtype": "float16", "format": "ND", "ori_shape": (2,64,80,64),"ori_format": "ND", "param_type": "input"},
#                                              {"shape": (2,128,160,16), "dtype": "float16", "format": "ND", "ori_shape": (2,128,160,16),"ori_format": "ND", "param_type": "output"},
#                                              2, "NHWC"],
#                                   "calc_expect_func": calc_expect_func,
#                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                   })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (2,2,111,9), "dtype": "float16", "format": "ND", "ori_shape": (2,2,111,9),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2,6,333,1), "dtype": "float16", "format": "ND", "ori_shape": (2,6,333,1),"ori_format": "ND", "param_type": "output"},
                                              3, "NHWC"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1,1,1,1440), "dtype": "float16", "format": "ND", "ori_shape": (1,1,1,1440),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1,4,4,90), "dtype": "float16", "format": "ND", "ori_shape": (1,4,4,90),"ori_format": "ND", "param_type": "output"},
                                              4, "NHWC"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend710"])
    exit(0)
