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
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
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


def test_op_select_format(test_arg):
    from impl.space_to_depth import op_select_format
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (), "dtype": "", "format": "", "ori_shape": (),"ori_format": ""},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (), "dtype": "", "format": "", "ori_shape": (20,28,16,16),"ori_format": ""},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"})

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_cust_test_func(test_func=test_op_select_format)
def calc_expect_func(x1, x2, y, block_size, data_format):
    input_data = x1['value']
    to_batch = tf.space_to_depth(input_data, block_size, data_format=data_format)
    with tf.Session() as sess:
        out_data = sess.run(to_batch)
    return out_data

precision_case1 = {"params": [{"shape": (1, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 87, 870, 11),"ori_format": "NHWC","param_type":"input"}, #x
                    None,
                    {"shape": (1,1,10,83259), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,10,83259),"ori_format": "NHWC","param_type":"output"},
                    87, "NHWC",
                    ],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (1, 20, 20, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 20, 20, 1),"ori_format": "NHWC","param_type":"input"}, #x
                    None,
                    {"shape": (1, 10, 10, 4), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 10, 10, 4),"ori_format": "NHWC","param_type":"output"},
                    2, "NHWC",
                    ],
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
