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

case6 = {"params": [{"shape": (100, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (100, 87, 870, 11),"ori_format": "NHWC"},
                    {"shape": (100, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (100, 87, 870, 11),"ori_format": "NHWC"},
                    {"shape": (100, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (100, 87, 870, 11),"ori_format": "NHWC"},
                    87, "NHWC",
                    ],
         "case_name": "SpaceToDepth_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (100, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (100, 87, 870, 11),"ori_format": "NHWC"},
                    None,
                    {"shape": (100, 87, 870, 11), "dtype": "float16", "format": "NHWC", "ori_shape": (100, 87, 870, 11),"ori_format": "NHWC"},
                    87, "ND",
                    ],
         "case_name": "SpaceToDepth_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (2, 20, 20000, 2), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20000, 2),"ori_format": "NHWC"},
                    None,
                    {"shape": (2, 20, 20000, 2), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20000, 2),"ori_format": "NHWC"},
                    20,"NHWC",
                    ],
         "case_name": "SpaceToDepth_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (2, 20, 20000, 16), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20000, 16),"ori_format": "NHWC"},
                    None,
                    {"shape": (2, 20, 20000, 16), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20000, 16),"ori_format": "NHWC"},
                    20,"NHWC",
                    ],
         "case_name": "SpaceToDepth_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (2, 20, 20, 16), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20, 16),"ori_format": "NHWC"},
                    None,
                    {"shape": (2, 20, 20, 16), "dtype": "uint8", "format": "NHWC", "ori_shape": (2, 20, 20, 16), "ori_format": "NHWC"},
                    20,"NHWC",
                    ],
         "case_name": "SpaceToDepth_10",
         "expect": "success",
         "support_expect": True}

case11 = {"params": [{"shape": (4, 300, 20, 16), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 300, 20, 16),"ori_format": "NHWC"},
                    None,
                    {"shape": (4, 300, 20, 16), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 300, 20, 16), "ori_format": "NHWC"},
                    20,"NHWC",
                    ],
         "case_name": "SpaceToDepth_11",
         "expect": "success",
         "support_expect": True}

case12 = {"params": [{"shape": (4, 20, 20, 253920), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 20, 253920),"ori_format": "NHWC"},
                    None,
                    {"shape": (4, 20, 20, 253920), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 20, 253920), "ori_format": "NHWC"},
                    10,"NHWC",
                    ],
         "case_name": "SpaceToDepth_12",
         "expect": "success",
         "support_expect": True}

case13 = {"params": [{"shape": (4, 20, 20, 20000), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 20, 20000),"ori_format": "NHWC"},
                    None,
                    {"shape": (4, 20, 20, 20000), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 20, 20000), "ori_format": "NHWC"},
                    10,"NHWC",
                    ],
         "case_name": "SpaceToDepth_13",
         "expect": "success",
         "support_expect": True}

case14 = {"params": [{"shape": (4, 20, 500, 32), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 500, 32),"ori_format": "NHWC"},
                    None,
                    {"shape": (4, 20, 500, 32), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 500, 32), "ori_format": "NHWC"},
                    10,"NHWC",
                    ],
         "case_name": "SpaceToDepth_14",
         "expect": "success",
         "support_expect": True}

case15 = {"params": [{"shape": (4, 20, 500, 32), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 500, 32),"ori_format": "NHWC"},
                    None,
                    {"shape": (4, 20, 500, 32), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 20, 500, 32), "ori_format": "NHWC"},
                    10,"NHWC",
                    ],
         "case_name": "SpaceToDepth_15",
         "expect": "success",
         "support_expect": True}

def test_op_select_format(test_arg):
    from impl.space_to_depth import op_select_format
    op_select_format({"shape": (60,60,60,60), "dtype": "float16", "format": "NHWC", "ori_shape": (60,60,60,60),"ori_format": "NHWC"},
                     {"shape": (), "dtype": "", "format": "", "ori_shape": (),"ori_format": ""},
                     {"shape": (60,10,10,2160), "dtype": "float16", "format": "NHWC", "ori_shape": (60,10,10,2160),"ori_format": "NHWC"},6,"NHWC")
    op_select_format({"shape": (60,60,60,60), "dtype": "float16", "format": "NHWC", "ori_shape": (60,60,60,60),"ori_format": "NHWC"},
                     {"shape": (2160,60,6,6), "dtype": "", "format": "", "ori_shape": (2160,60,6,6),"ori_format": "NCHW"},
                     {"shape": (60,10,10,2160), "dtype": "float16", "format": "NHWC", "ori_shape": (60,10,10,2160),"ori_format": "NHWC"},6,"NHWC")

def test_get_op_support_info(test_arg):
    from impl.space_to_depth import get_op_support_info
    get_op_support_info(
        {
            "shape": (60, 60, 60, 60),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (60, 60, 60, 60),
            "ori_format": "NHWC"
        },
        None, None, None, None,
    )

def test_dynamic_check_supported(test_arg):
    from impl.dynamic.space_to_depth import check_supported
    check_supported(
        {"shape": (60,60,60,60), "dtype": "float16", "format": "NHWC", "ori_shape": (60,60,60,60),"ori_format": "NHWC"},
                     {"shape": (), "dtype": "", "format": "", "ori_shape": (),"ori_format": ""},
                     {"shape": (60,10,10,2160), "dtype": "float16", "format": "NHWC", "ori_shape": (60,10,10,2160),"ori_format": "NHWC"},6,"NHWC"
    )

def test_static_check_supported(test_arg):
    from impl.space_to_depth import check_supported
    check_supported(
        {"shape": (60,60,60,60), "dtype": "float16", "format": "NHWC", "ori_shape": (60,60,60,60),"ori_format": "NHWC"},
                     {"shape": (), "dtype": "", "format": "", "ori_shape": (),"ori_format": ""},
                     {"shape": (60,10,10,2160), "dtype": "float16", "format": "NHWC", "ori_shape": (60,10,10,2160),"ori_format": "NHWC"},6,"NHWC"
    )


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910A","Ascend310"], case6)
ut_case.add_case(["Ascend910A","Ascend310"], case7)
ut_case.add_case(["Ascend910A","Ascend310"], case8)
ut_case.add_case(["Ascend910A","Ascend310"], case9)
ut_case.add_case(["Ascend910A","Ascend310"], case10)
ut_case.add_case(["Ascend910A","Ascend310"], case11)
ut_case.add_case(["Ascend910A","Ascend310"], case12)
ut_case.add_case(["Ascend910A","Ascend310"], case13)
ut_case.add_case(["Ascend910A","Ascend310"], case14)
ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)
ut_case.add_cust_test_func(test_func=test_dynamic_check_supported)
ut_case.add_cust_test_func(test_func=test_static_check_supported)

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
