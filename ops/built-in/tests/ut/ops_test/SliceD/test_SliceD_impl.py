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

SliceD ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
ut_case = OpUT("SliceD", None, None)

case1 = {"params": [{"shape": (5, 13, 4), "dtype": "int32", "format": "NCHW", "ori_shape": (5, 13, 4),"ori_format": "NCHW"}, #x
                    {"shape": (2, 12, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 12, 3),"ori_format": "NCHW"},
                    (0, 1, 1), (2, -1, -1),
                    ],
         "case_name": "SliceD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (65, 75), "dtype": "float32", "format": "NCHW", "ori_shape": (65, 75),"ori_format": "NCHW"}, #x
                    {"shape": (15, 33), "dtype": "float32", "format": "NCHW", "ori_shape": (15, 33),"ori_format": "NCHW"},
                    (13, 25), (15, 33),
                    ],
         "case_name": "SliceD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3),"ori_format": "NCHW"}, #x
                    {"shape": (2, 4, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 4, 3, 1),"ori_format": "NCHW"},
                    (0, 0, 0, 0), (2, 4, 3, 1),
                    ],
         "case_name": "SliceD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3),"ori_format": "NCHW"}, #x
                    {"shape": (1, 1, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 3, 1),"ori_format": "NCHW"},
                    (0, 0, 0, 0), (1, 1, 3, 1),
                    ],
         "case_name": "SliceD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (13, 7, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 1, 1),"ori_format": "NCHW"}, #x
                    {"shape": (2, 2, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 2, 1, 1),"ori_format": "NCHW"},
                    (0, 0, 0, 0), (2, 2, 1, 1),
                    ],
         "case_name": "SliceD_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (13, 7, 5, 5), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 5),"ori_format": "NCHW"}, #x
                    {"shape": (1, 1, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 1, 1),"ori_format": "NCHW"},
                    (0, 0, 0, 0), (1, 1, 1, 1),
                    ],
         "case_name": "SliceD_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [{"shape": (2, 70000), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 70000),"ori_format": "NCHW"}, #x
                    {"shape": (2, 69999), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 69999),"ori_format": "NCHW"},
                    (0, 0), (2, 69999),
                    ],
         "case_name": "SliceD_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [{"shape": (7, 200, 600), "dtype": "float16", "format": "NCHW", "ori_shape": (7, 200, 600),"ori_format": "NCHW"}, #x
                    {"shape": (3, 128, 512), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 128, 512),"ori_format": "NCHW"},
                    (1, 1, 1), (3, 128, 512), 
                    ],
         "case_name": "SliceD_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [{"shape": (9, 11, 270000), "dtype": "float16", "format": "NCHW", "ori_shape": (9, 11, 270000),"ori_format": "NCHW"}, #x
                    {"shape": (3, 5, 262144), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 5, 262144),"ori_format": "NCHW"},
                    (3, 4, 5), (3, 5, 262144), 
                    ],
         "case_name": "SliceD_9",
         "expect": "success",
         "support_expect": True}

case10 = {"params": [{"shape": (459999, ), "dtype": "float16", "format": "NCHW", "ori_shape": (459999, ),"ori_format": "NCHW"}, #x
                    {"shape": (458752, ), "dtype": "float16", "format": "NCHW", "ori_shape": (458752, ),"ori_format": "NCHW"},
                    (3, ), (458752, ), 
                    ],
         "case_name": "SliceD_10",
         "expect": "success",
         "support_expect": True}

case11 = {"params": [{"shape": (65536, 31748), "dtype": "int64", "format": "NCHW", "ori_shape": (65536, 31748),"ori_format": "NCHW"}, #x
                    {"shape": (0, 0), "dtype": "int64", "format": "NCHW", "ori_shape": (0, 0),"ori_format": "NCHW"},
                    (0, 0), (65536, 31748), 
                    ],
         "case_name": "SliceD_11",
         "expect": "success",
         "support_expect": True}

case12 = {"params": [{"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16),"ori_format": "NCHW"}, #x
                    {"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16),"ori_format": "NCHW"},
                    (0, 0), (160000, 16), 
                    ],
         "case_name": "SliceD_12",
         "expect": "success",
         "support_expect": True}

case13 = {"params": [{"shape": (15, 64, 568, 568), "dtype": "float16", "format": "NCHW", "ori_shape": (15, 64, 568, 568),"ori_format": "NCHW"}, #x
                    {"shape": (15, 64, 392, 392), "dtype": "int64", "format": "NCHW", "ori_shape": (15, 64, 392, 392),"ori_format": "NCHW"},
                    (0, 0, 0, 0), (15, 64, 392, 392), 
                    ],
         "case_name": "SliceD_13",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)
ut_case.add_case("Ascend910A", case5)
ut_case.add_case("Ascend910A", case6)
ut_case.add_case("Ascend910A", case7)
ut_case.add_case("Ascend910A", case8)
ut_case.add_case("Ascend910A", case9)
ut_case.add_case("Ascend910A", case10)
ut_case.add_case("Ascend910A", case11)
ut_case.add_case("Ascend910A", case12)
ut_case.add_case("Ascend910A", case13)

case_fz = {"params": [{"shape": (1*16*16, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 16, 16),"ori_format": "NCHW"},
                    {"shape": (1*16*16, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 16, 16),"ori_format": "NCHW"},
                    (0, 0, 0 ,0), (16, 16, 16, 16),
                    ],
         "case_name": "SliceD_FRACTAL_Z",
         "expect": "success",
         "support_expect": True}

ut_case.add_case("Ascend910A", case_fz)

def calc_expect_func(x, y, begin, size):
    expect = tf.slice(x['value'], begin, size)
    with tf.Session() as sess:
        expect_data = sess.run(expect)
    return expect_data

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (5, 13, 4), "dtype": "int32", "format": "NCHW", "ori_shape": (5, 13, 4),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2, 12, 3), "dtype": "int32", "format": "NCHW", "ori_shape": (2, 12, 3),"ori_format": "NCHW", "param_type": "output"},
                                              (0, 1, 1), (2, -1, -1)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (13, 7, 5, 5), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 5),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (1, 1, 1, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (1, 1, 1, 1),"ori_format": "NCHW", "param_type": "output"},
                                              (0, 0, 0, 0), (1, 1, 1, 1)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (65, 75), "dtype": "float32", "format": "NCHW", "ori_shape": (65, 75),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (15, 33), "dtype": "float32", "format": "NCHW", "ori_shape": (15, 33),"ori_format": "NCHW", "param_type": "output"},
                                              (13, 25), (15, 33)],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })




if __name__ == "__main__":
    ut_case.run("Ascend910A")