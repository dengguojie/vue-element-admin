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

AssignAdd ut case
"""
from op_test_frame.ut import OpUT
import tensorflow as tf
from tensorflow.python.ops import gen_state_ops
from op_test_frame.common import precision_info

ut_case = OpUT("AssignAdd", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},  # x
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},  # h
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    ],
         "case_name": "AssignAdd_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (3, 1), "ori_format": "ND"},  # x
                    {"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (3, 1), "ori_format": "ND"},  # h
                    {"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (5, 8, 16, 16), "ori_format": "ND"},
                    ],
         "case_name": "AssignAdd_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (278982,), "dtype": "int8", "format": "ND", "ori_shape": (278982,), "ori_format": "ND"},
                    # x
                    {"shape": (278982,), "dtype": "int8", "format": "ND", "ori_shape": (278982,), "ori_format": "ND"},
                    # h
                    {"shape": (278982,), "dtype": "int8", "format": "ND", "ori_shape": (278982,), "ori_format": "ND"},
                    ],
         "case_name": "AssignAdd_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case2)
ut_case.add_case(["Ascend910", "Ascend310", "Ascend710"], case3)

def calc_expect_func(ref, value, output):
    my_dtype = ref['dtype']
    shape = value['shape']
    input1=tf.Variable(ref['value'], dtype=ref['dtype'])
    out_pack = gen_state_ops.assign_add(input1, value['value'])
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(out_pack)
    return result

# ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })

# ut_case.add_precision_case("all", {"params": [{"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "output"},
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })

# ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
#                                               {"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "output"},
#                                               ],
#                                    "calc_expect_func": calc_expect_func,
#                                    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#                                    })
