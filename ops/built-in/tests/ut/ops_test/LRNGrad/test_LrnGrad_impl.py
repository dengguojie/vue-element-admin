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
from op_test_frame.common import precision_info
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

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
ut_case.add_case(["Ascend910","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

def calc_expect_func(grads, x, y, z):
    depth_radius = 5
    bias=1.0
    alpha=1.0
    beta=0.5
    out_grads_val = grads['value']
    in_image_val = x['value']
    out_image_val = y['value']
    shape = x['shape']
    out_grads_val = np.transpose(out_grads_val, [0, 2, 3, 1])
    in_image_val = np.transpose(in_image_val, [0, 2, 3, 1])
    out_image_val = np.transpose(out_image_val, [0, 2, 3, 1])

    tf_shape = (shape[0], shape[2], shape[3], shape[1])
    in_image = tf.placeholder("float32", shape=tf_shape)
    out_image = tf.placeholder("float32", shape=tf_shape)
    out_grads = tf.placeholder("float32", shape=tf_shape)
    out = gen_nn_ops.lrn_grad(out_grads, in_image, out_image, depth_radius,
                              bias, alpha, beta)
    result = tf.Session().run(out, feed_dict={out_grads:out_grads_val,
                                              in_image:in_image_val,
                                              out_image:out_image_val})
    result = np.transpose(result, [0,3,1,2])
    return result
precision_case1 = {"params": [{"shape": (1, 3, 16, 2), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 3, 16, 2),"ori_format": "NHWC","param_type": "input"}, #x
                              {"shape": (1, 3, 16, 2), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 3, 16, 2),"ori_format": "NHWC","param_type": "input"},
                              {"shape": (1, 3, 16, 2), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 3, 16, 2),"ori_format": "NHWC","param_type": "input"},
                              {"shape": (1, 3, 16, 2), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 3, 16, 2),"ori_format": "NHWC","param_type": "output"}
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case2 = {"params": [{"shape": (1, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 64, 16),"ori_format": "NHWC","param_type": "input"}, #x
                              {"shape": (1, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 64, 16),"ori_format": "NHWC","param_type": "input"},
                              {"shape": (1, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 64, 16),"ori_format": "NHWC","param_type": "input"},
                              {"shape": (1, 16, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 64, 16),"ori_format": "NHWC","param_type": "output"}
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}
precision_case3 = {"params": [{"shape": (1, 16, 65, 17), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 65, 17),"ori_format": "NHWC","param_type": "input"}, #x
                              {"shape": (1, 16, 65, 17), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 65, 17),"ori_format": "NHWC","param_type": "input"},
                              {"shape": (1, 16, 65, 17), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 65, 17),"ori_format": "NHWC","param_type": "input"},
                              {"shape": (1, 16, 65, 17), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 16, 65, 17),"ori_format": "NHWC","param_type": "output"}
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)}

ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)

