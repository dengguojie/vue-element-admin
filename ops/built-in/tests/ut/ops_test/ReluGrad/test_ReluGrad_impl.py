#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from op_test_frame.common import precision_info
import os

ut_case = BroadcastOpUT("ReluGrad", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 4, 32), (16, 2, 4, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (512, 1024), (512, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (2, 1024), (2, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (4096, 1024), (4096, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (32, 128, 1024), (32, 128, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (100, 100), (100, 100))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (1024, 1024, 256), (1024, 1024, 256))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8", "uint8"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(gradients, features, y):
    gradients_shape = gradients["shape"]
    features_shape = features["shape"]
    gradients_format = gradients["format"]
    features_format = features["format"]
    gradients_data = gradients["value"]
    features_data = features["value"]
        
    session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)

    input1 = tf.placeholder(gradients_data.dtype, gradients_data.shape)
    input2 = tf.placeholder(features_data.dtype, features_data.shape)
    output_var = gen_nn_ops.relu_grad(input1, input2)
    with tf.Session(config=session_config) as session:
        result = session.run(output_var, feed_dict={input1: gradients_data, input2: features_data})
    return (result,)

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "int8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })


# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)


