#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from op_test_frame.common import precision_info
import os

ut_case = ElementwiseOpUT("Rsqrt", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (512, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (100, 100))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (11, 33))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 12))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(x, y):
    x_shape = x.get("shape")
    x_value = x.get("value")
    
    session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
    
    input=tf.placeholder(x_value.dtype, x_value.shape)
    output_var =gen_math_ops.rsqrt(input)
    with tf.Session(config=session_config) as session:
        result = session.run(output_var, feed_dict={input: x_value})
    return (result,)

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (2, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 24, 1, 256), "dtype": "float32", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
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

