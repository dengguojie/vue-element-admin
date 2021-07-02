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

ApplyProximalAdagradD ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import tensorflow as tf
from tensorflow.python.training import gen_training_ops

ut_case = OpUT("ApplyProximalAdagradD", None, None)

case1 = {"params": [{"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #x
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #c
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #w
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},  #b
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #mask 
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #ft
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #ot            
                    ],
         "case_name": "ApplyProximalAdagradD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #x
                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #h
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #c
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}, #w
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},  #b
                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #mask 
                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #ft
                    {"shape": (2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #ot            
                    ],
         "case_name": "ApplyProximalAdagradD_2",
         "expect": "success",
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
def test_op_select_format(test_arg):
    from impl.apply_proximal_adagrad_d import op_select_format
    op_select_format({"shape": (16,16,5,5), "ori_shape": (16,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,16,5,5), "ori_shape": (16,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,16,5,5), "ori_shape": (16,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,16,5,5), "ori_shape": (16,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,16,5,5), "ori_shape": (16,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             False)
    op_select_format({"shape": (1,16,5,5), "ori_shape": (1,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,16,5,5), "ori_shape": (1,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,16,5,5), "ori_shape": (1,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,16,5,5), "ori_shape": (1,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,16,5,5), "ori_shape": (1,16,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             False)
    op_select_format({"shape": (16,), "ori_shape": (16,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,), "ori_shape": (16,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,), "ori_shape": (16,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,), "ori_shape": (16,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (16,), "ori_shape": (16,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             False)
    op_select_format({"shape": (1,2,5,5), "ori_shape": (1,2,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,2,5,5), "ori_shape": (1,2,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,), "ori_shape": (1,),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,2,5,5), "ori_shape": (1,2,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,2,5,5), "ori_shape": (1,2,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             {"shape": (1,2,5,5), "ori_shape": (1,2,5,5),"dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                             False)
ut_case.add_cust_test_func(test_func=test_op_select_format)
#precision cases
def _gen_outputs(input_var, input_accum, input_lr, input_l1, input_l2,
                 input_grad):
    input_var = tf.Variable(input_var, name="input_var")
    input_accum = tf.Variable(input_accum, name="input_accum")
    input_lr = tf.reshape(input_lr, [])
    input_l1 = tf.reshape(input_l1, [])
    input_l2 = tf.reshape(input_l2, [])
    input_grad = tf.reshape(input_grad, [])
    res = gen_training_ops.apply_proximal_adagrad(input_var, input_accum, input_lr, input_l1, input_l2, input_grad, use_locking=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_data = sess.run(res)
        output_var = sess.run(input_var)
        output_accum = sess.run(input_accum)
    return output_var, output_accum, output_data

def calc_expect_func(x1, x2, x3, x4, x5, x6, y1, y2):
    res1, res2, res3 = _gen_outputs(x1['value'], x2['value'], x3['value'], x4['value'],
                                    x5['value'], x6['value'])
    return res1, res2

precision_case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"input"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"},
                              {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"}
                              ],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

# ut_case.add_precision_case("Ascend910", precision_case1)
