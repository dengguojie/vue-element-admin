# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.utils import calc_shape_size
import tensorflow as tf

ut_case = OpUT("Mask2Argmax", "impl.mask2_argmax", "mask2_argmax")
ut_case2 = OpUT("MaxPoolWithArgmax", "impl.max_pool_with_argmax", "max_pool_with_argmax")

session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)


def calc_expect_func(x, y1, y2, ksize, strides, paddings):
    x1_shape = x["shape"]
    x1_data = x["value"][:calc_shape_size(x1_shape)].reshape(x1_shape)
    input = tf.placeholder(x1_data.dtype, x1_data.shape)
    output_var = tf.nn.max_pool_with_argmax(input, ksize, strides, paddings)
    with tf.Session(config=session_config) as session:
        result0, result1 = session.run(output_var, feed_dict={input: x1_data})
    return result0, result1


case1 = {"params": [{"dtype": "float16", "format": "NHWC", "ori_format": "NHWC", "ori_shape": (4, 64, 64, 16),
                     "shape": (4, 64, 64, 16),
                     "param_type": "input"},
                    {"dtype": "uint16", "format": "NHWC", "ori_format": "NHWC", "ori_shape": (4, 32, 32, 16),
                     "shape": (4, 32, 32, 16),
                     "param_type": "input"},
                    {"dtype": "float32", "format": "NHWC", "ori_format": "NHWC", "ori_shape": (4, 32, 32, 16),
                     "shape": (4, 32, 32, 16),
                     "param_type": "output"}, [1, 3, 3, 1], [1, 2, 2, 1], "SAME", [4, 64, 64, 16]],
         "case_name": "mask2argmax_case1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("all", case1)

ut_case2.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "NHWC", "ori_format": "NHWC", "ori_shape": (4, 64, 64, 16),
                "shape": (4, 64, 64, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "NHWC", "ori_format": "NHWC", "ori_shape": (4, 32, 32, 16),
                "shape": (4, 32, 32, 16),
                "param_type": "output"},
               {"dtype": "float32", "format": "NHWC", "ori_format": "NHWC", "ori_shape": (4, 32, 32, 16),
                "shape": (4, 32, 32, 16),
                "param_type": "output"}, [1, 3, 3, 1], [1, 2, 2, 1], "SAME"
               ],
    "calc_expect_func": calc_expect_func
})
