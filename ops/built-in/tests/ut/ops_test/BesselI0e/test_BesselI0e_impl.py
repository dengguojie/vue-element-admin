import tensorflow as tf
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np


ut_case = OpUT("BesselI0e", None, None)
case1 = {"params": [{"shape": (1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1),"ori_format": "NHWC"}, #x
                    {"shape": (1, 1), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1),"ori_format": "NHWC"}],
         "case_name": "BesselI0e_1",
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (16, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 32),"ori_format": "NHWC"}, #x
                    {"shape": (16, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 32),"ori_format": "NHWC"}],
         "case_name": "BesselI0e_2",
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (16, 2, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 2, 32),"ori_format": "NHWC"}, #x
                    {"shape": (16, 2, 32), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 2, 32),"ori_format": "NHWC"}],
         "case_name": "BesselI0e_3",
         "expect": "success",
         "support_expect": True}
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)

# ============ auto gen ["Ascend910"] test cases start ===============
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1,))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (512, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (100, 100))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (11, 33))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 12))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================

# def calc_expect_func(x, y):
#     input_Arr = x['value']
#     res = tf.math.bessel_i0e(input_Arr)
#     with tf.Session() as sess:
#         output = sess.run(res)
#     return output

# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (3, 16, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 3, 100, 16))
