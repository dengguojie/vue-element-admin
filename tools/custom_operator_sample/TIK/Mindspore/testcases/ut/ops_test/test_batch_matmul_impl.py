# -- coding:utf-8 --
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("batch_matmul_impl", op_func_name="batch_matmul")

value1 = np.ones([1, 64, 64]).astype("float32")
value2 = np.ones([1, 64, 64]).astype("float32")


# pylint: disable=unused-argument
def calc_expect_func(x1, x2, y):
    res = np.matmul(x1.get("value"), x2.get("value"))
    return [res, ]


ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 64, 64), "shape": (1, 64, 64),
         "value": value1,
         "param_type": "input"},
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 64, 64), "shape": (1, 64, 64),
         "value": value2,
         "param_type": "input"},
        {"dtype": "float32", "format": "NCHW", "ori_format": "NCHW", "ori_shape": (1, 64, 64), "shape": (1, 64, 64),
         "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})
