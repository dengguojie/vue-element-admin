# # -*- coding:utf-8 -*-
import numpy
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("hard_sigmoid_grad")


#pylint: disable=unused-argument
def calc_expect_func(grad, input_x, y):
    grad_value = grad["value"]
    dtypex = numpy.array(input_x["value"]).dtype
    shape = numpy.array(input_x["value"]).shape
    resultalpha = grad_value * 0.16666666
    zero_tensor_x = numpy.zeros(shape, dtype=dtypex)

    lhs = list(map(abs, input_x["value"]))
    lhs = numpy.array(lhs)
    rhs = 3.0 * numpy.ones(shape, dtype=dtypex)
    slhs = resultalpha
    srhs = zero_tensor_x
    lhs_sub_rhs = lhs - rhs
    res = numpy.zeros(shape, dtype=dtypex)
    print("lhs_sub_rhs", lhs_sub_rhs)
    for i in range(lhs_sub_rhs.size):
        if lhs_sub_rhs[i] < 0.0:
            res[i] = resultalpha[i]
    print("res", res)
    return res


ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (99, ), "shape": (99, ),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100, ), "shape": (100, ),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})