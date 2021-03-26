# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

from tbe import tvm
import tbe.dsl as dsl


def dsl_broadcast(x, y, dst_shape, kernel_name='dsl_broadcast'):
    input_dtype = x.get("dtype")
    input_shape = x.get("shape")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = dsl.broadcast(data1, dst_shape)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = dsl.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    dsl.build(sch, config)


ut_case = OpUT("broadcast", "broadcast.test_broadcast_impl", "dsl_broadcast")


def calc_expect_func(x, y, dst_shape):
    x_value = x.get("value")

    res = np.broadcast_to(x_value, dst_shape)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (1, 16),
                "dtype": "uint32",
                "param_type": "input"
            },
            {
                "shape": (18, 16),
                "dtype": "uint32",
                "param_type": "output"
            },
            [18,16]
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_broadcast_pre_1"
    })
