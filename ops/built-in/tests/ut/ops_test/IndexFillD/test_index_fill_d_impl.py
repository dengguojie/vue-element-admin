# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("IndexFillD")

#pylint: disable=unused-argument
def calc_expect_func(x1, x2, x3, y, dim):
    res = x1['value'] * x2['value'] + x3['value']
    return res

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    },{
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    },
        0],
    "calc_expect_func":calc_expect_func,
    "expect": "success"
})
ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    },{
        "shape": (2, 1, 2),
        "ori_shape": (2, 1, 2),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    },
        10],
    "calc_expect_func":calc_expect_func,
    "expect": RuntimeError
})


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)