# -*- coding:utf-8 -*-
import sys
import numpy as np
from op_test_frame.common import op_status
from op_test_frame.ut import BroadcastOpUT
import torch
from op_test_frame.utils import op_param_util

ut_case = BroadcastOpUT("cdist")


#pylint: disable=unused-argument
def calc_expect_func(x1, x2, y, p=2.0):
    x1_value = x1["value"][..., 0, :]
    x2_value = x2["value"][..., 0, :, :]

    a = torch.tensor(x1_value, dtype=torch.float32)
    b = torch.tensor(x2_value, dtype=torch.float32)
    res = torch.cdist(a, b, p)

    if x1["dtype"] == "float16":
        res = res.half()
    print(res.numpy())
    return [res.numpy(), ]


def get_case(dtype1, x1_shape, x2_shape, y_shape, p=2.0, dtype2=None,
             expect=op_status.SUCCESS, case_name=None):
    if dtype2 is None:
        dtype2 = dtype1
    x1_shape_bc = list(x1_shape)
    x1_shape_bc.insert(-1, x2_shape[-2])
    x1_shape_bc = tuple(x1_shape_bc)

    x2_shape_bc = list(x2_shape)
    x2_shape_bc.insert(-2, x1_shape[-2])
    x2_shape_bc = tuple(x2_shape_bc)

    x1_info = (dtype1, x1_shape_bc, "ND")
    x2_info = (dtype2, x2_shape_bc, "ND")
    y_info = (dtype1, y_shape, "ND")

    x1_param = op_param_util.build_op_param(x1_info)
    x2_param = op_param_util.build_op_param(x2_info)
    y_param = op_param_util.build_op_param(y_info)
    if p is not None:
        param_list = [x1_param, x2_param, y_param, p]
    else:
        param_list = [x1_param, x2_param, y_param]
    if expect == op_status.SUCCESS:
        case = {"params": param_list, "case_name": case_name}
    else:
        case = {"params": param_list, "case_name": case_name, "expect": expect}
    return case


ut_case.add_case("all", case=get_case("float32", (2, 3, 5), (2, 4, 5),
                                      (2, 3, 4), -1.0, expect=RuntimeError))
ut_case.add_case("all", case=get_case("float32", (2, 3, 5), (2, 4, 6),
                                      (2, 3, 4), 2.0, expect=RuntimeError))
ut_case.add_case("all", case=get_case("float32", (3, 5), (2, 4, 5),
                                      (2, 3, 4), 2.0, expect=RuntimeError))
ut_case.add_case("all", case=get_case("float32", (2, 3, 5), (2, 4, 5),
                                      (2, 3, 4), 2.0,
                                      dtype2="float16", expect=RuntimeError))


def pr_case(dtype1, x1_shape, x2_shape, y_shape, p=2.0,
            input1_range=(0.1, 1.0), input2_range=None, dtype2=None,
            expect=op_status.SUCCESS, case_name=None):
    np.random.seed(10086)
    if dtype2 is None:
        dtype2 = dtype1
    if input2_range is None:
        input2_range = input1_range
    x1_shape_bc = list(x1_shape)
    x1_shape_bc.insert(-1, x2_shape[-2])
    x1_shape_bc = tuple(x1_shape_bc)

    x1_info = (dtype1, x1_shape_bc, "ND")
    x2_info = (dtype1, x1_shape_bc, "ND")
    y_info = (dtype1, y_shape, "ND")

    x1_param = op_param_util.build_op_param(x1_info)
    x2_param = op_param_util.build_op_param(x2_info)
    y_param = op_param_util.build_op_param(y_info)

    np.random.seed(10086)
    x1_value = np.random.uniform(input1_range[0], input1_range[1],
                                 x1_shape).astype(dtype1)
    x2_value = np.random.uniform(input2_range[0], input2_range[1],
                                 x2_shape).astype(dtype2)
    x1_value = np.expand_dims(x1_value, -2)
    x2_value = np.expand_dims(x2_value, -3)
    x1_value = np.broadcast_to(x1_value, x1_shape_bc)
    x2_value = np.broadcast_to(x2_value, x1_shape_bc)

    for param, value in zip([x1_param, x2_param], [x1_value, x2_value]):
        param["param_type"] = "input"
        param["value"] = value
    y_param["param_type"] = "output"

    if p:
        param_list = [x1_param, x2_param, y_param, p]
    else:
        param_list = [x1_param, x2_param, y_param]

    if expect == op_status.SUCCESS:
        case = {"params": param_list, "case_name": case_name,
                "calc_expect_func": calc_expect_func}
    else:
        case = {"params": param_list, "case_name": case_name,
                "calc_expect_func": calc_expect_func, "expect": expect}
    return case


ut_case.add_precision_case("Ascend910A", case=pr_case("float32",
                                               [2, 5],
                                               [3, 5],
                                               [2, 3],
                                               2.0,
                                               input1_range=[0.01, 1]))

ut_case.add_precision_case("Ascend910A", case=pr_case("float16",
                                                      [3, 3, 5],
                                                      [3, 4, 5],
                                                      [3, 3, 4],
                                                      0.0,
                                                      input1_range=[10, 1000]))

ut_case.add_precision_case("all", case=pr_case("float32",
                                               [2, 5],
                                               [3, 5],
                                               [2, 3],
                                               1.0,
                                               input1_range=[0.01, 1]))


