#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("EluGrad", None, None)

def gen_elu_grad_case(shape_gradient, shape_activation, dtype, case_name_val):
    return {"params": [{"shape": shape_gradient, "dtype": dtype, "ori_shape": shape_gradient, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_activation, "dtype": dtype, "ori_shape": shape_activation, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_gradient, "dtype": dtype, "ori_shape": shape_gradient, "ori_format": "ND", "format": "ND"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_elu_grad_case((32, 112, 15, 112),(32, 112, 15, 112),"float32", "elu_grad_1")
case2 = gen_elu_grad_case((32, 112, 15),(32, 112, 15),"float32", "elu_grad_2")
case3 = gen_elu_grad_case((32, 112),(32, 112),"float32", "elu_grad_3")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)