#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("CeluV2", "impl.dynamic.celu_v2", "celu_v2")

def calc_expect_func(input_x, output_y, alpha=1.0):
    indtype = input_x["dtype"]
    x = input_x["value"].astype(np.float32)
    torch_x = torch.tensor(x)
    res = torch.celu(torch_x, alpha=alpha).numpy()
    if indtype == "float16":
        res = res.astype(indtype)
    return [res, ]

ut_case.add_case("all", {"params": [
    {"shape": (5, -1), "run_shape": (5, 13), "dtype": "float16", "format": "ND", "ori_shape": (5, 13), 
    "ori_format": "ND", "range": [(1, 100)]},
    {"shape": (5, -1), "run_shape": (5, 13), "dtype": "float16", "format": "ND", "ori_shape": (5, 13), 
    "ori_format": "ND", "range": [(1, 100)]}],
    "case_name": "test_dynamic_celu_v2_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True, 
    "op_imply_type": "dynamic"})

ut_case.add_case("Ascend910A", {"params": [
    {"shape": (1, 2), "run_shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND", 
    "range":[(1, 100)], "param_type": "input"},
    {"shape": (1, 2), "run_shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", 
    "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "op_imply_type": "static"})

ut_case.add_case("Ascend910A", {"params": [
    {"shape": (5, 13, 4), "run_shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", 
    "range":[(1, 100)], "param_type": "input"},
    {"shape": (5, 13, 4), "run_shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", 
    "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "op_imply_type": "static"})

ut_case.add_case("Ascend910A", {"params": [
    {"shape": (16, 32), "run_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", 
    "range":[(1, 100)], "param_type": "input"},
    {"shape": (16, 32), "run_shape": (16, 32), "dtype": "float16", "format": "ND", 
    "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "op_imply_type": "static"})

ut_case.add_case("Ascend910A", {"params": [
    {"shape": (16, 2, 32), "run_shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", 
    "range":[(1, 100)], "param_type": "input"},
    {"shape": (16, 2, 32), "run_shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", 
    "param_type": "output"}, 1.0],
    "calc_expect_func": calc_expect_func,
    "op_imply_type": "static"})

case_error = {"params": [{"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                         {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                         1.0, 0, 1.0],
                         "case_name": "case_error",
             "expect": RuntimeError,
             "format_expect": [],
             "support_expect": True}
ut_case.add_case(["Ascend910A"], case_error)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
