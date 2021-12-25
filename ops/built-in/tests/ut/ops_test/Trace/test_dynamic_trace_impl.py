import sys
import numpy
from op_test_frame.ut import OpUT

ut_case = OpUT("Trace", "impl.dynamic.trace", "trace")

def calc_expect_func(input_x):
    res = numpy.zeros([1,], input_x["dtype"])
    res[0] = numpy.trace(input_x["value"])
    return res

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
        "param_type": "input"},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_size_2",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (15, 15), "shape": (15, 15),
        "param_type": "input"},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_size_15",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (19, 16), "shape": (19, 16),
        "param_type": "input"},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_size_16",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (19, 16), "shape": (19, 16),
        "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_size_float32_16",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1500, 1089), "shape": (1500, 1089),
        "param_type": "input"},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_size_1089",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4096, 4096), "shape": (4096, 4096),
        "param_type": "input"},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_size_4096",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (10240, 15000), "shape": (10240, 15000),
        "param_type": "input"},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_size_10240",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, -1), "shape": (1, -1),
        "param_type": "input", "range": ((1, 1), (10000, 10000))},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_dynamic_size_10000",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, 1024), "shape": (-1, 1024),
        "param_type": "input", "range": ((1, 20), (1024, 1024))},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_dynamic_size_1024",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1), "shape": (-1, -1),
        "param_type": "input", "range": ((1, 1), (1024, 1024))},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_dynamic_two_size_1024",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 3), "shape": (3, 3, 3),
        "param_type": "input"},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
        "param_type": "output"}],
    "case_name": "test_input_invalid",
    "expect": RuntimeError
})

if __name__ == "__main__":
    ut_case.run("Ascend310")
    ut_case.run("Ascend910")
    exit(0)
