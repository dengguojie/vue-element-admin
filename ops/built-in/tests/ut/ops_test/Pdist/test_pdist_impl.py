# # -*- coding:utf-8 -*-
import torch
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("pdist")

#pylint: disable=unused-argument
def calc_expect_func(input_x, output_y, p):
    input_tensor = torch.from_numpy(input_x["value"])
    res_tensor = torch.nn.functional.pdist(input_tensor, p)
    output = res_tensor.numpy()
    return output

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 200), "shape": (10, 200),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, 1.0],
    "case_name": "test_is_pdist_precision_case_1",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (10, 200), "shape": (10, 200),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, 0.0],
    "case_name": "test_is_pdist_case_1",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (10, 1000), "shape": (10, 1000),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, 1.0],
    "case_name": "test_is_pdist_case_2",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 1000), "shape": (10, 1000),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, 1.0],
    "case_name": "test_is_pdist_case_3",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 2560), "shape": (10, 2560),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, 0.0],
    "case_name": "test_is_pdist_case_4",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 10), "shape": (2, 10),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"}, 1.0],
    "case_name": "test_is_pdist_case_5",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 102400), "shape": (2, 102400),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"}, 1.0],
    "case_name": "test_is_pdist_case_6",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 102400), "shape": (2, 102400),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"}, 1.0],
    "case_name": "test_is_pdist_case_7",
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 12), "shape": (10, 12),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, 1.0],
    "case_name": "test__error_case_1",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 11, 12), "shape": (10, 11, 12),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, 2.0],
    "case_name": "test__error_case_2",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10, 11), "shape": (10, 11),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (45, ), "shape": (45, ),
                "param_type": "output"}, -1.0],
    "case_name": "test__error_case_3",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (10000, 100), "shape": (10000, 100),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (49995000, ), "shape": (49995000, ),
                "param_type": "output"}, 1.0],
    "case_name": "test__error_case_4",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 100), "shape": (1, 100),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"}, 1.0],
    "case_name": "test__error_case_5",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (10000, 100), "shape": (10000, 100),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (49995000, ), "shape": (49995000, ),
                "param_type": "output"}, 1.0],
    "case_name": "test__error_case_6",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})
