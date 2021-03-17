# # -*- coding:utf-8 -*-
import sys
import torch
import numpy as np
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("resize_d")

def compute_by_torch_cubic(input_x, sizes, scales, flag):
    """
    get output by torch
    """
    x = torch.from_numpy(input_x)
    y = torch._C._nn.upsample_bicubic2d(x, sizes, flag, scales[0], scales[1])
    output = y.numpy()

    return output

def compute_by_torch_linear(input_x, flag, sizes=None, scales=None):
    """
    get output by torch
    """
    x = torch.from_numpy(input_x).float()
    x = x.squeeze(2)
    y = []
    if sizes:
        y = torch.nn.functional.interpolate(
            input=x, size=sizes, mode='linear', align_corners=flag)
    else:
        y = torch.nn.functional.interpolate(
            input=x, scale_factor=scales, mode='linear', align_corners=flag)
    output = y.numpy()
    return output

# pylint: disable=unused-argument,invalid-name,consider-using-enumerate
def calc_expect_func(input_x, y, sizes, scales, roi, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode):
    flag = False
    if coordinate_transformation_mode == "align_corners":
        flag = True

    output = []
    if mode == "linear":
        if coordinate_transformation_mode == "align_corners":
            output = compute_by_torch_linear(input_x["value"], flag, sizes=sizes[0])
        else:
            output = compute_by_torch_linear(input_x["value"], flag, scales=scales[0])
    else:
        output = compute_by_torch_cubic(input_x["value"], sizes, scales, flag)

    if input_x["dtype"] == "float16":
        res = output.astype(np.float16)
        return [res, ]
    else:
        return [output, ]

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "output"}, [2, 2], [0.5, 0.5], [0, 0], "half_pixel", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_cubic_y_1"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 20), "shape": (1, 1, 3, 20),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 30), "shape": (1, 1, 3, 30),
                "param_type": "output"}, [3, 30], [0.5, 0.5], [0, 0], "align_corners", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_cubic_y_2"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 20), "shape": (1, 1, 3, 20),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 30), "shape": (1, 1, 3, 30),
                "param_type": "output"}, [3, 30], [0.0, 0.0], [0, 0], "half_pixel", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_cubic_y_3"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "output"}, [2, 1], [0.5, 0.5], [0, 0], "half_pixel", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_cubic_y_4"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "output"}, [1, 2], [0.5, 0.5], [0, 0], "half_pixel", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_cubic_y_5"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 20), "shape": (1, 1, 3, 20),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 30), "shape": (1, 1, 3, 30),
                "param_type": "output"}, [3, 3, 30], [0.5, 0.5], [0, 0], "half_pixel", -0.75, 0, 0.0],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError,
    "case_name": "test_is_resize_d_case_cubic_6"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 20), "shape": (1, 1, 3, 20),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 3, 30), "shape": (1, 1, 3, 30),
                "param_type": "output"}, [3, 3, 30], [0.5, 0.5], [0, 0], "half_pixel", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError,
    "case_name": "test_is_resize_d_case_cubic_7"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 0, 2), "shape": (1, 1, 0, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "output"}, [2, 2], [0.5, 0.5], [0, 0], "half_pixel", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError,
    "case_name": "test_is_resize_d_case_cubic_8"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 2, 2), "shape": (1, 1, 2, 2),
                "param_type": "output"}, [2, 2], [0.5, 0.5, 0.5], [0, 0], "half_pixel", -0.75, 0, 0.0, "cubic"],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError,
    "case_name": "test_is_resize_d_case_cubic_9"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 1, 2), "shape": (4, 1, 1, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 4), "shape": (4, 1, 4),
                "param_type": "output"}, [4, ], [2.0, ], [2, ], "half_pixel", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_precision_case_linear_y_1"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9, 7, 1, 2), "shape": (9, 7, 1, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9, 7, 14), "shape": (9, 7, 14),
                "param_type": "output"}, [14, ], [7.0, ], [7, ], "align_corners", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_precision_case_linear_y_2"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (627, 2, 1, 3), "shape": (627, 2, 1, 3),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (627, 2, 18), "shape": (627, 2, 18),
                "param_type": "output"}, [18, ], [6.0, ], [6, ], "half_pixel", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_precision_case_linear_y_3"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 1, 2), "shape": (4, 1, 1, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 4), "shape": (4, 1, 4),
                "param_type": "output"}, [4, ], [2.0, ], [2, ], "half_pixel", 0.75, 0, 0.0],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_linear_1",
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 1, 2), "shape": (4, 1, 1, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 4), "shape": (4, 1, 4),
                "param_type": "output"}, [4, 4], [2.0, ], [2, ], "half_pixel", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_linear_2",
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 1, 2), "shape": (4, 1, 1, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 4), "shape": (4, 1, 4),
                "param_type": "output"}, [4, ], [2.0, 2.0], [2, ], "half_pixel", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_linear_3",
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9, 7, 1, 0), "shape": (9, 7, 1, 0),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9, 7, 18), "shape": (9, 7, 18),
                "param_type": "output"}, [18, ], [6.0, ], [18, ], "half_pixel", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_linear_4",
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 2, 2), "shape": (4, 1, 2, 2),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 4), "shape": (4, 1, 4),
                "param_type": "output"}, [4, ], [2.0, ], [2, ], "half_pixel", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_linear_5",
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 1, 1), "shape": (4, 1, 1, 1),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 4), "shape": (4, 1, 4),
                "param_type": "output"}, [4, ], [2.0, ], [2, ], "half_pixel", 0.75, 0, 0.0, "linear"],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_resize_d_case_linear_6",
    "expect": RuntimeError
})
