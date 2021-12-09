# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import torch
import numpy as np

ut_case = OpUT("reduce_std_v2_update")

# pylint: disable=unused-argument
def calc_expect_func(input_x, mean, output_var, dim, if_std, unbiased, keepdim):
    x = input_x["value"]
    x = torch.tensor(x)
    if(if_std):
        res = torch.std(x, dim, unbiased, keepdim).numpy()
        return[res, ]
    else:
        res = torch.var(x, dim, unbiased, keepdim).numpy()
        return [res, ]


def gen_case(dtype_val, inp_shape_val, outp_shape_val, dim_val, if_std_val, unbiased_val, keepdim_val, x_val, mean_val, precision):
    return({
        "params": [{"dtype": dtype_val,
                    "format": "ND", "ori_format": "ND",
                    "shape": inp_shape_val, "ori_shape": inp_shape_val,
                    "value": np.array(x_val, dtype=dtype_val), "param_type": "input"},
                   {"dtype": dtype_val,
                    "format": "ND", "ori_format": "ND",
                    "shape": inp_shape_val, "ori_shape": inp_shape_val,
                    "value": np.array(mean_val, dtype=dtype_val), "param_type": "input"},
                   {"dtype": dtype_val,
                    "format": "ND", "ori_format": "ND",
                    "shape": outp_shape_val, "ori_shape": outp_shape_val, "param_type": "output"},
                   dim_val, if_std_val, unbiased_val, keepdim_val],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(precision, precision),
        "case_name": dtype_val

    })


ut_case.add_precision_case("all", gen_case("float16", (2, 4), (2, 1), [1, ], True, True, True,
                                           [[1, 2, 3, 4], [5, 6, 7, 8]],
                                           [[2.5, 2.5, 2.5, 2.5], [6.5, 6.5, 6.5, 6.5]], 0.001))

ut_case.add_precision_case("all", gen_case("float32", (2, 4), (2, 1), [1, ], False, False, True,
                                           [[1, 2, 3, 4], [5, 6, 7, 8]],
                                           [[2.5, 2.5, 2.5, 2.5], [6.5, 6.5, 6.5, 6.5]], 0.0001))

ut_case.add_precision_case("all", gen_case("float16", (3, 2, 2), (3, ), [1, 2], False, False, False,
                                           [[[1, 3], [3, 5]], [[5, 7], [7, 9]], [[2, 6], [8, 4]]],
                                           [[[3, 3], [3, 3]], [[7, 7], [7, 7]], [[5, 5], [5, 5]]], 0.001))

ut_case.add_precision_case("all", gen_case("float32", (3, 2, 2), (3, 2), [1, ], True, True, False,
                                           [[[1, 3], [3, 5]], [[5, 7], [7, 9]], [[2, 6], [8, 4]]],
                                           [[[2, 4], [2, 4]], [[6, 8], [6, 8]], [[5, 5], [5, 5]]], 0.0001))

ut_case.add_case("all", {
    "params": [{"dtype": "float16",
                "format": "ND", "ori_format": "ND",
                "shape": (2, 2), "ori_shape": (2, 2)},
               {"dtype": "float16",
                "format": "ND", "ori_format": "ND",
                "shape": (2, 3), "ori_shape": (2, 3)},
               {"dtype": "float16",
                "format": "ND", "ori_format": "ND",
                "shape": (2, 2), "ori_shape": (2, 2)},
               [1], True, True, True],
    "expect": RuntimeError,
    "case_name": "para_check"
})
