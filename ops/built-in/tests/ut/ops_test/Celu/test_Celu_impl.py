#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import torch

ut_case = OpUT("Celu",None,None)

def calc_expect_func(input_x, output_y, alpha=1.0):
    x = input_x["value"]
    torch_x = torch.Tensor(x)
    res = torch.celu(torch_x, alpha=alpha).numpy()
    return [res, ]


ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              1.0],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (5, 13, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "output"},
                                              1.0],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
                                              1.0],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "output"},
                                              1.0],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              -1.0],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (5, 13, 4), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (5, 13, 4), "dtype": "float16", "format": "ND", "ori_shape": (5, 13, 4),"ori_format": "ND", "param_type": "output"},
                                              -3.8],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"},
                                              1.5],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("Ascend310", {"params": [{"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (16, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "output"},
                                              -2.2],
                                   "calc_expect_func": calc_expect_func
                                   })

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
               2.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
               -1.1],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 32, 3, 3),
                "shape": (32, 2, 3, 3, 16), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 32, 3, 3,),
                "shape": (32, 2, 3, 3, 16), "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 30, 3, 3),
                "shape": (32, 2, 3, 3, 16), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 30, 3, 3,),
                "shape": (32, 2, 3, 3, 16), "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape": (32, 32),
                "shape": (2, 2, 16, 16), "param_type": "input"},
               {"dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape": (32, 32),
                "shape": (2, 2, 16, 16), "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case(["Ascend910A", "Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case(["Ascend910A", "Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
               2.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case(["Ascend910A", "Ascend710"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"},
               -1.1],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case(["Ascend910A", "Ascend710"], {
    "params": [{"dtype": "float32", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 32, 3, 3),
                "shape": (32, 2, 3, 3, 16), "param_type": "input"},
               {"dtype": "float32", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 32, 3, 3,),
                "shape": (32, 2, 3, 3, 16), "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case(["Ascend910A", "Ascend710"], {
    "params": [{"dtype": "float32", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 30, 3, 3),
                "shape": (32, 2, 3, 3, 16), "param_type": "input"},
               {"dtype": "float32", "format": "NC1HWC0", "ori_format": "ND", "ori_shape": (32, 30, 3, 3,),
                "shape": (32, 2, 3, 3, 16), "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case(["Ascend910A", "Ascend710"], {
    "params": [{"dtype": "float32", "format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape": (32, 32),
                "shape": (2, 2, 16, 16), "param_type": "input"},
               {"dtype": "float32", "format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape": (32, 32),
                "shape": (2, 2, 16, 16), "param_type": "output"},
               1.0],

    "calc_expect_func": calc_expect_func
})

case_error = {"params": [{"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                         {"shape": (1,16), "dtype": "float32", "format": "ND", "ori_shape": (1,16),"ori_format": "ND"},
                         1.0, 0, 1.0],
                         "case_name": "case_error",
             "expect": RuntimeError,
             "format_expect": [],
             "support_expect": True}
ut_case.add_case(["Ascend910A"], case_error)
