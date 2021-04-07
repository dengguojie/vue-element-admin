#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from impl.batch_matmul import batch_matmul_compute
from impl.drop_out_do_mask_v3_d import drop_out_do_mask_v3_d_compute
from op_test_frame.common import precision_info
from te import tvm

ut_case = OpUT("DropOutDoMaskV3D", None, None)

case1 = {"params": [{"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "uint8", "format": "ND", "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (1, 128), "dtype": "float32", "format": "ND", "ori_shape": (1, 128), "ori_format": "ND"},
                    0.1],
         "case_name": "drop_out_do_mask_v3_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (8, 8, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 128),
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (16384,), "dtype": "uint8", "format": "FRACTAL_NZ", "ori_shape": (16384,),
                     "ori_format": "FRACTAL_NZ"},
                    {"shape": (8, 8, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (128, 128),
                     "ori_format": "FRACTAL_NZ"},
                    0.1],
         "case_name": "drop_out_do_mask_v3_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)


def test_matmul_dropout_ub_fusion_pass(test_arg):
    case0 = {"input_x": (512, 64), "input_x_fractal": (4, 32, 16, 16),
             "input_y": (64, 512), "input_y_fractal": (32, 4, 16, 16),
             "output_z": (512, 512), "output_z_fractal": (32, 32, 16, 16)}
    case1 = {"input_x": (16, 512, 64), "input_x_fractal": (16, 4, 32, 16, 16),
             "input_y": (64, 512), "input_y_fractal": (32, 4, 16, 16),
             "output_z": (16, 512, 512), "output_z_fractal": (16, 32, 32, 16, 16)}
    case2 = {"input_x": (384, 512, 64), "input_x_fractal": (384, 4, 32, 16, 16),
             "input_y": (64, 512), "input_y_fractal": (32, 4, 16, 16),
             "output_z": (24, 16, 512, 512), "output_z_fractal": (24, 16, 32, 32, 16, 16)}
    case3 = {"input_x": (384, 512, 64), "input_x_fractal": (384, 4, 32, 16, 16),
             "input_y": (64, 512), "input_y_fractal": (32, 4, 16, 16),
             "output_z": (1, 24, 16, 16, 512, 512), "output_z_fractal": (1, 24, 16, 32, 32, 16, 16)}
    case4 = {"input_x": (384, 512, 64), "input_x_fractal": (384, 4, 32, 16, 16),
             "input_y": (64, 512), "input_y_fractal": (32, 4, 16, 16),
             "output_z": (1, 1, 24, 16, 512, 512), "output_z_fractal": (1, 1, 24, 16, 32, 32, 16, 16)}
    case5 = {"input_x": (384, 512, 64), "input_x_fractal": (384, 4, 32, 16, 16),
             "input_y": (64, 512), "input_y_fractal": (32, 4, 16, 16),
             "output_z": (1, 1, 1, 24, 16, 512, 512), "output_z_fractal": (1, 1, 1, 24, 16, 32, 32, 16, 16)}
    cases = [case0, case1, case2, case3, case4, case5]

    for case in cases:
        input_x = tvm.placeholder(case["input_x_fractal"], name="input_x", dtype="float16",
                                attrs={"ori_shape": case["input_x"], "format": "FRACTAL_NZ", "ori_format": "ND"})
        input_y = tvm.placeholder(case["input_y_fractal"], name="input_y", dtype="float16",
                                attrs={"ori_shape": case["input_y"], "format": "FRACTAL_NZ", "ori_format": "ND"})
        output_z = {"shape": case["output_z_fractal"], "dtype": "float16", "format": "FRACTAL_NZ",
                    "ori_shape": case["output_z"], "ori_format": "NHWC"}
        input_tensor = batch_matmul_compute(input_x, input_y, output_z=output_z)

        input_mask = tvm.placeholder(case["output_z_fractal"], name="gen_mask", dtype="uint8",
                                    attrs={"ori_shape": case["output_z"], "format": "FRACTAL_NZ", "ori_format": "NHWC"})
        output = tvm.placeholder(case["output_z_fractal"], name="batch_matmul_dropout", dtype="float16",
                                attrs={"ori_shape": case["output_z"], "format": "FRACTAL_NZ", "ori_format": "NHWC"})
        input_keep_prob = 0.5

        try:
            output = drop_out_do_mask_v3_d_compute(
                input_tensor,
                input_mask,
                output,
                input_keep_prob,
                kernel_name="drop_out_do_mask_v3_d"
            )
        except RuntimeError as e:
            print(e)
            pass

ut_case.add_cust_test_func(test_func=test_matmul_dropout_ub_fusion_pass)


if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
