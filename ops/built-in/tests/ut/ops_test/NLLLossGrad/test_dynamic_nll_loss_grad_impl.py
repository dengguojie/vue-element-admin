#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("NLLLossGrad", "impl.dynamic.nll_loss_grad", "nll_loss_grad")

def test_op_check_supported(test_arg):
    from impl.dynamic.nll_loss_grad import check_supported
    check_supported({"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND"},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (2, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 16), "ori_format": "ND"},
                    "sum", -100)

    check_supported({"shape": (2, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 16), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (2, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 16), "ori_format": "ND"},
                    "sum", -100)

    check_supported({"shape": (818497, 2), "dtype": "float32", "format": "ND", "ori_shape": (818497, 2), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (818497,), "dtype": "int32", "format": "ND", "ori_shape": (818497,), "ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (818497, 2), "dtype": "float32", "format": "ND", "ori_shape": (818497, 2), "ori_format": "ND"},
                    "sum", -100)


def gen_nlllossgrad_case(dynamic_input_shape_list, ori_input_shape_list, dtype, dtype_target, src_format, reduction,
                         ignore_idx, case_name_val, expect):
    """
    generate ut case
    """
    inputs = []
    for i in range(5):
        if i == 2:
            input_type = dtype_target
        else:
            input_type = dtype
        inputs.append({"shape": dynamic_input_shape_list[i],
                       "dtype": input_type,
                       "ori_shape": ori_input_shape_list[i],
                       "ori_format": src_format,
                       "format": src_format,
                       'range': [[1, 200000000]] * len(dynamic_input_shape_list[i])})
    outputs = (
        {"shape": dynamic_input_shape_list[0],
         "dtype": dtype,
         "ori_shape": ori_input_shape_list[0],
         "ori_format": src_format,
         "format": src_format,
         'range': [[1, 200000000]] * len(dynamic_input_shape_list[0])},
    )
    return {"params": [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                       outputs[0], reduction, ignore_idx],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1,), (-1,), (-1,), (-1,)],
                                      [(2, 16), (2,), (2,), (16,), (1,)],
                                      "float32", "int32", "ND", "none", 3, "case_1", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1,), (-1,), (-1,), (-1,)],
                                      [(2, 16), (1,), (2,), (16,), (1,)],
                                      "float32", "int32", "ND", "sum", -100, "case_2", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1,), (-1,), (-1,), (-1,)],
                                      [(2, 16), (1,), (2,), (16,), (1,)],
                                      "float32", "int32", "ND", "mean", 100, "case_3", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1,), (-1,), (-1,), (-1,), (-1,)],
                                      [(16,), (1,), (16,), (16,), (1,)],
                                      "float32", "int32", "ND", "mean", 5, "case_4", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1,), (-1,), (-1,), (-1,), (-1,)],
                                      [(16,), (1,), (16,), (16,), (1,)],
                                      "float32", "int32", "ND", "none", -1, "case_5", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1,), (-1,), (-1,), (-1,), (-1,)],
                                      [(16,), (1,), (16,), (16,), (1,)],
                                      "float32", "int32", "ND", "sum", 100, "case_6", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1,), (-1,), (-1,), (-1,), (-1,)],
                                      [(16,), (1,), (16,), (16,), (1,)],
                                      "float16", "int32", "ND", "sum", 100, "case_7", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1,), (-1,), (-1,), (-1,), (-1,)],
                                      [(16,), (1,), (16,), (16,), (1,)],
                                      "float32", "float32", "ND", "sum", 100, "case_8", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1, -1), (-1,), (-1,), (-1,), (-1,)],
                                      [(2, 16, 16), (2,), (2,), (16,), (1,)],
                                      "float32", "int32", "ND", "none", 3, "case_9", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1, -1), (-1,), (-1,), (-1,)],
                                      [(2, 16), (2, 16), (2,), (16,), (1,)],
                                      "float32", "int32", "ND", "none", 3, "case_10", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1,), (-1, -1), (-1,), (-1,)],
                                      [(2, 16), (2,), (2, 16), (16,), (1,)],
                                      "float32", "int32", "ND", "none", 3, "case_11", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1,), (-1,), (-1, -1), (-1,)],
                                      [(2, 16), (2,), (2,), (16, 16), (1,)],
                                      "float32", "int32", "ND", "none", 3, "case_12", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1,), (-1,), (-1,), (-1, -1)],
                                      [(2, 16), (2,), (2,), (16,), (1, 1)],
                                      "float32", "int32", "ND", "none", 3, "case_13", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nlllossgrad_case([(-1, -1), (-1,), (-1,), (-1,), (-1,)],
                                      [(2, 16), (2,), (2,), (16,), (1,)],
                                      "float32", "int32", "ND", "test", 3, "case_14", RuntimeError))

ut_case.add_cust_test_func(test_func=test_op_check_supported)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
