#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("NLLLoss", "impl.dynamic.nll_loss", "nll_loss")


def test_op_check_supported(test_arg):
    from impl.dynamic.nll_loss import check_supported
    check_supported({"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1), "ori_format": "ND"},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND"},
                    {"shape": (-1,), "dtype": "float32", "format": "ND", "ori_shape": (-1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    "sum", -100)
    check_supported({"shape": (2, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 16), "ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2,), "ori_format": "ND"},
                    {"shape": (16,), "dtype": "float32", "format": "ND", "ori_shape": (16,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                    "sum", -100)

def gen_nllloss_case(dynamic_input_shape_list, ori_input_shape_list,
                     dtype, dtype_target, src_format, reduction,
                     ignore_idx, case_name_val, expect):
    """
    generate ut case
    """
    inputs = []
    for i in range(3):
        if i == 1:
            input_type = dtype_target
        else:
            input_type = dtype
        inputs.append({"shape": dynamic_input_shape_list[i],
                       "dtype": input_type,
                       "ori_shape": ori_input_shape_list[i],
                       "ori_format": src_format,
                       "format": src_format,
                       'range': [[1, 200000000]] * len(dynamic_input_shape_list[i])})
    outputs = []
    for i in range(2):
        outputs.append(
            {"shape": (-1,),
            "dtype": dtype,
            "ori_shape": (1,),
            "ori_format": src_format,
            "format": src_format,
            'range': [[1, 200000000]]},
        )
    return {"params": [inputs[0], inputs[1], inputs[2], outputs[0],
                       outputs[1], reduction, ignore_idx],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1,)],
                                  [(2, 16), (2,), (16,)],
                                  "float32", "int32", "ND",
                                  "none", -100, "case_1", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1,)],
                                  [(2, 16), (2,), (16,)],
                                  "float32", "int32", "ND",
                                  "sum", -100, "case_2", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1,)],
                                  [(2, 16), (2,), (16,)],
                                  "float32", "int32", "ND",
                                  "mean", -100, "case_3", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1,), (-1,), (-1,)],
                                  [(16,), (1,), (16,)],
                                  "float32", "int32", "ND",
                                  "mean", 5, "case_4", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1,), (-1,), (-1,)],
                                  [(16,), (1,), (16,)],
                                  "float32", "int32", "ND",
                                  "none", 5, "case_5", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1,), (-1,), (-1,)],
                                  [(16,), (1,), (16,)],
                                  "float32", "int32", "ND",
                                  "sum", 5, "case_6", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1,)],
                                  [(200, 15003), (200,), (15003,)],
                                  "float32", "int32", "ND",
                                  "none", -100, "case_7", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1,)],
                                  [(200, 15003), (200,), (15003,)],
                                  "float32", "int32", "ND",
                                  "sum", -100, "case_8", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1,)],
                                  [(200, 15003), (200,), (15003,)],
                                  "float32", "int32", "ND",
                                  "mean", -100, "case_9", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1, -1), (-1,), (-1,)],
                                  [(200, 15003, 1), (200,), (15003,)],
                                  "float32", "int32", "ND",
                                  "mean", -100, "case_10", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1, -1), (-1,)],
                                  [(200, 15003), (200, 1), (15003,)],
                                  "float32", "int32", "ND",
                                  "mean", -100, "case_11", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1, -1)],
                                  [(200, 15003), (200,), (15003, 1)],
                                  "float32", "int32", "ND",
                                  "mean", -100, "case_12", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(200, -1), (-1,), (-1,)],
                                  [(200, 15003), (200,), (15003,)],
                                  "float32", "int32", "ND",
                                  "mean", -100, "case_13", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, 15003), (-1,), (-1,)],
                                  [(200, 15003), (200,), (15003,)],
                                  "float32", "int32", "ND",
                                  "mean", -100, "case_14", RuntimeError))

ut_case.add_case(["Ascend910A"],
                 gen_nllloss_case([(-1, -1), (-1,), (-1,)],
                                  [(200, 15003), (200,), (15003,)],
                                  "float32", "int32", "ND",
                                  "other", -100, "case_15", RuntimeError))

ut_case.add_cust_test_func(test_func=test_op_check_supported)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
