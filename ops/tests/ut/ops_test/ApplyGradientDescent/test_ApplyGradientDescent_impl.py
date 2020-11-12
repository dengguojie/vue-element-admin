
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = OpUT("ApplyGradientDescent", None, None)

def gen_ApplyGradientDescent_case(shape, dtype, case_name_val, expect):
    return {"params": [{"shape": shape, "dtype": dtype, "ori_shape": shape, "ori_format": "ND", "format": "ND"},
                       {"shape": (1,), "dtype": dtype, "ori_shape": (1,), "ori_format": "ND", "format": "ND"},
                       {"shape": shape, "dtype": dtype, "ori_shape": shape, "ori_format": "ND", "format": "ND"},
                       {"shape": shape, "dtype": dtype, "ori_shape": shape, "ori_format": "ND", "format": "ND"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = gen_ApplyGradientDescent_case((1,), "float16", "case_1", "success")
case2 = gen_ApplyGradientDescent_case((1,), "float32", "case_2", "success")

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)

def calc_expect_func(x1, x2, x3, y):
    reuse_var = x1['value'] - x2['value'] * x3['value']
    return reuse_var

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 32),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 2, 32), "dtype": "float32", "format": "ND", "ori_shape": (16, 2, 32),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (11, 33), "dtype": "float32", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (11, 33), "dtype": "float32", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (11, 33), "dtype": "float32", "format": "ND", "ori_shape": (11, 33),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (4,1,1,16,16), "dtype": "float32", "format": "ND", "ori_shape": (4,1,1,16,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4,1,1,16,16), "dtype": "float32", "format": "ND", "ori_shape": (4,1,1,16,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4,1,1,16,16), "dtype": "float32", "format": "ND", "ori_shape": (4,1,1,16,16),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })


