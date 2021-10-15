from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("FastGeluGrad", None, None)


case1 = {"params": [{"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND"}],
         "case_name": "FastGeluGrad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (2, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND"}],
         "case_name": "FastGeluGrad_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (2, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND"},
                    {"shape": (2, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND"}],
         "case_name": "FastGeluGrad_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)
ut_case.add_case("Ascend910", case3)

def calc_expect_func(dy, x, y):
    input_x = x['value']
    attr = 1.702
    attr_opp = 0 - attr
    attr_half = attr / 2

    abs_x = np.abs(input_x)
    mul_abs_x = abs_x * attr_opp
    exp_x = np.exp(mul_abs_x)

    add_2 = input_x * exp_x * attr
    exp_pn_x = np.exp((input_x - abs_x) * attr)

    div_up = exp_x + add_2 +exp_pn_x
    div_down = (exp_x + 1) ** 2

    res = div_up / div_down
    res = dy['value'] * res
    return res

ut_case.add_precision_case("all", {"params": [{"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (3, 16, 32), "dtype": "float16", "format": "ND", "ori_shape": (3, 16, 32),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                   })

ut_case.add_precision_case("all", {"params": [{"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1, 3, 100, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 100, 16),"ori_format": "ND", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.01, 0.01)
                                   })

