from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("AcosGrad", None, None)

def calc_expect_func(inputA, inputB, output):
    input_A_Arr = inputA['value']
    input_B_Arr = inputB['value']
    output_Arr = -1 * input_B_Arr / (np.sqrt(1 - np.square(input_A_Arr)))
    return output_Arr

case1 = {"params": [{"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"},
                    {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"},
                    {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"}],
         "case_name": "acos_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"},
                    {"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"},
                    {"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND"}],
         "case_name": "acos_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

precision_case1 = {"params": [{"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input"},
                              {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input"},
                              {"shape": (10,1), "dtype": "float16", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input"},
                              {"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"input"},
                              {"shape": (10,1), "dtype": "float32", "format": "ND", "ori_shape": (10,1),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910",precision_case1)
ut_case.add_precision_case("Ascend910",precision_case2)
ut_case.add_precision_case("Ascend910",precision_case3)


if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")
