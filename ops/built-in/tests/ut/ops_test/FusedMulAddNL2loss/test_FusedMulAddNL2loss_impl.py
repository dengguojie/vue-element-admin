from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("FusedMulAddnL2Loss", None, "fused_mul_addn_l2loss")

def calc_expect_func(x, y, z, output_x, output_y):
    res = x["value"] * z["value"] + y["value"]
    res1 = res.astype(output_x['dtype'])

    axis = [i for i in range(len(x["shape"]))]
    sqrt = 1.0 / np.sqrt(2)
    res2 = (x["value"] * sqrt)**2
    res2 = res2.sum(axis=tuple(axis))
    res2 = res2.astype(output_y["dtype"])
    return res1, res2

case1 = {"params": [{"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"},
                    {"shape": (2, 4, 4), "dtype": "float32", "format": "ND", "ori_shape": (2, 4, 4),"ori_format": "ND"}],
         "case_name": "FusedMulAddNL2loss_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "FusedMulAddNL2loss_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (4, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 64, 64),"ori_format": "HWCN"},
                    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 64, 64),"ori_format": "HWCN"},
                    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 64, 64),"ori_format": "HWCN"},
                    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 64, 64),"ori_format": "HWCN"},
                    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (1, 1, 64, 64),"ori_format": "HWCN"}],
         "case_name": "FusedMulAddNL2loss_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

precision_case1 = {"params": [{"shape": (2,4,4), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,4,4), "param_type":"input"},
                              {"shape": (2,4,4), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,4,4),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (2,4,4), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,4,4),"param_type":"output"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (1,2), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,2), "param_type":"input"},
                              {"shape": (1,2), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,2),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (1,2), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,2),"param_type":"output"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

# ut_case.add_precision_case("Ascend910", precision_case1)
# ut_case.add_precision_case("Ascend910", precision_case2)

