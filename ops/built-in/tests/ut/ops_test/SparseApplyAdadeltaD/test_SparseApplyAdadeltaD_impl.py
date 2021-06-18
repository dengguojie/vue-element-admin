"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

SparseApplyAdadeltaD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
import random
ut_case = OpUT("SparseApplyAdadeltaD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001],
         "case_name": "SparseApplyAdadeltaD_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.0001],
         "case_name": "SparseApplyAdadeltaD_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)



def calc_expect_func(var, accum, accum_update, lr, rho, grad, indices,
                     out_var, out_accum, out_accum_update, epsilon):
    var_data = var["value"]
    accum_data = accum["value"]
    accum_update_data = accum_update["value"]
    grad = grad["value"]
    indices = indices["value"]

    lr = lr["value"]
    rho = rho["value"]
    for i, idx in enumerate(indices):
        accum_data[idx] = accum_data[idx] * rho[0] + grad[i] * grad[i] * (1 - rho[0])
        update = np.sqrt(accum_update_data[idx] + epsilon) * grad[i] / np.sqrt(accum_data[idx] + epsilon)
        var_data[idx] = var_data[idx] - update * lr[0]
        accum_update_data[idx] = accum_update_data[idx] * rho[0] + update * update * (1 - rho[0])
    return [var_data, accum_data, accum_update_data]

ut_case.add_precision_case(["Ascend710", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 64), "shape": (4, 1, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 64), "shape": (4, 1, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 64), "shape": (4, 1, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 64), "shape": (4, 1, 64), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (4, ), "shape": (4, ), "param_type": "input", "value":np.array(random.sample(range(4), 4)).astype("int32")},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 64), "shape": (4, 1, 64), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 64), "shape": (4, 1, 64), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 64), "shape": (4, 1, 64), "param_type": "output"},
               1e-8],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case(["Ascend710", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (14, 1, 8, 16), "shape": (14, 1, 8, 16), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (14, 1, 8, 16), "shape": (14, 1, 8, 16), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (14, 1, 8, 16), "shape": (14, 1, 8, 16), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (14, 1, 8, 16), "shape": (14, 1, 8, 16), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (14, ), "shape": (14, ), "param_type": "input", "value":np.array(random.sample(range(14), 14)).astype("int32")},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (14, 1, 8, 16), "shape": (14, 1, 8, 16), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (14, 1, 8, 16), "shape": (14, 1, 8, 16), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (14, 1, 8, 16), "shape": (14, 1, 8, 16), "param_type": "output"},
               1e-8],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case(["Ascend710", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, 1, 64), "shape": (128, 1, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, 1, 64), "shape": (128, 1, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, 1, 64), "shape": (128, 1, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, 1, 64), "shape": (128, 1, 64), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "input", "value":np.array(random.sample(range(128), 128)).astype("int32")},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, 1, 64), "shape": (128, 1, 64), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, 1, 64), "shape": (128, 1, 64), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, 1, 64), "shape": (128, 1, 64), "param_type": "output"},
               1e-8],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case(["Ascend710", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 317), "shape": (4, 1, 317), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 317), "shape": (4, 1, 317), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 317), "shape": (4, 1, 317), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 317), "shape": (4, 1, 317), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (4, ), "shape": (4, ), "param_type": "input", "value":np.array(random.sample(range(4), 4)).astype("int32")},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 317), "shape": (4, 1, 317), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 317), "shape": (4, 1, 317), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1, 317), "shape": (4, 1, 317), "param_type": "output"},
               1e-8],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run("Ascend910A")
