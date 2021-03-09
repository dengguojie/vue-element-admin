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

HardSigmoid ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("hard_sigmoid")

platforms = ["Ascend910A", "Ascend310", "Ascend610", "Ascend615", "Ascend710", "Ascend910"]
platforms_only_fp16 = ["Hi3796CV300CS", "Hi3796CV300ES", "SD3403"]

ut_case.add_case(support_soc="all", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }, 0.166, 0.5],
    "expect": "success"
})

ut_case.add_case(support_soc="all", case={
    "params": [{
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (1,),
        "ori_shape": (1,),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }, 0.166, -0.5],
    "expect": RuntimeError
})

ut_case.add_case(support_soc="all", case={
    "params": [{
        "shape": (11, 33),
        "ori_shape": (11, 33),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int32",
        "param_type": "input"
    }, {
        "shape": (11, 33),
        "ori_shape": (11, 33),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int32",
        "param_type": "output"
    }, 0.166, 0.5],
    "expect": "success"
})

ut_case.add_case(support_soc="all", case={
    "params": [{
        "shape": (11, 33),
        "ori_shape": (11, 33),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int32",
        "param_type": "input"
    }, {
        "shape": (11, 33),
        "ori_shape": (11, 33),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int32",
        "param_type": "output"
    }, 0.166, -0.5],
    "expect": RuntimeError
})

ut_case.add_case(support_soc=platforms, case={
    "params": [{
        "shape": (10, 6),
        "ori_shape": (10, 6),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "input"
    }, {
        "shape": (10, 6),
        "ori_shape": (10, 6),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "output"
    }, 0.166, 0.5],
    "expect": "success"
})

ut_case.add_case(support_soc=platforms, case={
    "params": [{
        "shape": (10, 6),
        "ori_shape": (10, 6),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "input"
    }, {
        "shape": (10, 6),
        "ori_shape": (10, 6),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float32",
        "param_type": "output"
    }, 0.166, -0.5],
    "expect": RuntimeError
})

ut_case.add_case(support_soc=platforms, case={
    "params": [{
        "shape": (10, 12),
        "ori_shape": (10, 12),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float",
        "param_type": "input"
    }, {
        "shape": (10, 12),
        "ori_shape": (10, 12),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float",
        "param_type": "output"
    }, 0.166, 0.5],
    "expect": RuntimeError
})

ut_case.add_case(support_soc="all", case={
    "params": [{
        "shape": (10, 12),
        "ori_shape": (10, 12),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "uint16",
        "param_type": "input"
    }, {
        "shape": (10, 12),
        "ori_shape": (10, 12),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "uint16",
        "param_type": "output"
    }, 0.166, 0.5],
    "expect": RuntimeError
})

ut_case.add_case(support_soc="all", case={
    "params": [{
        "shape": (3, 7),
        "ori_shape": (3, 7),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int16",
        "param_type": "input"
    }, {
        "shape": (3, 7),
        "ori_shape": (3, 7),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "int16",
        "param_type": "output"
    }, 0.166, 0.5],
    "expect": RuntimeError
})

def gen_params(x_dim_min=1, x_dim_max=4, dim_val_min=1, dim_val_max=10):
    dtype_range = ['float16', 'float32', 'int32']
    np.random.seed(10)
    for cur_type in dtype_range:
        x_dim = np.random.randint(x_dim_min, x_dim_max)
        cur_x_shape = np.random.randint(dim_val_min, dim_val_max, (x_dim, )).tolist()
        cur_alpha = np.random.uniform(0, 1)
        cur_beta = np.random.uniform(0, 1)
        yield cur_type, cur_x_shape, cur_alpha, cur_beta

def calc_expect_func_infer(x, y, alpha, beta):
    x_value = x.get("value")
    result = np.maximum(0, np.minimum(1, alpha * x_value + beta))
    result = result.astype(x.get("dtype"))
    return (result, )


idx = 1
param_gen = gen_params()
for dtype, x_shape, alpha, beta in param_gen:
    if dtype == "float32": # SD3403/Hi3796CV300CS/Hi3796CV300ES cannot support fp32 input data
        platform_select = platforms 
        precision_limit = 0.001
    else:  # fp16 and int32 input data
        platform_select = "all"
        precision_limit = 0.001
    
    ut_case.add_precision_case(platform_select, {"params": [{"shape": x_shape, "dtype": dtype, "format": "ND", "ori_shape": x_shape, "ori_format": "ND", "param_type": "input", "value_range": [-1.0, 1.0]},
                    {"shape": x_shape, "dtype": dtype, "format": "ND", "ori_shape": x_shape,"ori_format": "ND", "param_type": "output"},
                    alpha, beta],
        "case_name": "hard_sigmoid_precision" + str(idx),
        "calc_expect_func": calc_expect_func_infer,
        "precision_standard": precision_info.PrecisionStandard(precision_limit, precision_limit)
        })
    idx += 1


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)