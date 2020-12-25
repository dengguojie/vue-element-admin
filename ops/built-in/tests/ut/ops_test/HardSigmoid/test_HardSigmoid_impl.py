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

ut_case = OpUT("hard_sigmoid")

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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

ut_case.add_case(support_soc="Ascend910A", case={
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


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)