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

HardShrink ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("HardShrink")

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (32, 32),
        "ori_shape": (32, 32),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (32, 32),
        "ori_shape": (32, 32),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }, 1.0],
    "expect": "success"
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{
        "shape": (32, 32),
        "ori_shape": (32, 32),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "input"
    }, {
        "shape": (32, 32),
        "ori_shape": (32, 32),
        "format": "ND",
        "ori_format": "ND",
        "dtype": "float16",
        "param_type": "output"
    }, -1.0],
    "expect": RuntimeError
})


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
