#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

Dynamic Unpack ut case
"""
import tbe
from op_test_frame.ut import OpUT


ut_case = OpUT("Unpack", "impl.dynamic.unpack", "unpack")


def gen_dynamic_unpack_case(dict_input, list_dict_outs, num, axis, kernel_name_val, expect):
    return {"params": [dict_input, list_dict_outs, num, axis],
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_dynamic_unpack_case(
                     {"shape": (-1, -1, -1), "dtype": "float32", "ori_shape": (-1, -1, -1),
                      "format": "ND", "ori_format": "ND", "range": ((1, 32), (1, 2),(1, 20))},
                    [{"shape": (32,20), "dtype": "float32", "ori_shape": (32,20),
                      "format": "ND", "ori_format": "ND","range":((1,32),(1,20))},{"shape": (32,20), "dtype": "float32", "ori_shape": (32,20),
                      "format": "ND", "ori_format": "ND","range":((1,32),(1,20))}],2,1,
                     "dynamic_unpack_01", "success"))

ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"],
                 gen_dynamic_unpack_case(
                     {"shape": (-1, -1, -1), "dtype": "float32", "ori_shape": (-1, -1, -1),
                      "format": "ND", "ori_format": "ND", "range": ((1, None), (1, None), (1, 20))},
                    [{"shape": (32,20), "dtype": "float32", "ori_shape": (32,20),
                      "format": "ND", "ori_format": "ND","range":((1,32),(1,20))},{"shape": (32,20), "dtype": "float32", "ori_shape": (32,20),
                      "format": "ND", "ori_format": "ND","range":((1,32),(1,20))}],2,1,
                     "dynamic_unpack_01", "success"))


if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
