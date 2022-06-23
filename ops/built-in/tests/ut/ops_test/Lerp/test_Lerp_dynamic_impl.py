"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Lerp ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("Lerp", "impl.dynamic.lerp", "lerp")


ut_case.add_case(["all"], {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "input", "range": [(1, 100)]},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "input", "range": [(1, 100)]},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "input", "range": [(1, 100)]},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "output", "range": [(1, 100)]}],
    "case_name": "test_lerp_dynamic_unknownshape_case_fp16",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
})
ut_case.add_case(["all"], {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "input", "range": [(1, 100)]},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "input", "range": [(1, 100)]},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "input", "range": [(1, 100)]},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 1, 2), "shape": (-1, 1, 2),"param_type": "output", "range": [(1, 100)]}],
    "case_name": "test_lerp_dynamic_unknownshape_case_fp32",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
})
if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
    exit(0)
