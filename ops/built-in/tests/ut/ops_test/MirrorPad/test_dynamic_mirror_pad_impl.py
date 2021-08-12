"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic Pad ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("MirrorPad", "impl.dynamic.mirror_pad", "mirror_pad")


# pylint: disable=unused-argument
def tensor_dict(tensor_shape, tensor_format, tensor_type):
    """
    return a dict
    """
    gen_dict = dict()
    gen_dict["ori_shape"] = tensor_shape
    gen_dict["ori_format"] = tensor_format
    gen_dict["dtype"] = tensor_type
    gen_dict["shape"] = tensor_shape
    gen_dict["format"] = tensor_format
    gen_dict["range"] = [(1, 100000)] * len(tensor_shape)

    return gen_dict


ut_case.add_case(["Ascend910A"],
                 {"params": [tensor_dict([-1, -1, -1], "ND", "float16"),
                             tensor_dict([-1, -1, -1], "ND", "int32"),
                             tensor_dict([-1, -1, -1], "ND", "float16"),
                             "REFLECT",
                            ],
                  "case_name": "dynamic_mirror_pad_case_1",
                  "expect": "success",
                  "support_expect": True})
ut_case.add_case(["Ascend910A"],
                 {"params": [tensor_dict([-2], "ND", "float32"),
                             tensor_dict([-2], "ND", "int64"),
                             tensor_dict([-2], "ND", "float32"),
                             "SYMMETRIC"
                            ],
                  "case_name": "dynamic_mirror_pad_case_2",
                  "expect": "success",
                  "support_expect": True})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
