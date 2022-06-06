# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT("SignBitsUnpack", "impl.dynamic.sign_bits_unpack", "sign_bits_unpack")

ut_case.add_case(["Ascend910A"], {
    "params": [{"dtype": "uint8", "format": "ND", "shape": (242,), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "shape": (1, 242 * 8,), "param_type": "output"},
               1,
               1
               ],
    "case_name": "sign_bits_unpack_0",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)