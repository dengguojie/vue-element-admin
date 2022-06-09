# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT("SignBitsPack", "impl.dynamic.sign_bits_pack", "sign_bits_pack")

ut_case.add_case(["Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "shape": (12348 * 8,), "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "shape": (1, 12348,), "param_type": "output"},
               1
               ],
    "case_name": "sign_bit_pack_0",
    "expect": "success",
    "format_expect": ["ND"],
    "support_expect": True})

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
    exit(0)