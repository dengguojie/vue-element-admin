#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("DropOutDoMask", "impl.dynamic.drop_out_do_mask", "drop_out_do_mask")

case1 = {"params": [{"shape": (20480*32,), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (2560*32,), "dtype": "uint8", "format": "ND", "ori_shape": (16,),"ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (20480*32,), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "range":[(1, 100)]}],
         "case_name": "drop_out_do_mask_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (20480*32,), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (2560*32,), "dtype": "uint8", "format": "ND", "ori_shape": (16,),"ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (20480*32,), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "range":[(1, 100)]}],
         "case_name": "drop_out_do_mask_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)

