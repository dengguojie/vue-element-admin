# -*- coding: UTF-8 -*-
import tbe

from op_test_frame.ut import OpUT
ut_case = OpUT("Relu6Grad", "impl.dynamic.relu6_grad", "relu6_grad")

case1 = {"params": [{"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #x
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]}, #h
                    {"shape": (-1,), "dtype": "float16", "format": "ND", "ori_shape": (15, 32),"ori_format": "ND","range":[(1, 100)]},
                    ],
         "case_name": "Assign_1",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])

