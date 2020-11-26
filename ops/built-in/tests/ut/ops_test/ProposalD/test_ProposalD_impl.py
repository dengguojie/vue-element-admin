
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("ProposalD", None, None)

case1 = {"params": [{"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "ND", "ori_shape": (1,4,1,16),"ori_format": "ND"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    0.5, 0.5, 1.0, [1.0], [1.0], 1, 1, 0.5, False],
         "case_name": "proposal_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
