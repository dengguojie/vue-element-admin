
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

case2 = {"params": [{"shape": (1, 18, 23, 77), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 18, 23, 77),"ori_format": "NCHW"},
            {"shape": (1, 36, 23, 77), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 36, 23, 77),"ori_format": "NCHW"},
            {"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2),"ori_format": "NCHW"},
            {"shape": (1, 36, 23, 77), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 36, 23, 77),"ori_format": "NCHW"},
            {"shape": (1, 5, 304), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 5, 304),"ori_format": "NCHW"},
            {"shape": (1, 8), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 8),"ori_format": "NCHW"},
            16.0,
            16.0,
            10.0,
            (0.5, 1.0, 2.0),
            (8.0, 16.0, 32.0),
            3000,
            304,
            1.0,
            True],
         "case_name": "proposal_d_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "ND", "ori_shape": (1,4,1,16),"ori_format": "ND"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,4,1,16), "dtype": "float16", "format": "NCHW", "ori_shape": (1,4,1,16),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int32", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    0.5, 0.5, 1.0, [1.0], [1.0], 1, 3001, 0.5, False],
         "case_name": "proposal_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

#ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend910A"], case2)
#ut_case.add_case(["Hi3796CV300ES"], case3)
def test_proposald_sd3403_1(test_arg):
    from impl.proposal_d import proposal_d
    from te import platform as cce_conf
    cce_conf.cce_conf.te_set_version("SD3403", core_type="AiCore")
    proposal_d({"shape": (1, 18, 23, 77), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 18, 23, 77),"ori_format": "NCHW"},
            {"shape": (1, 36, 23, 77), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 36, 23, 77),"ori_format": "NCHW"},
            {"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2),"ori_format": "NCHW"},
            {"shape": (1, 36, 23, 77), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 36, 23, 77),"ori_format": "NCHW"},
            {"shape": (1, 5, 304), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 5, 304),"ori_format": "NCHW"},
            {"shape": (1, 8), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 8),"ori_format": "NCHW"},
            16.0,
            16.0,
            10.0,
            (0.5, 1.0, 2.0),
            (8.0, 16.0, 32.0),
            3000,
            304,
            0.69999999,
            True)
    cce_conf.cce_conf.te_set_version(test_arg)
ut_case.add_cust_test_func(test_func=test_proposald_sd3403_1)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
