
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ApplyAdaMaxD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"}],
         "case_name": "apply_ada_max_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (2,4), "dtype": "float16", "format": "ND", "ori_shape": (2,4),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"}],
         "case_name": "apply_ada_max_d_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,2), "dtype": "float16", "format": "ND", "ori_shape": (1,2),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"},
                    {"shape": (1,5), "dtype": "float16", "format": "ND", "ori_shape": (1,5),"ori_format": "ND"},
                    {"shape": (1,6), "dtype": "float16", "format": "ND", "ori_shape": (1,6),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    {"shape": (1,3), "dtype": "float16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"}],
         "case_name": "apply_ada_max_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)




if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)




