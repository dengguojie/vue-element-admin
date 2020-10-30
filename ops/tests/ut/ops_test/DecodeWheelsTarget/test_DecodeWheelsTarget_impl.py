
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("DecodeWheelsTarget", None, None)

case1 = {"params": [{"shape": (1,8), "dtype": "float16", "format": "ND", "ori_shape": (1,8),"ori_format": "ND"},
                    {"shape": (1,4), "dtype": "float16", "format": "ND", "ori_shape": (1,4),"ori_format": "ND"},
                    {"shape": (1,8), "dtype": "float16", "format": "ND", "ori_shape": (1,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,8), "dtype": "float16", "format": "ND", "ori_shape": (2,8),"ori_format": "ND"},
                    {"shape": (2,4), "dtype": "float16", "format": "ND", "ori_shape": (2,4),"ori_format": "ND"},
                    {"shape": (2,8), "dtype": "float16", "format": "ND", "ori_shape": (2,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (7,8), "dtype": "float16", "format": "ND", "ori_shape": (7,8),"ori_format": "ND"},
                    {"shape": (7,4), "dtype": "float16", "format": "ND", "ori_shape": (7,4),"ori_format": "ND"},
                    {"shape": (7,8), "dtype": "float16", "format": "ND", "ori_shape": (7,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (16,8), "dtype": "float16", "format": "ND", "ori_shape": (16,8),"ori_format": "ND"},
                    {"shape": (16,4), "dtype": "float16", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16,8), "dtype": "float16", "format": "ND", "ori_shape": (16,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (777,8), "dtype": "float16", "format": "ND", "ori_shape": (777,8),"ori_format": "ND"},
                    {"shape": (777,4), "dtype": "float16", "format": "ND", "ori_shape": (777,4),"ori_format": "ND"},
                    {"shape": (777,8), "dtype": "float16", "format": "ND", "ori_shape": (777,8),"ori_format": "ND"}],
         "case_name": "DecodeWheelsTarget_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

err1 = {"params": [{"shape": (100,8), "dtype": "float32", "format": "ND", "ori_shape": (100,8),"ori_format": "ND"},
                   {"shape": (100,4), "dtype": "float32", "format": "ND", "ori_shape": (100,4),"ori_format": "ND"},
                   {"shape": (100,8), "dtype": "float32", "format": "ND", "ori_shape": (100,8),"ori_format": "ND"}],
        "case_name": "err_1",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True}
err2 = {"params": [{"shape": (800,), "dtype": "float16", "format": "ND", "ori_shape": (800,),"ori_format": "ND"},
                   {"shape": (400,), "dtype": "float16", "format": "ND", "ori_shape": (400,),"ori_format": "ND"},
                   {"shape": (800,), "dtype": "float16", "format": "ND", "ori_shape": (800,),"ori_format": "ND"}],
        "case_name": "err_2",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True}
err3 = {"params": [{"shape": (100, 6), "dtype": "float16", "format": "ND", "ori_shape": (100, 6),"ori_format": "ND"},
                   {"shape": (100, 4), "dtype": "float16", "format": "ND", "ori_shape": (100, 4),"ori_format": "ND"},
                   {"shape": (100, 6), "dtype": "float16", "format": "ND", "ori_shape": (100, 6),"ori_format": "ND"}],
        "case_name": "err_3",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True}

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)
ut_case.add_case("Ascend910", case3)
ut_case.add_case("Ascend910", case4)
ut_case.add_case("Ascend910", case5)
ut_case.add_case("Ascend910", err1)
ut_case.add_case("Ascend910", err2)
ut_case.add_case("Ascend910", err3)


if __name__ == '__main__':
    ut_case.run("Ascend910")



