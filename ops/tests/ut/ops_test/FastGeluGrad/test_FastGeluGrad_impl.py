from op_test_frame.ut import OpUT
ut_case = OpUT("FastGeluGrad", None, None)


case1 = {"params": [{"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND"}],
         "case_name": "FastGeluGrad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (2, 3), "dtype": "float16", "format": "ND", "ori_shape": (1, 3),"ori_format": "ND"},
                    {"shape": (1, 3), "dtype": "float16", "format": "ND", "ori_shape": (2, 3),"ori_format": "ND"}],
         "case_name": "FastGeluGrad_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)
