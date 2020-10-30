from op_test_frame.ut import OpUT
ut_case = OpUT("AccumulateNv2", None, None)

case1 = {"params": [[{"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3),"ori_format": "NCHW"},
                     {"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3),"ori_format": "NCHW"}],
                    {"shape": (2, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 3),"ori_format": "NCHW"},
                    2],
         "case_name": "accumulate_nv2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)


if __name__ == '__main__':
    ut_case.run()
    exit(0)
