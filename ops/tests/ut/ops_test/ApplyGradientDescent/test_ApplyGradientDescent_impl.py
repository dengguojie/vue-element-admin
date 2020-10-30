
from op_test_frame.ut import OpUT
ut_case = OpUT("ApplyGradientDescent", None, None)

def gen_ApplyGradientDescent_case(shape, dtype, case_name_val, expect):
    return {"params": [{"shape": shape, "dtype": dtype, "ori_shape": shape, "ori_format": "ND", "format": "ND"},
                       {"shape": (1,), "dtype": dtype, "ori_shape": (1,), "ori_format": "ND", "format": "ND"},
                       {"shape": shape, "dtype": dtype, "ori_shape": shape, "ori_format": "ND", "format": "ND"},
                       {"shape": shape, "dtype": dtype, "ori_shape": shape, "ori_format": "ND", "format": "ND"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = gen_ApplyGradientDescent_case((1,), "float16", "case_1", "success")
case2 = gen_ApplyGradientDescent_case((1,), "float32", "case_2", "success")

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
