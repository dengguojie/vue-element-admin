#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("InplaceUpdate", None, None)

case1 = {"params": [{"shape": (32, 5), "dtype": "int32", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "int32", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "int32", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"}],
         "case_name": "inplace_update_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32, 5), "dtype": "float16", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "float16", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"},
                    {"shape": (32, 5), "dtype": "float16", "format": "ND", "ori_shape": (32, 5),"ori_format": "ND"}],
         "case_name": "inplace_update_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (16,4), "dtype": "int32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16, ), "dtype": "int32", "format": "ND", "ori_shape": (16, ),"ori_format": "ND"},
                    {"shape": (16,4), "dtype": "int32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"},
                    {"shape": (16,4), "dtype": "int32", "format": "ND", "ori_shape": (16,4),"ori_format": "ND"}],
         "case_name": "inplace_update_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (4,16,16), "dtype": "int32", "format": "ND", "ori_shape": (4,16,16),"ori_format": "ND"},
                    {"shape": (4, ), "dtype": "int32", "format": "ND", "ori_shape": (4, ),"ori_format": "ND"},
                    {"shape": (4,16,16), "dtype": "int32", "format": "ND", "ori_shape": (4,16,16),"ori_format": "ND"},
                    {"shape": (4,16,16), "dtype": "int32", "format": "ND", "ori_shape": (4,16,16),"ori_format": "ND"}],
         "case_name": "inplace_update_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (32,16), "dtype": "int32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "int32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "int32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}],
         "case_name": "inplace_update_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"},
                    {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}],
         "case_name": "inplace_update_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)

def test_op_select_format(test_arg):
    from impl.inplace_update import check_supported
    check_supported(
        {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
        {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}
    )


ut_case.add_cust_test_func(test_func=test_op_select_format)


def test_check_supported(test_arg):
    from impl.inplace_update import check_supported
    x = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    indices = {"shape": (32, ), "dtype": "int32", "format": "ND", "ori_shape": (32, ),"ori_format": "ND"}
    v = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    y = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    supported, reason = check_supported(x, indices, v, y)
    assert supported

    x = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    indices = {"shape": (32, 32), "dtype": "int32", "format": "ND", "ori_shape": (32, 32),"ori_format": "ND"}
    v = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    y = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    supported, reason = check_supported(x, indices, v, y)
    assert not supported

    x = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    indices = {"shape": (32, 32), "dtype": "int32", "format": "ND", "ori_format": "ND"}
    v = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    y = {"shape": (32,16), "dtype": "float32", "format": "ND", "ori_shape": (32,16),"ori_format": "ND"}
    supported, reason = check_supported(x, indices, v, y)
    assert not supported


ut_case.add_cust_test_func(test_func=test_check_supported)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
