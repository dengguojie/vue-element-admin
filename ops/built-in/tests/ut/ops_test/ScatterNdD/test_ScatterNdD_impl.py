#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ScatterNdD", None, None)


def gen_data(ind_shape, updates_shape, output_y_shape, shape_out, update_dtype, ind_dtype, case_name, expect):
    indices = {"shape": ind_shape, "ori_shape": ind_shape, "format": "ND", "ori_format": "ND", "dtype": ind_dtype}
    updates = {"shape": updates_shape, "ori_shape": updates_shape, "format": "ND", "ori_format": "ND", "dtype": update_dtype}
    output_y = {"shape": output_y_shape, "ori_shape": output_y_shape, "format": "ND", "ori_format": "ND", "dtype": update_dtype}
    shape = shape_out

    return {"params": [indices, updates, output_y, shape],
            "case_name":case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_data((10241, 3), (10241, 3, 3), (3, 3, 3), (3, 3, 3), "float32", "int32", "scatter_nd_1", "success")
case2 = gen_data((4, ), (4,), (4,), (4,), "float32", "int32", "scatter_nd_2", "success")
case3 = gen_data((2, 3), (2, 3, 3), (3, 3, 3), (3, 3, 3), "float32", "int32", "scatter_nd_3", "success")
case4 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "float32", "int32", "scatter_nd_4", "success")
case5 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "float16", "int32", "scatter_nd_5", "success")
case6 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "int32", "int32", "scatter_nd_6", "success")
case7 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "int8", "int32", "scatter_nd_7", "success")
case8 = gen_data((2, 1), (2, 1, 1), (3, 1, 1), (3, 1, 1), "uint8", "int32", "scatter_nd_8", "success")

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)
ut_case.add_case("Ascend910A", case5)
ut_case.add_case("Ascend910A", case6)
ut_case.add_case("Ascend910A", case7)
ut_case.add_case("Ascend910A", case8)

from impl.scatter_nd_d import check_supported
def test_check_support(test_arg):
    check_supported({"shape": (-1,32,1), "dtype": "float16", "format": "ND", "ori_shape": (-1,32,1),"ori_format": "ND", "param_type": "indice"},
    {"shape": (1,32,100), "dtype": "float16", "format": "ND", "ori_shape": (1,32,100),"ori_format": "ND", "param_type": "x"},
    {"shape": (17, 28, 100), "dtype": "float16", "format": "ND", "ori_shape": (17, 28, 100),"ori_format": "ND", "param_type": "y"},
    [17 ,28, 100])
    check_supported({"shape": (-1,32,1), "dtype": "float16", "format": "ND", "ori_shape": (-1,32,1),"ori_format": "ND", "param_type": "indice"},
    {"shape": (1,32,1), "dtype": "float16", "format": "ND", "ori_shape": (1,32,1),"ori_format": "ND", "param_type": "x"},
    {"shape": (17, 28, 1), "dtype": "float16", "format": "ND", "ori_shape": (17, 28, 1),"ori_format": "ND", "param_type": "y"},
    [17 ,28, 1])
    check_supported({"shape": (1,32,2), "dtype": "float16", "format": "ND", "ori_shape": (1,32,2),"ori_format": "ND", "param_type": "indice"},
    {"shape": (1,32,100), "dtype": "float16", "format": "ND", "ori_shape": (1,32,100),"ori_format": "ND", "param_type": "x"},
    {"shape": (17, 28, 100), "dtype": "float16", "format": "ND", "ori_shape": (17, 28, 100),"ori_format": "ND", "param_type": "y"},
    [17 ,28, 100])
    check_supported({"shape": (1,32,2), "dtype": "float16", "format": "ND", "ori_shape": (1,32,2),"ori_format": "ND", "param_type": "indice"},
    {"shape": (1,32,100), "dtype": "int8", "format": "ND", "ori_shape": (1,32,100),"ori_format": "ND", "param_type": "x"},
    {"shape": (17, 28, 100), "dtype": "int8", "format": "ND", "ori_shape": (17, 28, 100),"ori_format": "ND", "param_type": "y"},
    [17 ,28, 100])

ut_case.add_cust_test_func(test_func=test_check_support)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
