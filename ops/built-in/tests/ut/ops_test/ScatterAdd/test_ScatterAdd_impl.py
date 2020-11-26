#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ScatterAdd", None, None)


def gen_data(shape_var, shape_indices, shape_updates, dtype_var, dtype_indices, use_locking, case_name, expect):

    return {"params": [{"shape": shape_var, "ori_shape": shape_var, "ori_format":'ND', "format":'ND', "dtype": dtype_var},
                       {"shape": shape_indices, "ori_shape": shape_indices, "ori_format":'ND', "format":'ND', "dtype": dtype_indices},
                       {"shape": shape_updates, "ori_shape": shape_updates, "ori_format":'ND', "format":'ND', "dtype": dtype_var},
                       {"shape": shape_var, "ori_shape": shape_var, "ori_format":'ND', "format":'ND', "dtype": dtype_var},
                       use_locking],
            "case_name":case_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_data((255,8,33), (33,), (33,8,33), dtype_var="float32", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_1", expect="success")
case2 = gen_data((4,32,32), (2,), (2,32,32), dtype_var="int8", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_2", expect="success")
case3 = gen_data((4,32,32), (1,), (1,32,32), dtype_var="int8", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_3", expect="success")
case4 = gen_data((4,17,17), (2,), (2,17,17), dtype_var="uint8", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_4", expect="success")
case5 = gen_data((255,220,300), (33,), (33,220,300), dtype_var="float32", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_5", expect="success")
case6 = gen_data((255,220,300), (32,), (32,220,300), dtype_var="int8", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_6", expect="success")
case7 = gen_data((255, 33), (220, 300), (220, 300, 33), dtype_var="int8", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_7", expect="success")
case8 = gen_data((255, 33), (220,), (220, 33), dtype_var="float32", dtype_indices="int32",
                   use_locking=False, case_name ="scatter_add_8", expect="success")

ut_case.add_case("Ascend910", case1)
ut_case.add_case("Ascend910", case2)
ut_case.add_case("Ascend910", case3)
ut_case.add_case("Ascend910", case4)
ut_case.add_case("Ascend910", case5)
ut_case.add_case("Ascend910", case6)
ut_case.add_case("Ascend910", case7)
ut_case.add_case("Ascend910", case8)


if __name__ == '__main__':
    ut_case.run("Ascend910")
