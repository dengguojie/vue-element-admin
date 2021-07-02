#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("BatchToSpace", "impl.dynamic.batch_to_space", "batch_to_space")

def test_get_op_support_info_1(test_arg):
    from impl.dynamic.batch_to_space import get_op_support_info
    get_op_support_info({"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 16, 16, 16),"ori_format": "NHWC", "param_type": "input"},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "input"},
                    {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 16, 16, 16),"ori_format": "NHWC", "param_type": "input"},
                    )

def test_get_op_support_info_2(test_arg):
    from impl.dynamic.batch_to_space import get_op_support_info
    get_op_support_info({"shape": (16, 1, 16, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 16, 16),"ori_format": "NHWC", "param_type": "input"},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "input"},
                    {"shape": (16, 1, 16, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 16, 16, 16),"ori_format": "NHWC", "param_type": "input"},
                    )


ut_case.add_cust_test_func(test_func=test_get_op_support_info_1)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_2)

def gen_dynamic_batchtospace_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, in_format, ori_format, dtype_val, kernel_name_val, block_size, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": in_format, "range": range_x},
                       {"shape": (2,2), "dtype": "int32", "ori_shape": (2,2), "ori_format": "ND", "format": "ND", "range": ((1, None),(1, None))},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape": ori_shape_y, "ori_format": ori_format, "format": in_format, "range": range_y},
                       block_size,],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_batchtospace_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                           ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                           "NC1HWC0","NHWC","float16","batchtospace_case",2,"success"))
ut_case.add_case("all",
                 gen_dynamic_batchtospace_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                               ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                               "ND","NHWC","float16","batchtospace_case",2,RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_batchtospace_case((-2,),(-2,),(-2,),(-2,),
                                           ((1, None),),((1, None),),
                                           "NC1HWC0","NHWC","float16","batchtospace_case",2,"success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
