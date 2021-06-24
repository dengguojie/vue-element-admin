#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SpaceToBatchND", "impl.dynamic.space_to_batch_nd", "space_to_batch_nd")


def gen_dynamic_spacetobatchnd_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, in_format, ori_format, dtype_val, kernel_name_val, shape_block, shape_pads, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": in_format, "range": range_x},
                       {"shape": shape_block, "dtype": "int32", "ori_shape": shape_block, "ori_format": "ND", "format": "ND", "range": ((1, None),)},
                       {"shape": shape_pads, "dtype": "int32", "ori_shape": shape_pads, "ori_format": "ND", "format": "ND", "range": ((1, None),(1, None))},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape": ori_shape_y, "ori_format": ori_format, "format": in_format, "range": range_y},],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                           ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                           "NC1HWC0","NHWC","float16","batchtospace_case",(2,),(2,2),"success"))
ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-1,-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),
                                                 ((1, None),(1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None),(1, None)),
                                                 "NDC1HWC0","NDHWC","float32","batchtospace_case",(3,),(3,2),"success"))
ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                               ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                               "ND","NHWC","float16","batchtospace_case",(2,),(2,2),RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-2,),(-2,),(-2,),(-2,),
                                           ((1, None),),((1, None),),
                                           "NC1HWC0","NHWC","float16","batchtospace_case",(2,),(2,2),"success"))


def test_op_check_supported_0(test_arg):
    from impl.dynamic.space_to_batch_nd import check_supported
    x = {'ori_shape': (1, 1, 1, 1), 'shape': (1, 1, 1, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    block_shape = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    paddings = {'ori_shape': (2, 2), 'shape': (2, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    y = {'ori_shape': (1, 1, 1, 1), 'shape': (1, 1, 1, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(x, block_shape, paddings, y) == False:
        raise Exception("Failed to call check_supported in Batch_to_space_nd.")


def test_op_check_supported_1(test_arg):
    from impl.dynamic.space_to_batch_nd import check_supported
    x = {'ori_shape': (1, 1, 1, 1), 'shape': (1, 1, 1, 1), 'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    block_shape = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    paddings = {'ori_shape': (2, 2), 'shape': (2, 2), 'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    y = {'ori_shape': (1, 1, 1, 1), 'shape': (1, 1, 1, 1), 'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    if check_supported(x, block_shape, paddings, y) == False:
        raise Exception("Failed to call check_supported in Batch_to_space_nd.")


def test_op_check_supported_2(test_arg):
    from impl.dynamic.space_to_batch_nd import check_supported
    x = {'ori_shape': (1, 1, 1), 'shape': (1, 1, 1), 'ori_format': 'NCHW', 'format': 'NCHW', 'dtype': 'float16'}
    block_shape = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NCHW', 'format': 'NCHW', 'dtype': 'float16'}
    paddings = {'ori_shape': (2, 2), 'shape': (2, 2), 'ori_format': 'NCHW', 'format': 'NCHW', 'dtype': 'float16'}
    y = {'ori_shape': (1, 1, 1), 'shape': (1, 1, 1), 'ori_format': 'NCHW', 'format': 'NCHW', 'dtype': 'float16'}
    if check_supported(x, block_shape, paddings, y) == False:
        raise Exception("Failed to call check_supported in Batch_to_space_nd.")


def test_op_check_supported_3(test_arg):
    from impl.dynamic.space_to_batch_nd import check_supported
    x = {'ori_shape': (1, 1, 1), 'shape': (1, 1, 1), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    block_shape = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    paddings = {'ori_shape': (2, 2), 'shape': (2, 2), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    y = {'ori_shape': (1, 1, 1), 'shape': (1, 1, 1), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    if check_supported(x, block_shape, paddings, y) == False:
        raise Exception("Failed to call check_supported in Batch_to_space_nd.")


def test_op_select_format(test_arg):
    from impl.dynamic.space_to_batch_nd import op_select_format
    op_select_format({"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 16, 16, 16),"ori_format": "NHWC", "param_type": "input"},
                    {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2,),"ori_format": "ND", "param_type": "input"},
                    {"shape": (2, 2), "dtype": "float16", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND", "param_type": "input"},
                    {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (16, 16, 16, 16),"ori_format": "NHWC", "param_type": "input"},
                    )

ut_case.add_cust_test_func(test_func=test_op_select_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported_0)
ut_case.add_cust_test_func(test_func=test_op_check_supported_1)
ut_case.add_cust_test_func(test_func=test_op_check_supported_2)
ut_case.add_cust_test_func(test_func=test_op_check_supported_3)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
