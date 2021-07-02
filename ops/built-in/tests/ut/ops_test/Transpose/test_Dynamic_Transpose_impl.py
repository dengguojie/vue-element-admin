#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import time
from impl.util.platform_adapter import tbe_context
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(linewidth=10000)

ut_case = OpUT("Transpose", "impl.dynamic.transpose", "transpose")

def calc_expect_func(x, perm, actual):
    x_val = x.get("value")
    p_val = perm.get("value")
    expect = np.transpose(x_val, p_val)
    print("------------------actual---------------------")
    #print(actual.get("value"))
    print("------------------expect---------------------")
    #print(expect)
    return (expect,)

def gen_transpose_case(dynamic_input_shapes, ori_input_shapes, dtype, perm_shape, case_name_val, expect, input_format="ND"):
    inputs = (
        {"shape": dynamic_input_shapes, "dtype": dtype, "ori_shape": ori_input_shapes, "ori_format": input_format, "format": input_format,
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
    )
    perm = {"dtype": dtype, "orig_shape": ori_input_shapes, "shape":perm_shape}
    return {"params": [inputs[0], perm, inputs[0]], "case_name": case_name_val, "expect": expect, "support_expect": True}

#ut_case.add_case(["Ascend910A", "Ascend310"], gen_transpose_case((-1, -1), (66, 2), "float32", (0, 1), "case_1", "success"))

def test_op_check_supported(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (-1, -1), 'shape': (2, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (-1,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (-1, -1), 'shape': (3, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == False:
        raise Exception("Failed to call check_supported in Transpose.")

def test_op_check_supported(test_arg):
    from impl.dynamic.transpose import _by_dynamic_static_union_version
    if _by_dynamic_static_union_version((1, 24, 3, 20), 1) == False:
        raise Exception("Failed to call check_supported in Transpose.")

def test_op_check_supported_in_white_list_return_false(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == False:
        raise Exception("1024,1024,fp16 in white list, should return True, then call transpose instead of transpose_d")

def test_op_check_supported_not_in_white_list_return_true(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (1234, 4321), 'shape': (1234, 4321), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (4321, 1234), 'shape': (4321, 1234), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == True:
        raise Exception("1234,4321,fp16 not in white list, should return False")

def test_op_check_supported_dtype_not_in_white_list_return_true(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'int8'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'int8'}
    if check_supported(input_x, perm, output_y) == True:
        raise Exception("1024,1024,int8 not in white list, should return False")

def test_op_check_cpu_false(test_arg):
    from impl.dynamic.transpose import by_dynamic_static_union_version 
    input_x = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if by_dynamic_static_union_version(input_x, 1) == True:
        raise Exception("lhisi not support, should return False")

def test_op_check_supported_not_in_white_list_but_fuzzily_build(test_arg):
    from impl.dynamic.transpose import check_supported
    import tbe
    input_x = {'ori_shape': (1234, 4321), 'shape': (1234, 4321), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (4321, 1234), 'shape': (4321, 1234), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    with tbe.common.context.op_context.OpContext("transpose_ut"):
        tbe_context.get_context().set_build_type("fuzzily_build")
        if check_supported(input_x, perm, output_y) == False:
            raise Exception("fuzzily build, should return True")
			
def test_op_check_supported_in_white_list_fuzzy_match_return_true_1(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (128, 12, 197, 64), 'shape': (128, 12, 197, 64), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (128, 12, 197, 64), 'shape': (128, 12, 197, 64), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == False:
        raise Exception("128, 12, 197, 64,fp16 in fuzzy white list, should return True")

def test_op_check_supported_in_white_list_fuzzy_match_return_true_2(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (768, 12, 197), 'shape': (768, 12, 197), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    perm = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (197, 12, 768), 'shape': (197, 12, 768), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == False:
        raise Exception("768, 12, 197,fp16 in fuzzy white list, should return True")

def test_op_check_supported_in_white_list_fuzzy_match_return_false_3(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (768, 12, 997), 'shape': (768, 12, 997), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    perm = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (997, 12, 768), 'shape': (997, 12, 768), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    if check_supported(input_x, perm, output_y) == True:
        raise Exception("768, 12, 997,fp16 not in fuzzy white list, should return False")


ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_return_false)
ut_case.add_cust_test_func(test_func=test_op_check_supported_not_in_white_list_return_true)
ut_case.add_cust_test_func(test_func=test_op_check_supported_dtype_not_in_white_list_return_true)
ut_case.add_cust_test_func(test_func=test_op_check_supported_not_in_white_list_but_fuzzily_build)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_fuzzy_match_return_true_1)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_fuzzy_match_return_true_2)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_fuzzy_match_return_false_3)


def add_ts_case(soc, d_type, x, perm, y, value_type="default"):
    if d_type == "bool":
        d_type ="uint8"
    perm = np.array(perm)
    x_dynamic = x
    x_dynamic = (-1,) + x[1:]
    print(x_dynamic)
    x_range = []
    y_range = []
    p_shape = len(perm)
    vol = 1

    for i in x:
        x_range.append((i,i))

    for i in y:
        y_range.append((i,i))

    for i in x:
        vol = vol * i

    if value_type == "arange":
        value = np.arange(0, vol, dtype="int32").reshape(x)
    else:
        value = np.random.randint(100, size=vol, dtype="int32").astype(d_type).reshape(x)

    ut_case.add_precision_case(["Ascend910A"],
                               {
                                   "params":
                                       [
                                           {
                                               "shape": x_dynamic,
                                               "dtype": d_type,
                                               "format": "ND",
                                               "ori_shape": x,
                                               "range": x_range,
                                               "run_shape": x,
                                               "ori_format": "ND",
                                               "param_type": "input",
                                               "value": value
                                           },
                                           {
                                               "shape": (p_shape,),
                                               "run_shape": (p_shape,),
                                               "dtype": "int32",
                                               "ori_shape": (p_shape),
                                               "ori_format" : "ND",
                                               "format": "ND",
                                               "value": perm,
                                               "value_need_in_tiling": True,
                                               "param_type": "input"
                                           },
                                           {
                                               "shape": y,
                                               "dtype": d_type,
                                               "format": "ND",
                                               "ori_shape": y,
                                               "range": y_range,
                                               "run_shape": y,
                                               "ori_format": "ND",
                                               "param_type": "output"
                                           },
                                       ],
                                   "calc_expect_func": calc_expect_func,
                                   #"case_name": "case_" + str(os.getpid())+ "_" + str((int)(time.time())),
                                   "precision_standard": precision_info.PrecisionStandard(0, 0)
                               })

#------------------------------------------------------------------------------------------------
#
#                                          ok
#
#------------------------------------------------------------------------------------------------
add_ts_case(["Ascend910A"], "uint8",   (33, 200),                  (1, 0),                 (200, 33),               "random")
add_ts_case(["Ascend910A"], "float32", (1000, 2000),               (1, 0),                 (2000, 1000),            "random")
add_ts_case(["Ascend910A"], "float32", (2, 3, 4, 500, 601),        (0, 2, 1, 3, 4),        (2, 4, 3, 500, 601),     "random")
add_ts_case(["Ascend910A"], "float32", (2, 3, 4, 5, 6),            (0, 2, 1, 3, 4),        (2, 4, 3, 5, 6),         "random")
add_ts_case(["Ascend910A"], "float32", (20000, 3),                 (1, 0),                 (3, 20000),              "random")
add_ts_case(["Ascend910A"], "float32", (3, 20000),                 (1, 0),                 (20000, 3),              "random")
add_ts_case(["Ascend910A"], "float16", (20000, 3),                 (1, 0),                 (3, 20000),              "random")
add_ts_case(["Ascend910A"], "float16", (3, 20000),                 (1, 0),                 (20000, 3),              "random")
add_ts_case(["Ascend910A"], "int32",   (95, 32, 96, 8),            (2, 1, 0, 3,),          (96, 32, 95, 8),         "arange")
add_ts_case(["Ascend910A"], "int32",   (9, 32, 5, 30, 3, 8),       (3, 1, 4, 2, 0, 5),     (30, 32, 3, 5, 9, 8),    "arange")
add_ts_case(["Ascend910A"], "int32",   (9, 32, 5, 9, 5, 8),        (4, 3, 1, 0, 2, 5),     (5, 9, 32, 9, 5, 8),     "arange")
add_ts_case(["Ascend910A"], "int32",   (32, 9, 5, 9, 5, 8),        (0, 4, 3, 2, 1, 5),     (32, 5, 9, 5, 9, 8),     "arange")
add_ts_case(["Ascend910A"], "int32",   (100, 70, 9, 8),            (0, 2, 1, 3),           (100, 9, 70, 8),         "arange")
add_ts_case(["Ascend910A"], "int32",   (70, 9, 100, 8),            (2, 1, 0, 3),           (100, 9, 70, 8),         "arange")
add_ts_case(["Ascend910A"], "int32",   (32, 10, 30, 3, 8),         (0, 3, 2, 1, 4),        (32, 3, 30, 10, 8),      "arange")
add_ts_case(["Ascend910A"], "int32",   (32, 10, 37, 3, 8),         (3, 0, 2, 1, 4),        (3, 32, 37, 10, 8),      "arange")
add_ts_case(["Ascend910A"], "int16",   (16, 4, 6410),              (1, 0, 2),              (4, 16, 6410),           "random")

#add_ts_case(["Ascend910A"], "int32",   (9, 32, 11, 31, 3, 8),      (3, 1, 4, 2, 0, 5),     (31, 32, 3, 11, 9, 8),   "arange")
#add_ts_case(["Ascend910A"], "int32",   (6, 32, 9, 31, 3, 8),       (3, 1, 4, 0, 2, 5),     (31, 32, 3, 6, 9, 8),    "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 2, 50, 36, 5, 8),     (3, 0, 4, 2, 1, 5),     (36, 32, 5, 50, 2, 8),   "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 2, 50, 40, 5, 8),     (3, 0, 4, 2, 1, 5),     (40, 32, 5, 50, 2, 8),   "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 50, 4, 8, 40, 5, 8),   (4, 3, 5, 2, 1, 0, 6),  (40, 8, 5, 4, 50, 2, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 2, 22, 13, 5, 8),     (3, 0, 4, 2, 1, 5),     (13, 32, 5, 22, 2, 8),   "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 4, 50, 8, 40, 5, 8),   (4, 3, 5, 1, 2, 0, 6),  (40, 8, 5, 4, 50, 2, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 4, 50, 8, 40, 5, 3),   (4, 3, 5, 1, 2, 0, 6),  (40, 8, 5, 4, 50, 2, 3), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 4, 50, 8, 40, 5, 8),   (5, 3, 4, 1, 2, 0, 6),  (5, 8, 40, 4, 50, 2, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (50, 4, 2, 8, 40, 5, 8),   (5, 3, 4, 1, 0, 2, 6),  (5, 8, 40, 4, 50, 2, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 4, 51, 33, 8),      (2, 3, 0, 4, 1, 5),     (4, 51, 2, 33, 3, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 4, 51, 33, 8),      (2, 3, 1, 4, 0, 5),     (4, 51, 3, 33, 2, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 33, 4, 51, 8),      (4, 0, 1, 3, 2, 5),     (51, 2, 3, 4, 33, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 33, 51, 4, 8),      (1, 3, 0, 4, 2, 5),     (3, 51, 2, 4, 33, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 33, 3, 51, 4, 8),      (0, 2, 4, 1, 3, 5),     (2, 3, 4, 33, 51, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (33, 32, 51, 4, 8),        (1, 3, 0, 2, 4),     (32, 4, 33, 51, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (10, 90, 5, 8),            (2, 1, 0, 3),           (5, 90, 10, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (6, 10, 7, 20, 5, 8),            (2, 3, 0, 4, 1, 5),           (7, 20, 6, 5, 10, 8), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 10000, 8),             (1, 0, 2),              (10000, 2, 8),           "arange")
#add_ts_case(["Ascend910A"], "int32", (10000, 2, 8),            (1, 0, 2),              (2, 10000, 8),          "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 100, 20, 5, 8),        (2, 0, 3, 1, 4),        (20, 2, 5, 100, 8),              "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 100, 20, 5, 8),        (3, 0, 2, 1, 4),        (5, 2, 20, 100, 8),              "arange")
#add_ts_case(["Ascend910A"], "int32", (5, 2, 20, 100, 8),        (3, 1, 0, 2, 4),        (100, 2, 5, 20, 8),              "arange")
#add_ts_case(["Ascend910A"], "int32", (5, 2, 20, 100, 8),        (3, 1, 2, 0, 4),        (100, 2, 20, 5, 8),              "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 10, 197, 3, 8),        (3, 0, 2, 1, 4),        (3, 32, 197, 10, 8),      "arange")
#add_ts_case(["Ascend910A"], "int32", (25, 32, 25, 3),           (2, 1, 0, 3),           (25, 32, 25, 3),         "arange")
#add_ts_case(["Ascend910A"], "int32", (25, 32, 29, 3),           (2, 1, 0, 3),           (29, 32, 25, 3),         "arange")
#add_ts_case(["Ascend910A"], "int32", (29, 32, 25, 3),           (2, 1, 0, 3),           (25, 32, 29, 3),         "arange")
#add_ts_case(["Ascend910A"], "int32", (70, 9, 101, 3),           (2, 1, 0, 3),           (101, 9, 70, 3),         "arange")
#add_ts_case(["Ascend910A"], "int32", (26, 32, 25, 7),           (2, 1, 0, 3),           (25, 32, 26, 7),         "arange")
#add_ts_case(["Ascend910A"], "int32", (26, 32, 25, 3),           (2, 1, 0, 3),           (25, 32, 26, 3),         "arange")
#add_ts_case(["Ascend910A"], "int32", (26, 32, 27, 3),           (2, 1, 0, 3),           (27, 32, 26, 3),         "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 10, 37, 3, 3),        (3, 0, 2, 1, 4),        (3, 32, 37, 10, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 10, 37, 3, 3),        (2, 0, 3, 1, 4),        (37, 32, 3, 10, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 7, 37, 10, 3),        (2, 0, 3, 1, 4),        (37, 32, 10, 7, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 20196, 3),        (1, 0, 2),        (20196, 2, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 51, 33, 4, 3, 3),        (2, 0, 4, 1, 3, 5),        (33, 2, 3, 51, 4, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (20196, 2, 3),        (1, 0, 2),        (2, 20196, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (66, 153, 4, 3),        (0, 2, 1, 3),        (66, 4, 153, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 4, 51, 2, 33, 3),        (4, 2, 3, 0, 1, 5),        (33, 51, 2, 3, 4, 3),      "arange")
#add_ts_case(["Ascend910A"], "int32", (51, 4, 33, 2, 3, 5),        (1, 4, 0, 2, 3, 5),        (4, 3, 51, 33, 2, 5),      "arange")
#add_ts_case(["Ascend910A"], "float16", (2, 12, 51, 33, 15),        (1, 3, 2, 0, 4),        (12, 33, 51, 2, 15),      "arange")
#add_ts_case(["Ascend910A"], "float16", (10000, 20, 100),        (1, 0, 2),        (20, 10000, 100),      "arange")
#add_ts_case(["Ascend910A"], "float16", (1, 128, 12, 26),        (0, 2, 1, 3),        (1, 12, 128, 26),      "arange")
#add_ts_case(["Ascend910A"], "float16", (2, 50, 7, 9, 40, 5, 8),        (4, 3, 5, 2, 1, 0, 6),        (40, 9, 5, 7, 50, 2, 8),      "random")
#add_ts_case(["Ascend910A"], "int8", (3, 19, 2, 3976),        (0, 2, 1, 3),        (3, 2, 19, 3976),      "random")
#add_ts_case(["Ascend910A"], "int16", (2, 17, 2, 4, 5, 8),        (3, 2, 4, 1, 0, 5),        (4, 2, 5, 17, 2, 8),      "random")
#add_ts_case(["Ascend910A"], "float16", (2048, 10, 512),        (1, 0, 2),        (10, 2048, 512),      "random")
#add_ts_case(["Ascend920A"], "int16",   (4, 255, 3, 8),             (2, 1, 0, 3),           (2, 255, 4, 8),          "random")

#arr = [2, 3, 4, 33, 51, 32]
#import itertools
#import random
#pmu = list(itertools.permutations([0,1,2,3,4]))
#perm = list(itertools.permutations([0,1,2,3,4]))
#count = 0
#case_num = 0
#for i in pmu:
#    j = list(i)
#    j.append(5)
#    for k in perm:
#        p = list(k)
#        if (p[4] == 4):
#            continue
#        p.append(5)
#        x = (arr[j[0]], arr[j[1]], arr[j[2]], arr[j[3]], arr[j[4]], arr[j[5]])
#        count = count + 1
#        y = (x[p[0]], x[p[1]], x[p[2]], x[p[3]], x[p[4]], x[p[5]]) 
#        r = random.randint(0,100)
#        if r == 99:
#            case_num = case_num + 1
#            print("add case:", case_num, ", x:", x, ", perm :", p, ", y :", y)
#            add_ts_case(["Ascend910A"], "int32", x,  p, y, "arange")

#arr = [2, 3, 4, 33, 51, 35]
#import itertools
#import random
#pmu = list(itertools.permutations([0,1,2,3,4]))
#perm = list(itertools.permutations([0,1,2,3,4]))
#count = 0
#case_num = 0
#for i in pmu:
#    j = list(i)
#    j.append(5)
#    for k in perm:
#        p = list(k)
#        if (p[4] == 4):
#            continue
#        p.append(5)
#        x = (arr[j[0]], arr[j[1]], arr[j[2]], arr[j[3]], arr[j[4]], arr[j[5]])
#        count = count + 1
#        y = (x[p[0]], x[p[1]], x[p[2]], x[p[3]], x[p[4]], x[p[5]]) 
#        r = random.randint(0,100)
#        if r == 99:
#            case_num = case_num + 1
#            print("add case:", case_num, ", x:", x, ", perm :", p, ", y :", y)
#            add_ts_case(["Ascend910A"], "float16", x,  p, y, "arange")

#def test_transpose_920a(test_arg):
#    from impl.dynamic.transpose import transpose 
#    from te import platform as cce_conf
#    cce_conf.cce_conf.te_set_version("Ascend920A", core_type="VectorCore")
#    transpose(
#                {
#                    "shape": (4, 255, 3, 8),
#                    "dtype": "int32",
#                    "format": "ND",
#                    "ori_shape": (-1, 255, 3, 8),
#                    "range": ((4, 4), (255, 255), (3, 3), (8, 8),),
#                    "run_shape": (4, 255,3, 8),
#                    "ori_format": "ND",
#                    "param_type": "input",
#                    #"value": np.arange(0, 256*8*4*20*16*16, dtype="int32").reshape(256, 8, 4, 20, 16, 16)
#                },
#                {
#                    "shape": (4,),
#                    "run_shape": (4,),
#                    "dtype": "int32",
#                    "ori_shape": (4),
#                    "ori_format" : "ND",
#                    "format": "ND",
#                    "value": np.array([2, 1, 0, 3]),
#                    "value_need_in_tiling": True,
#                    "param_type": "input"
#                },
#                {
#                    "shape": (3, 255, 4, 8),
#                    "dtype": "int32",
#                    "format": "ND",
#                    "ori_shape": (3, 255, 4, 8),
#                    "range": ((3, 3), (255, 255), (4, 4), (8, 8), ),
#                    "run_shape": (3, 255, 4, 8),
#                    "ori_format": "ND",
#                    "param_type": "output"
#                },
#              )
#    cce_conf.cce_conf.te_set_version(test_arg)
#
#ut_case.add_cust_test_func(test_func=test_transpose_920a)

if __name__ == '__main__':
    simulator_lib_path = "/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend920A"], simulator_mode="esl", simulator_lib_path=simulator_lib_path)

