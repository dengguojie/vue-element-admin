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
    res, _ = check_supported(input_x, perm, output_y)
    if res:
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
    res, _ = check_supported(input_x, perm, output_y)
    if not res:
        raise Exception("1024,1024,fp16 in white list, should return True, then call transpose instead of transpose_d")

def test_op_check_supported_dtype_not_in_white_list_return_true(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'int8'}
    perm = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (1024, 1024), 'shape': (1024, 1024), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'int8'}
    res, _ = check_supported(input_x, perm, output_y)
    if not res:
        raise Exception("1024,1024,int8 in white list, should return True")

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
        res, _ = check_supported(input_x, perm, output_y)
        if not res:
            raise Exception("fuzzily build, should return True")
			
def test_op_check_supported_in_white_list_fuzzy_match_return_true_1(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (128, 12, 197, 64), 'shape': (128, 12, 197, 64), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    perm = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (128, 12, 197, 64), 'shape': (128, 12, 197, 64), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    res, _ = check_supported(input_x, perm, output_y)
    if not res:
        raise Exception("128, 12, 197, 64,fp16 in fuzzy white list, should return True")

def test_op_check_supported_in_white_list_fuzzy_match_return_true_2(test_arg):
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (768, 12, 197), 'shape': (768, 12, 197), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    perm = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (197, 12, 768), 'shape': (197, 12, 768), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    res, _ = check_supported(input_x, perm, output_y)
    if not res:
        raise Exception("768, 12, 197,fp16 in fuzzy white list, should return True")

def test_get_ub_core_for_cov(test_arg):
    from impl.dynamic.transpose import get_ub_size 
    from impl.dynamic.transpose import get_core_num
    from impl.dynamic.transpose import _static_scenario_goto_old_version 
    from impl.dynamic.transpose import check_supported
    get_ub_size()
    get_core_num()
    shape_hit = [1, 128, 128, 3]
    shape_miss = [2, 128, 128, 3]
    _static_scenario_goto_old_version(shape_hit,  2)
    _static_scenario_goto_old_version(shape_miss, 2)
    input_x = {'ori_shape': (1, 128, 128, 3), 'shape': (1, 128, 128, 3), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    perm = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (1, 3, 18, 128), 'shape': (1, 3, 128, 128), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}

    res, _ = check_supported(input_x, perm, output_y)
    if res:
        raise Exception("1, 128, 128, 3 is in black list, should return False")
    

ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_return_false)
ut_case.add_cust_test_func(test_func=test_op_check_supported_dtype_not_in_white_list_return_true)
ut_case.add_cust_test_func(test_func=test_op_check_supported_not_in_white_list_but_fuzzily_build)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_fuzzy_match_return_true_1)
ut_case.add_cust_test_func(test_func=test_op_check_supported_in_white_list_fuzzy_match_return_true_2)
ut_case.add_cust_test_func(test_func=test_get_ub_core_for_cov)


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
        value = np.arange(0, vol, dtype="int32").astype(d_type).reshape(x)
    else:
        value = np.random.randint(100, size=vol, dtype="int32").astype(d_type).reshape(x)

    ut_case.add_precision_case(soc,
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
add_ts_case(["Ascend910A"], "float16",   (1024, 12, 30, 26),           (0, 2, 1, 3),           (1024, 30,12,26),        "random")
#add_ts_case(["Ascend910A"], "int8",    (3, 1900, 2, 13),           (0, 2, 1, 3),           (3, 2, 1900, 13),        "arange")
#add_ts_case(["Ascend910A"], "int16",    (3, 1900, 2, 13),           (0, 2, 1, 3),           (3, 2, 1900, 13),        "arange")
#add_ts_case(["Ascend910A"], "uint8",   (8, 16),                    (1, 0),                 (16, 8),                 "random")
#add_ts_case(["Ascend910A"], "uint8",   (10,3,4,2,3,4),             (2, 1, 0, 5, 4, 3),     (4, 3, 10, 4, 3, 2),     "random")
#add_ts_case(["Ascend910A"], "float32", (1000, 2000),               (1, 0),                 (2000, 1000),            "random")
#add_ts_case(["Ascend910A"], "float16", (3, 20000),                 (1, 0),                 (20000, 3),              "random")
#add_ts_case(["Ascend910A"], "int64",   (2, 3),                     (1, 0),                 (3, 2),                  "random")
#add_ts_case(["Ascend910A"], "int8",    (56, 56, 3),                (2, 1, 0),              (3, 56, 56),             "random")
#add_ts_case(["Ascend910A"], "int8",    (50, 56, 21, 3),            (0, 3, 2, 1),           (50, 3, 21, 56),         "arange")
#add_ts_case(["Ascend910A"], "uint8",   (33, 200),                  (1, 0),                 (200, 33),               "random")
#add_ts_case(["Ascend910A"], "int8",    (3, 19, 2, 3976),           (0, 2, 1, 3),           (3, 2, 19, 3976),        "random")
#add_ts_case(["Ascend910A"], "int8",    (3, 2, 320000),             (1, 0, 2),              (2, 3, 320000),          "random")
#add_ts_case(["Ascend910A"], "int8",    (3, 2, 160000),             (1, 0, 2),              (2, 3, 160000),          "random")
#add_ts_case(["Ascend910A"], "int8",    (3, 2, 64000),              (1, 0, 2),              (2, 3, 64000),           "random")
#add_ts_case(["Ascend910A"], "int8",    (3, 2, 64001),              (1, 0, 2),              (2, 3, 64001),           "random")
#add_ts_case(["Ascend910A"], "int8",    (2, 3, 3876),               (1, 0, 2),              (3, 2, 3876),            "random")
#add_ts_case(["Ascend910A"], "float32", (2, 3, 4, 500, 601),        (0, 2, 1, 3, 4),        (2, 4, 3, 500, 601),     "random")
#add_ts_case(["Ascend910A"], "float32", (2, 3, 4, 5, 6),            (0, 2, 1, 3, 4),        (2, 4, 3, 5, 6),         "random")
#add_ts_case(["Ascend910A"], "float32", (20000, 3),                 (1, 0),                 (3, 20000),              "random")
#add_ts_case(["Ascend910A"], "float32", (3, 20000),                 (1, 0),                 (20000, 3),              "random")
#add_ts_case(["Ascend910A"], "float16", (20000, 3),                 (1, 0),                 (3, 20000),              "random")
#add_ts_case(["Ascend910A"], "float16", (3, 20000),                 (1, 0),                 (20000, 3),              "random")
#add_ts_case(["Ascend910A"], "int32",   (95, 32, 96, 8),            (2, 1, 0, 3,),          (96, 32, 95, 8),         "random")
#add_ts_case(["Ascend910A"], "int32",   (9, 32, 5, 30, 3, 8),       (3, 1, 4, 2, 0, 5),     (30, 32, 3, 5, 9, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (9, 32, 5, 9, 5, 8),        (4, 3, 1, 0, 2, 5),     (5, 9, 32, 9, 5, 8),     "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 9, 5, 9, 5, 8),        (0, 4, 3, 2, 1, 5),     (32, 5, 9, 5, 9, 8),     "random")
#add_ts_case(["Ascend910A"], "int32",   (100, 70, 9, 8),            (0, 2, 1, 3),           (100, 9, 70, 8),         "random")
#add_ts_case(["Ascend910A"], "int32",   (70, 9, 100, 8),            (2, 1, 0, 3),           (100, 9, 70, 8),         "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 10, 30, 3, 8),         (0, 3, 2, 1, 4),        (32, 3, 30, 10, 8),      "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 10, 37, 3, 8),         (3, 0, 2, 1, 4),        (3, 32, 37, 10, 8),      "random")
#add_ts_case(["Ascend910A"], "int16",   (16, 4, 6410),              (1, 0, 2),              (4, 16, 6410),           "random")
#add_ts_case(["Ascend910A"], "int16",   (2, 17, 2, 4, 5, 8),        (3, 2, 4, 1, 0, 5),     (4, 2, 5, 17, 2, 8),     "random")
#add_ts_case(["Ascend910A"], "int32",   (3, 20, 30, 40, 64),        (0, 3, 2, 1, 4),        (3, 40, 30, 20, 64),     "random")
#add_ts_case(["Ascend910A"], "int64",   (3, 2, 3, 40, 64),          (0, 3, 2, 1, 4),        (3, 40, 3, 2, 64),       "random")
#add_ts_case(["Ascend910A"], "int16",   (13824, 1536),              (1, 0),                 (1536, 13824),           "random")
#add_ts_case(["Ascend910A"], "int16",   (64, 32),                   (1, 0),                 (32, 64),                "random")
#add_ts_case(["Ascend910A"], "int16",   (64, 64),                   (1, 0),                 (64, 64),                "random")
#add_ts_case(["Ascend910A"], "int16",   (4, 256, 160),              (0, 2, 1),              (4, 160, 256),           "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 128, 80),               (0, 2, 1),              (2, 80, 128),            "arange")
#add_ts_case(["Ascend910A"], "int32",   (2, 120, 80),               (0, 2, 1),              (2, 80, 120),            "arange")
#add_ts_case(["Ascend910A"], "int16",   (13824, 1536+16),           (1, 0),                 (1536+16, 13824),        "random")
#add_ts_case(["Ascend910A"], "int16",   (13824+16, 1536),           (1, 0),                 (1536, 13824+16),        "random")
#add_ts_case(["Ascend910A"], "int16",   (13824+16, 1536+16),        (1, 0),                 (1536+16, 13824+16),     "random")
#add_ts_case(["Ascend910A"], "int16",   (256, 1536),                (1, 0),                 (1536, 256),             "random")
#add_ts_case(["Ascend910A"], "int16",   (1536, 256),                (1, 0),                 (256, 1536),             "random")
#add_ts_case(["Ascend910A"], "int16",   (1536, 128),                (1, 0),                 (128, 1536),             "random")
#add_ts_case(["Ascend910A"], "int16",   (1536, 96),                 (1, 0),                 (96, 1536),              "random")
#add_ts_case(["Ascend910A"], "int16",   (96, 1536),                 (1, 0),                 (1536, 96),              "random")
#add_ts_case(["Ascend310"], "float16",  (2, 3, 123, 234),           (1, 0, 3, 2),           (3, 2, 234, 123),        "random")
#add_ts_case(["Ascend910A"], "int32",   (9, 32, 11, 31, 3, 8),      (3, 1, 4, 2, 0, 5),     (31, 32, 3, 11, 9, 8),   "random")
#add_ts_case(["Ascend910A"], "int32",   (6, 32, 9, 31, 3, 8),       (3, 1, 4, 0, 2, 5),     (31, 32, 3, 6, 9, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 2, 50, 36, 5, 8),      (3, 0, 4, 2, 1, 5),     (36, 32, 5, 50, 2, 8),   "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 2, 50, 40, 5, 8),      (3, 0, 4, 2, 1, 5),     (40, 32, 5, 50, 2, 8),   "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 50, 4, 8, 40, 5, 8),    (4, 3, 5, 2, 1, 0, 6),  (40, 8, 5, 4, 50, 2, 8), "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 2, 22, 13, 5, 8),      (3, 0, 4, 2, 1, 5),     (13, 32, 5, 22, 2, 8),   "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 4, 50, 8, 40, 5, 8),    (4, 3, 5, 1, 2, 0, 6),  (40, 8, 5, 4, 50, 2, 8), "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 4, 50, 8, 40, 5, 3),    (4, 3, 5, 1, 2, 0, 6),  (40, 8, 5, 4, 50, 2, 3), "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 4, 50, 8, 40, 5, 8),    (5, 3, 4, 1, 2, 0, 6),  (5, 8, 40, 4, 50, 2, 8), "random")
#add_ts_case(["Ascend910A"], "int32",   (50, 4, 2, 8, 40, 5, 8),    (5, 3, 4, 1, 0, 2, 6),  (5, 8, 40, 4, 50, 2, 8), "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 3, 4, 51, 33, 8),       (2, 3, 0, 4, 1, 5),     (4, 51, 2, 33, 3, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 3, 4, 51, 33, 8),       (2, 3, 1, 4, 0, 5),     (4, 51, 3, 33, 2, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 3, 33, 4, 51, 8),       (4, 0, 1, 3, 2, 5),     (51, 2, 3, 4, 33, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 3, 33, 51, 4, 8),       (1, 3, 0, 4, 2, 5),     (3, 51, 2, 4, 33, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 33, 3, 51, 4, 8),       (0, 2, 4, 1, 3, 5),     (2, 3, 4, 33, 51, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (33, 32, 51, 4, 8),         (1, 3, 0, 2, 4),        (32, 4, 33, 51, 8),      "random")
#add_ts_case(["Ascend910A"], "int32",   (10, 90, 5, 8),             (2, 1, 0, 3),           (5, 90, 10, 8),          "random")
#add_ts_case(["Ascend910A"], "int32",   (6, 10, 7, 20, 5, 8),       (2, 3, 0, 4, 1, 5),     (7, 20, 6, 5, 10, 8),    "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 10000, 8),              (1, 0, 2),              (10000, 2, 8),           "random")
#add_ts_case(["Ascend910A"], "int32",   (10000, 2, 8),              (1, 0, 2),              (2, 10000, 8),           "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 100, 20, 5, 8),         (2, 0, 3, 1, 4),        (20, 2, 5, 100, 8),      "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 100, 20, 5, 8),         (3, 0, 2, 1, 4),        (5, 2, 20, 100, 8),      "random")
#add_ts_case(["Ascend910A"], "int32",   (5, 2, 20, 100, 8),         (3, 1, 0, 2, 4),        (100, 2, 5, 20, 8),      "random")
#add_ts_case(["Ascend910A"], "int32",   (5, 2, 20, 100, 8),         (3, 1, 2, 0, 4),        (100, 2, 20, 5, 8),      "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 10, 197, 3, 8),        (3, 0, 2, 1, 4),        (3, 32, 197, 10, 8),     "random")
#add_ts_case(["Ascend910A"], "int32",   (25, 32, 25, 3),            (2, 1, 0, 3),           (25, 32, 25, 3),         "random")
#add_ts_case(["Ascend910A"], "int32",   (25, 32, 29, 3),            (2, 1, 0, 3),           (29, 32, 25, 3),         "random")
#add_ts_case(["Ascend910A"], "int32",   (29, 32, 25, 3),            (2, 1, 0, 3),           (25, 32, 29, 3),         "random")
#add_ts_case(["Ascend910A"], "int32",   (70, 9, 101, 3),            (2, 1, 0, 3),           (101, 9, 70, 3),         "random")
#add_ts_case(["Ascend910A"], "int32",   (26, 32, 25, 7),            (2, 1, 0, 3),           (25, 32, 26, 7),         "random")
#add_ts_case(["Ascend910A"], "int32",   (26, 32, 25, 3),            (2, 1, 0, 3),           (25, 32, 26, 3),         "random")
#add_ts_case(["Ascend910A"], "int32",   (26, 32, 27, 3),            (2, 1, 0, 3),           (27, 32, 26, 3),         "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 10, 37, 3, 3),         (3, 0, 2, 1, 4),        (3, 32, 37, 10, 3),      "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 10, 37, 3, 3),         (2, 0, 3, 1, 4),        (37, 32, 3, 10, 3),      "random")
#add_ts_case(["Ascend910A"], "int32",   (32, 7, 37, 10, 3),         (2, 0, 3, 1, 4),        (37, 32, 10, 7, 3),      "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 20196, 3),              (1, 0, 2),              (20196, 2, 3),           "random")
#add_ts_case(["Ascend910A"], "int32",   (2, 51, 33, 4, 3, 3),       (2, 0, 4, 1, 3, 5),     (33, 2, 3, 51, 4, 3),    "random")
#add_ts_case(["Ascend910A"], "int32",   (20196, 2, 3),              (1, 0, 2),              (2, 20196, 3),           "random")
#add_ts_case(["Ascend910A"], "int32",   (66, 153, 4, 3),            (0, 2, 1, 3),           (66, 4, 153, 3),         "random")
#add_ts_case(["Ascend910A"], "int32",   (3, 4, 51, 2, 33, 3),       (4, 2, 3, 0, 1, 5),     (33, 51, 2, 3, 4, 3),    "random")
#add_ts_case(["Ascend910A"], "int32",   (51, 4, 33, 2, 3, 5),       (1, 4, 0, 2, 3, 5),     (4, 3, 51, 33, 2, 5),    "random")
#add_ts_case(["Ascend910A"], "float16", (2, 12, 51, 33, 15),        (1, 3, 2, 0, 4),        (12, 33, 51, 2, 15),     "random")
#add_ts_case(["Ascend910A"], "float16", (2, 50, 4, 3, 40, 5, 8),    (4, 3, 5, 2, 1, 0, 6),  (20, 3, 5, 4, 50, 2, 8), "random")
#add_ts_case(["Ascend910A"], "float16", (2048, 10, 512),            (1, 0, 2),              (10, 2048, 512),         "random")
#add_ts_case(["Ascend910A"], "int16",   (4, 255, 3, 8),             (2, 1, 0, 3),           (2, 255, 4, 8),          "random")
#add_ts_case(["Ascend910A"], "float16", (1000, 20, 100),            (1, 0, 2),              (20, 1000, 100),         "random")
#add_ts_case(["Ascend910A"], "float16", (1, 128, 12, 26),           (0, 2, 1, 3),           (1, 12, 128, 26),        "random")
#add_ts_case(["Ascend910A"], "int16",   (2, 17, 2, 4, 5, 8),        (3, 2, 4, 1, 0, 5),     (4, 2, 5, 17, 2, 8),     "random")
#add_ts_case(["Ascend910A"], "int16",   (2, 100, 3, 7),             (2, 1, 0, 3),           (3, 100, 2, 7),          "arange")
#add_ts_case(["Ascend910A"], "int16",   (2, 200, 3, 7),             (2, 1, 0, 3),           (3, 200, 2, 7),          "arange")
#add_ts_case(["Ascend910A"], "int16",   (2, 400, 3, 16),            (2, 1, 0, 3),           (3, 400, 2, 16),         "arange")
#add_ts_case(["Ascend910A"], "int16",   (2, 77*5, 3, 7),            (2, 1, 0, 3),           (3, 77*5, 2, 7),         "arange")
#add_ts_case(["Ascend910A"], "int16",   (2, 3, 4, 33, 51, 35),      (2, 4, 0, 1, 3, 5),     (4, 51, 2, 3, 33, 35),   "random")
#add_ts_case(["Ascend910A"], "float16", (2, 3, 4, 33, 51, 35),      (2, 4, 0, 1, 3, 5),     (4, 51, 2, 3, 33, 35),   "random")
#add_ts_case(["Ascend910A"], "float16", (2, 51, 4, 33, 3, 7),       (0, 3, 4, 1, 2, 5),     (2, 33, 3, 51, 4, 7),    "random")
#add_ts_case(["Ascend910A"], "int32",   (300, 400),                 (1, 0),                 (400, 300),              "arange")
#add_ts_case(["Ascend910A"], "int32",   (2, 10000, 3),              (0, 2, 1),              (2, 3, 10000),           "arange")
#add_ts_case(["Ascend910A"], "int16",   (2, 3, 10000),              (0, 2, 1),              (2, 10000, 3),           "arange")
#add_ts_case(["Ascend910A"], "int16",   (2, 3, 1296),               (0, 2, 1),              (2, 1296, 3),            "arange")
#add_ts_case(["Ascend910A"], "int32",   (4, 400, 500),              (0, 2, 1),              (4, 500, 400),           "arange")
#add_ts_case(["Ascend910A"], "int16",   (190, 3, 2, 2),             (0, 2, 1, 3),           (190, 2, 3, 2),          "arange")
#add_ts_case(["Ascend910A"], "int16",   (25, 16, 16),               (0, 2, 1),              (25, 16, 16),            "arange")
#add_ts_case(["Ascend910A"], "int16",   (3, 7, 16, 16),             (1, 0, 3, 2),           (7, 3, 16, 16),          "arange")
#add_ts_case(["Ascend910A"], "int16",   (2, 4, 4, 16, 16),          (0, 2, 1, 4, 3),        (2, 4, 4, 16, 16),       "random")
#add_ts_case(["Ascend910A"], "int16",   (5000, 4, 4, 16, 16),       (0, 2, 1, 4, 3),        (5000, 4, 4, 16, 16),    "random")
#add_ts_case(["Ascend910A"], "int16",   (4, 32, 48),                (0, 2, 1),              (4, 48, 32),             "arange")
#add_ts_case(["Ascend910A"], "int16",   (4, 48, 16),                (0, 2, 1),              (4, 16, 48),             "arange")
#add_ts_case(["Ascend910A"], "int16",   (4, 48, 32),                (0, 2, 1),              (4, 32, 48),             "arange")
#add_ts_case(["Ascend910A"], "int16",   (5000, 4, 4, 16, 32),       (0, 2, 1, 4, 3),        (5000, 4, 4, 32, 16),    "random")
#add_ts_case(["Ascend910A"], "int16",   (16, 32),                   (1, 0),                 (32, 16),                "random")
#add_ts_case(["Ascend910A"], "int16",   (16, 32),                   (1, 0),                 (32, 16),                "random")
#add_ts_case(["Ascend910A"], "int16",   (2, 16, 32),                (0, 2, 1),              (2, 32, 16),             "random")
#add_ts_case(["Ascend910A"], "int64",   (256, 65),                  (1, 0),                 (65, 256),               "arange")
#add_ts_case(["Ascend910A"], "int64",   (256, 35),                  (1, 0),                 (35, 256),               "arange")
#add_ts_case(["Ascend910A"], "int64",   (195, 128),                 (1, 0),                 (128, 195),              "arange")
#add_ts_case(["Ascend910A"], "int64",   (512, 128),                 (1, 0),                 (128, 512),              "arange")
#add_ts_case(["Ascend910A"], "int32",   (256, 128),                 (1, 0),                 (128, 256),              "arange")
#add_ts_case(["Ascend910A"], "int16",   (256, 128),                 (1, 0),                 (128, 256),              "arange")
#add_ts_case(["Ascend910A"], "int32",   (256, 128),                 (1, 0),                 (128, 256),              "arange")
#add_ts_case(["Ascend910A"], "float16", (1, 32, 2, 2, 1, 7),        (0, 1, 2, 4, 3, 5),     (1, 32, 2, 1, 2, 7),     "random")
#add_ts_case(["Ascend910A"], "float16", (32, 16, 64),               (0, 2, 1),              (32, 64, 16),            "random")
#add_ts_case(["Ascend910A"], "float16", (1000, 5, 64, 64),          (0, 2, 1, 3),           (1000, 64, 5, 64),       "random")
#add_ts_case(["Ascend910A"], "float16", (2, 8, 1000, 64),           (0, 2, 1, 3),           (2, 1000, 8, 64),        "random")
#add_ts_case(["Ascend910A"], "float32", (2, 5, 8, 100, 64),         (1, 3, 2, 0, 4),        (5, 100, 8, 2, 64),      "random")
#add_ts_case(["Ascend910A"], "float16", (2, 1000, 8, 64),           (0, 2, 1, 3),           (2, 8, 1000, 64),        "random")
#add_ts_case(["Ascend910A"], "float16", (100, 200, 300),            (2, 1, 0),              (300, 200, 100),         "random")
#add_ts_case(["Ascend910A"], "float16", (20, 3, 4, 5, 6),           (0, 2, 1, 4, 3),        (20, 4, 3, 6, 5),        "random")
#add_ts_case(["Ascend910A"], "uint8",   (2000, 7, 32),              (0, 2, 1),              (2000, 32, 7),           "arange")
#add_ts_case(["Ascend910A"], "uint8",   (2000, 7, 31),              (0, 2, 1),              (2000, 31, 7),           "arange")
#add_ts_case(["Ascend910A"], "uint16",  (40, 50, 7),                (1, 0, 2),              (50, 40, 7),             "arange")
#add_ts_case(["Ascend910A"], "uint8",   (21, 21, 7),                (1, 0, 2),              (21, 21, 7),             "arange")
#add_ts_case(["Ascend910A"], "uint8",   (210, 210, 7),              (1, 0, 2),              (210, 210, 7),           "arange")
#add_ts_case(["Ascend910A"], "uint8",   (21, 42, 7),                (1, 0, 2),              (42, 21, 7),             "arange")
#add_ts_case(["Ascend910A"], "uint8",   (42, 42, 7),                (1, 0, 2),              (42, 42, 7),             "arange")
#add_ts_case(["Ascend910A"], "uint8",   (40, 50, 7, 32),            (1, 0, 3, 2),           (50, 40, 32, 7),         "arange")
#add_ts_case(["Ascend910A"], "uint8",   (40, 50, 7, 31),            (1, 0, 3, 2),           (50, 40, 31, 7),         "arange")
#add_ts_case(["Ascend910A"], "uint8",   (41, 50, 7, 31),            (1, 0, 3, 2),           (50, 41, 31, 7),         "arange")
#add_ts_case(["Ascend910A"], "uint16",  (40, 50, 7, 31),            (1, 0, 3, 2),           (50, 40, 31, 7),         "arange")
#add_ts_case(["Ascend910A"], "int16",   (2000, 3, 32),              (0, 2, 1),              (2000, 32, 3),           "arange")
#add_ts_case(["Ascend910A"], "int16",   (21, 21, 32),               (1, 0, 2),              (21, 21, 32),            "arange")
#add_ts_case(["Ascend910A"], "int8",    (21, 21, 320),              (1, 0, 2),              (21, 21, 320),           "arange")
#add_ts_case(["Ascend910A"], "int8",    (10, 20, 30, 320),          (2, 1, 0, 3),           (30, 20, 10, 320),       "arange")
#add_ts_case(["Ascend910A"], "int16",   (21, 21, 31),               (1, 0, 2),              (21, 21, 31),            "arange")
#add_ts_case(["Ascend910A"], "float32", (64, 2, 6, 240),            (0, 2, 1, 3),           (64, 6, 2, 240),         "arange")
#add_ts_case(["Ascend910A"], "int32",   (129, 2, 2, 2),             (0, 2, 1, 3),           (129, 2, 2, 2),          "arange")
#add_ts_case(["Ascend910A"], "int16",   (129, 2, 2, 4),             (0, 2, 1, 3),           (129, 2, 2, 4),          "arange")
#add_ts_case(["Ascend910A"], "int16", (64, 128, 32),              (0, 2, 1),              (64, 32, 128),           "arange")
#add_ts_case(["Ascend910A"], "int32", (1000, 2000),              (1, 0),              (2000, 1000),           "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 2000),              (1, 0),              (2000, 32),           "arange")
#add_ts_case(["Ascend910A"], "int32", (24, 2000),              (1, 0),              (2000, 24),           "arange")
#add_ts_case(["Ascend910A"], "int16", (128, 16, 16),              (0, 2, 1),              (128, 16, 16),           "arange")
#add_ts_case(["Ascend910A"], "int32", (16, 2000),              (1, 0),              (2000, 16),           "arange")
#add_ts_case(["Ascend910A"], "int32", (1000, 2000),              (1, 0),              (2000, 1000),           "arange")
#add_ts_case(["Ascend910A"], "int16", (4000, 2000),              (1, 0),              (2000, 4000),           "arange")
#add_ts_case(["Ascend910A"], "int32", (8, 2000),              (1, 0),              (2000, 8),           "arange")
#add_ts_case(["Ascend910A"], "int32", (64, 128000),              (1, 0),              (128000, 64),           "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 2320, 2320),              (1, 0, 2),              (2320, 4, 2320),           "arange")
#add_ts_case(["Ascend910A"], "int16", (1024, 100, 128),              (1, 0, 2),              (100, 1024, 128),           "arange")
#add_ts_case(["Ascend910A"], "int32", (4, 664, 664),              (1, 0, 2),              (664, 4, 664),           "arange")
#add_ts_case(["Ascend910A"], "int16", (64, 128000),              (1, 0),              (128000, 64),           "arange")

##---------------------------------------------------------------------------------------------------------------------------------------------
##                                                           b16
##---------------------------------------------------------------------------------------------------------------------------------------------
#
#add_ts_case(["Ascend910A"], "int16", (7, 8, 9, 10, 11),          (2,4,3,1,0),               (9,11,10,8,7),             "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 12, 21, 8),             (3, 2, 1, 0),              (8, 21, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 12, 22, 8),             (3, 2, 1, 0),              (8, 22, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 8, 14, 8),              (3, 2, 1, 0),              (8, 14, 8, 9),             "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 7, 14, 8),              (3, 2, 1, 0),              (8, 14, 7, 9),             "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 8, 22, 8),              (3, 2, 1, 0),              (8, 22, 8, 9),             "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 200, 8),                (2, 1, 0),                 (8, 200, 9),               "arange")
#add_ts_case(["Ascend910A"], "int16", (3, 9, 200, 8),             (0, 3, 2, 1),              (3, 8, 200, 9),            "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 100, 8),                (2, 1, 0),                 (8, 100, 9),               "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 2, 16, 12),             (2, 0, 3, 1),              (16, 2, 12, 2),            "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 10, 12, 2),             (1, 3, 2, 0),              (10, 2, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 11, 12, 2),             (1, 3, 2, 0),              (11, 2, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int16", (19, 11, 12, 2),            (1, 3, 2, 0),              (11, 2, 12, 19),           "arange")
#add_ts_case(["Ascend910A"], "int16", (32, 4, 3, 7),              (0, 3, 2, 1),              (32, 7, 3, 4),             "arange")
#add_ts_case(["Ascend910A"], "int16", (200, 300),                 (1, 0),                    (300, 200),                "arange") # A, (B) ->  B, (A) 
#add_ts_case(["Ascend910A"], "int16", (3, 1240*2+9),              (1, 0),                    (1240*2+9, 3),             "arange") # (A B) ->  (B A) 
#add_ts_case(["Ascend910A"], "int16", (2, 3, 4000),               (1, 2, 0),                 (3, 4000, 2),              "arange") # A, (B, C) ->  B, (C, A) 
#add_ts_case(["Ascend910A"], "int16", (2, 4000, 3),               (2, 1, 0),                 (3, 4000, 2),              "arange") # A, (B, C) ->  C, (B, A) 
#add_ts_case(["Ascend910A"], "int16", (20, 3, 80, 5),             (1, 3, 0, 2),              (3, 5, 20, 80),            "arange") # A, (B, C) ->  C, (A, B) 
#add_ts_case(["Ascend910A"], "int16", (10000, 2),                 (1, 0),                    (2, 10000),                "arange") # (A, B) ->  B, (A) 
#add_ts_case(["Ascend910A"], "int16", (50, 3, 30, 10),            (1, 3, 0, 2),              (3, 10, 50, 30),           "arange") # A, (B, C) ->  C, (A, B) 
#add_ts_case(["Ascend910A"], "int16", (2, 3, 4000),               (2, 1, 0),                 (4000, 3, 2),              "arange") # A, B, (C) ->  (C, B, A) 
#add_ts_case(["Ascend910A"], "int16", (20, 10, 9),                (2, 1, 0),                 (9, 10, 20),               "arange") # A, (B, C) ->  (C, B, A) 
#add_ts_case(["Ascend910A"], "int16", (90, 20, 5),                (0, 2, 1),                 (90, 5, 20),               "arange") # A, (B, C) ->  (A, C, B) 
#add_ts_case(["Ascend910A"], "int16", (2, 3, 20, 5),              (1, 3, 0, 2),              (3, 5, 2, 20),             "arange") # A, (B, C) ->  (C, A, B) 
#add_ts_case(["Ascend910A"], "int16", (2, 7, 3, 4, 20),           (3, 1, 4, 0, 2),           (4, 7, 20, 2, 3),          "arange") # A, B, (C, D) ->  C, (D, A, B) 
#add_ts_case(["Ascend910A"], "int16", (7, 2, 3, 30, 5),           (3, 1, 4, 2, 0),           (30, 2, 5, 3, 7),          "arange") # A, B, (C, D) ->  C, (D, B, A) 
#add_ts_case(["Ascend910A"], "int16", (3, 4, 11, 7, 8, 2),        (5, 4, 3, 2, 1, 0),        (2, 8, 7, 11, 4, 3),       "arange") # A, B, C, (D, E, F) -> F, E, D, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (5, 3, 4, 5, 3, 8, 2),      (6, 5, 0, 4, 3, 2, 1),     (2, 8, 5, 3, 5, 4, 3),     "arange") # A, B, C, (D, E, F) -> F, E, D, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (3, 7, 4, 5, 3, 8, 2),      (1, 6, 5, 4, 3, 0, 2),     (7, 2, 8, 3, 5, 3, 4),     "arange") # A, B, C, (D, E, F) -> F, E, D, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (3, 10, 7, 5, 11, 8, 2),    (2, 6, 5, 4, 1, 3, 0),     (7, 2, 8, 11, 10, 5, 3),   "arange") # A, B, C, (D, E, F) -> F, E, D, (B, C, A)
#add_ts_case(["Ascend910A"], "int16", (3, 10, 5, 11, 8, 2),       (5, 4, 3, 1, 0, 2),        (2, 8, 11, 10, 3, 5),      "arange") # A, B, C, (D, E, F) -> F, E, D, (B, C, A)
#add_ts_case(["Ascend910A"], "int16", (3, 4, 5, 11, 8, 2),        (5, 4, 3, 0, 2, 1),        (2, 8, 11, 3, 5, 4),       "arange") # A, B, C, (D, E, F) -> F, E, D, (A, C, B)
#add_ts_case(["Ascend910A"], "int16", (3, 2, 4, 2, 5, 11, 8, 3),  (1, 3, 7, 6, 5, 0, 2, 4),  (2, 2, 3, 8, 11, 3, 4, 5), "arange") # A, B, C, (D, E, F) -> F, E, D, (A, C, B)
#add_ts_case(["Ascend910A"], "int16", (2, 3, 4, 5, 3, 8, 2),      (6, 4, 0, 5, 3, 2, 1),     (2, 3, 2, 8, 5, 4, 3),     "arange") # A, B, C, (D, E, F) -> F, D, E, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (3,2,4,7,6,2),              (1,3,5,4,0,2),             (2,7,2,6,3,4),             "arange") # A, B, (C, D, E)) -> C, E, (D, A, B)
#add_ts_case(["Ascend910A"], "int16", (2, 3, 2, 4, 5, 11, 8, 2),  (7, 2, 5, 0, 6, 4, 1, 3),  (2, 2, 11, 2, 8, 5, 3, 4), "arange") # A, B, C, (D, E, F) -> F, D, E, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (3, 8, 4, 5, 6),            (4, 2, 0, 3, 1),           (6, 4, 3, 5, 8),           "arange") # A, B, (C, D, E) -> E, C, (A, D, B)
#add_ts_case(["Ascend910A"], "int16", (6, 5, 3, 9, 4),            (3, 2, 0, 4, 1),           (9,3, 6,4, 5),             "arange") # A, B, (C, D, E) ->  D, C, (A, E, B)
#add_ts_case(["Ascend910A"], "int16", (2,65,16,3,4),              (2,4,1,0,3),               (16,4,65,2,3),             "arange") # A, B, (C, D, E) -> C, E, (B, A, D)
#add_ts_case(["Ascend910A"], "int16", (8, 14, 7, 5),              (3, 2, 1, 0),              (5, 7, 14, 8),             "arange") # A (B, C, D) -> D, C, (B, A)
#add_ts_case(["Ascend910A"], "int16", (4,8,3,7,6),                (4,3,0,2,1),               (6,7,4,3,8),               "arange") # A B, (C, D, E) -> E, D, (A, C, B)
#add_ts_case(["Ascend910A"], "int16", (2, 2,6,4,7,5),             (4,0,5,1,3,2),             (7,2,5,2,4,6),             "arange") # A B, (C, D, E) -> D, E, (A, C, B)
#add_ts_case(["Ascend910A"], "int16", (2,6,4,7,5),                (3,4,0,2,1),               (7,5,2,4,6),               "arange") # A B, (C, D, E) -> D, E, (A, C, B)
#add_ts_case(["Ascend910A"], "int16", (6,7,5,8),                  (3,1,0,2),                 (8,7,6,5),                 "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 2, 7, 9, 2, 6),         (1, 4, 3, 5, 0, 2),        (2, 2, 9, 6, 4, 7),        "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 7, 2, 9, 6, 12),        (4, 0, 5, 3, 2, 1),        (6, 2, 12, 9, 2, 7),       "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 58, 30, 2),             (2, 1, 3, 0),              (30, 58, 2, 5),            "arange") # A, B, (C, D) ->  C, (B, D, A) 
#add_ts_case(["Ascend910A"], "int16", (5, 30, 100, 2),            (2, 1, 3, 0),              (100, 30, 2, 5),           "arange") # A, B, (C, D) ->  C, (B, D, A) 
#add_ts_case(["Ascend910A"], "int16", (2,6,9,5,7,3,8,4),          (0,4,7,3,5,1,6,2),         (2,7,4,5,3,6,8,9),         "arange")
#add_ts_case(["Ascend910A"], "int16", (2,4,6,3,9,8,7,5),          (0,2,4,1,7,6,5,3),         (2,6,9,4,5,7,8,3),         "arange") # A, (B, C, D) -> D, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (2,30,3,4,7,9,8),           (0,3,2,1,4,6,5),           (2,4,3,30,7,8,9),          "arange") #
#add_ts_case(["Ascend910A"], "int16", (2,6,5,3,4,7,9,8),          (0,4,3,1,2,5,7,6),         (2,4,3,6,5,7,8,9),         "arange") #
#add_ts_case(["Ascend910A"], "int16", (2,3,6,4,5,9,7,8),          (0,1,4,2,3,7,5,6),         (2,3,5,6,4,8,9,7),         "arange") # A, (B, C, D) -> D, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (2,5,4,7,9,3,8,6),          (0,3,2,5,7,1,6,4),         (2,7,4,3,6,5,8,9),         "arange") # A, (B, C, D) -> D, (C, B, A)
#add_ts_case(["Ascend910A"], "int16", (2,3,7,9,8,5,6,4),          (0,1,5,7,6,3,4,2),         (2,3,5,4,6,9,8,7),         "arange")
#add_ts_case(["Ascend910A"], "int16", (2,6,5,8,7,3,9,4),          (0,4,3,6,5,1,7,2 ),        (2,7,8,9,3,6,4,5),         "arange")
#add_ts_case(["Ascend910A"], "int16", (2,6,7,9,4,5,3,8),          (0,4,5,7,2,3,1,6),         (2,4,5,8,7,9,6,3),         "arange")
#add_ts_case(["Ascend910A"], "int16", (2,4,9,8,3,6,7,5),          (0,2,7,6,1,4,5,3),         (2,9,5,7,4,3,6,8),         "arange") # A (B, C, D) -> D, C, (B, A)
#add_ts_case(["Ascend910A"], "int16", (2,4,9,5,8,3,7,6),          (0,2,7,3,6,1,5,4),         (2,9,6,5,7,4,3,8),         "arange") # A B, (C, D, E) -> E, D, (A, C, B)
#add_ts_case(["Ascend910A"], "int16", (2,3,8,6,9,4,7,5),          (0,1,6,4,7,2,5,3),         (2,3,7,9,5,8,4,6),         "arange") # A B, (C, D, E) -> E, D, (A, C, B)
#add_ts_case(["Ascend910A"], "int16", (2,7,9,3,4,5,8,6),          (0,5,7,1,2,3,6,4),         (2,5,6,7,9,3,8,4),         "arange")
#add_ts_case(["Ascend910A"], "int16", (3,2,2,6,4,7,5,4),          (1,0,7,4,2,5,3,6),         (2,3,4,4,2,7,6,5),         "arange")
#add_ts_case(["Ascend910A"], "int16", (3,2,9,6,4,7,5,8),          (1,0,7,4,2,5,3,6),         (2,3,8,4,9,7,6,5),         "arange")
#add_ts_case(["Ascend910A"], "int16", (3,5,2,9,4,8,6,7),          (1,3,0,7,2,6,4,5),         (5,9,3,7,2,6,4,8),         "arange")
#add_ts_case(["Ascend910A"], "int16", (3,8,5,9,7,2,6,4),          (1,6,3,7,5,0,4,2),         (8,6,9,4,2,3,7,5),         "arange")
#add_ts_case(["Ascend910A"], "int16", (4,8,5,2,9,7,6,3),          (2,6,3,0,7,5,4,1),         (5,6,2,4,3,7,9,8),         "arange")
#add_ts_case(["Ascend910A"], "int16", (6,7,3,8,2,5,9,4),          (4,5,1,6,0,3,7,2),         (2,5,7,9,6,8,4,3),         "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 3, 9, 6, 7, 4, 5, 8),   (0, 1, 7, 4, 5, 2, 3, 6),  (2, 3, 8, 7, 4, 9, 6, 5),  "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 3, 9, 8, 6, 4, 5, 7),   (0, 1, 7, 6, 4, 2, 3, 5),  (2, 3, 7, 5, 6, 9, 8, 4),  "arange")
#add_ts_case(["Ascend910A"], "int16", (3, 5, 6, 8, 9, 4, 2, 7),   (1, 3, 4, 6, 7, 2, 0, 5),  (5, 8, 9, 2, 7, 6, 3, 4),  "arange")
#add_ts_case(["Ascend910A"], "int16", (3, 7, 8, 6, 9, 5, 2, 4),   (1, 5, 6, 4, 7, 3, 0, 2),  (7, 5, 2, 9, 4, 6, 3, 8),  "arange")
#add_ts_case(["Ascend910A"], "int16", (3, 9, 7, 2, 8, 6, 5, 4),   (1, 7, 5, 0, 6, 4, 3, 2),  (9, 4, 6, 3, 5, 8, 2, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 2, 8, 9, 5, 6, 3, 7),   (2, 0, 6, 7, 3, 4, 1, 5),  (8, 4, 3, 7, 9, 5, 2, 6),  "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 3, 5, 7, 9, 6, 8, 2),   (2, 1, 3, 5, 7, 4, 6, 0),  (5, 3, 7, 6, 2, 9, 8, 4),  "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 3, 4, 9, 2, 8, 6, 7),   (3, 1, 2, 7, 0, 6, 4, 5),  (9, 3, 4, 7, 5, 6, 2, 8),  "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 4, 2, 8, 9, 6, 3, 7),   (3, 2, 0, 6, 7, 4, 1, 5),  (8, 2, 5, 3, 7, 9, 4, 6),  "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 6, 3, 2, 7, 4, 9, 8),   (3, 4, 1, 0, 5, 2, 7, 6),  (2, 7, 6, 5, 4, 3, 8, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 8, 9, 5, 3, 7, 6, 4),   (0, 6, 7, 3, 1, 5, 4, 2),  (2, 6, 4, 5, 8, 7, 3, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 5, 3, 8, 7, 9, 2, 6),   (2, 3, 1, 6, 5, 7, 0, 4),  (3, 8, 5, 2, 9, 6, 4, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 8, 5, 2, 6, 9, 7, 3),   (2, 6, 3, 0, 4, 7, 5, 1),  (5, 7, 2, 4, 6, 3, 9, 8),  "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 6, 3, 4, 8, 2, 9, 7),   (3, 4, 1, 2, 6, 0, 7, 5),  (4, 8, 6, 3, 9, 5, 7, 2),  "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 7, 2, 8, 9, 6, 4, 3),   (3, 5, 0, 6, 7, 4, 2, 1),  (8, 6, 5, 4, 3, 9, 2, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 7, 6, 9, 2, 8, 4, 3),   (3, 5, 4, 7, 0, 6, 2, 1),  (9, 8, 2, 3, 5, 4, 6, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 9, 2, 3, 4, 8, 6, 7),   (3, 7, 0, 1, 2, 6, 4, 5),  (3, 7, 5, 9, 2, 6, 4, 8),  "arange")
#add_ts_case(["Ascend910A"], "int16", (6, 2, 8, 9, 5, 3, 7, 4),   (4, 0, 6, 7, 3, 1, 5, 2),  (5, 6, 7, 4, 9, 2, 3, 8),  "arange")
#add_ts_case(["Ascend910A"], "int16", (6, 4, 8, 9, 7, 2, 3, 5),   (4, 2, 6, 7, 5, 0, 1, 3),  (7, 8, 3, 5, 2, 6, 4, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (6, 5, 4, 2, 9, 3, 8, 7),   (4, 3, 2, 0, 7, 1, 6, 5),  (9, 2, 4, 6, 7, 5, 8, 3),  "arange")
#add_ts_case(["Ascend910A"], "int16", (7, 6, 9, 3, 5, 4, 2, 8),   (5, 4, 7, 1, 3, 2, 0, 6),  (4, 5, 8, 6, 3, 9, 7, 2),  "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 2, 9, 6, 7, 4, 5, 3),   (6, 0, 7, 4, 5, 2, 3, 1),  (5, 8, 3, 7, 4, 9, 6, 2),  "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 5, 9, 2, 6, 7, 4, 3),   (6, 3, 7, 0, 4, 5, 2, 1),  (4, 2, 3, 8, 6, 7, 9, 5),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 2, 8, 4, 5, 6, 7, 3),   (7, 0, 6, 2, 3, 4, 5, 1),  (3, 9, 7, 8, 4, 5, 6, 2),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 4, 2, 3, 6, 5, 8, 7),   (7, 2, 0, 1, 4, 3, 6, 5),  (7, 2, 9, 4, 6, 3, 8, 5),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 5, 8, 4, 2, 6, 7, 3),   (7, 3, 6, 2, 0, 4, 5, 1),  (3, 4, 7, 8, 9, 2, 6, 5),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 7, 8, 3, 5, 4, 2, 6),   (7, 5, 6, 1, 3, 2, 0, 4),  (6, 4, 2, 7, 3, 8, 9, 5),  "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 4, 3, 9, 6, 2, 5, 7),   (6, 2, 1, 7, 4, 0, 3, 5),  (5, 3, 4, 7, 6, 8, 9, 2),  "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 5, 4, 3, 9, 2, 6, 7),   (6, 3, 2, 1, 7, 0, 4, 5),  (6, 3, 4, 5, 7, 8, 9, 2),  "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 6, 4, 5, 3, 9, 2, 7),   (6, 4, 2, 3, 1, 7, 0, 5),  (2, 3, 4, 5, 6, 7, 8, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 6, 5, 2, 3, 9, 4, 7),   (6, 4, 3, 0, 1, 7, 2, 5),  (4, 3, 2, 8, 6, 7, 5, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (3, 4, 6, 2, 7, 8, 9, 5),   (1, 2, 4, 0, 5, 6, 7, 3),  (4, 6, 7, 3, 8, 9, 5, 2),  "arange")
#add_ts_case(["Ascend910A"], "int16", (3, 8, 4, 2, 9, 7, 5, 6),   (1, 6, 2, 0, 7, 5, 3, 4),  (8, 5, 4, 3, 6, 7, 2, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 2, 3, 9, 8, 5, 6, 7),   (2, 0, 1, 7, 6, 3, 4, 5),  (3, 4, 2, 7, 6, 9, 8, 5),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 3, 7, 6, 2, 8, 5, 4),   (7, 1, 5, 4, 0, 6, 3, 2),  (4, 3, 8, 2, 9, 5, 6, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 7, 6, 9, 5, 8, 3, 4),   (0, 5, 4, 7, 3, 6, 1, 2),  (2, 8, 5, 4, 9, 3, 7, 6),  "arange")
#add_ts_case(["Ascend910A"], "int16", (3, 8, 9, 2, 7, 4, 5, 6),   (1, 6, 7, 0, 5, 2, 3, 4),  (8, 5, 6, 3, 4, 9 ,2, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 7, 5, 8, 2, 9, 6, 3),   (2, 5, 3, 6, 0, 7, 4, 1),  (5, 9, 8, 6, 4, 3, 2, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (6, 4, 9, 2, 8, 5, 7, 3),   (4, 2, 7, 0, 6, 3, 5, 1),  (8, 9, 3, 6, 7, 2, 5, 4),  "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 5, 7, 6, 3, 9, 4, 2),   (6, 3, 5, 4, 1, 7, 2, 0),  (4, 6, 9, 3, 5, 2, 7, 8),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 2, 7, 8, 3, 6, 4, 5),   (7, 0, 5, 6, 1, 4, 2, 3),  (5, 9, 6, 4, 2, 3, 7, 8),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 3, 5, 6, 8, 4, 7, 2),   (7, 1, 3, 4, 6, 2, 5, 0),  (2, 3, 6, 8, 7, 5, 4, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 5, 8, 4, 6, 3, 7, 2),   (7, 3, 6, 2, 4, 1, 5, 0),  (2, 4, 7, 8, 6, 5, 3, 9),  "arange")
#add_ts_case(["Ascend910A"], "int16", (9, 6, 7, 8, 2, 5, 3, 4),   (7, 4, 5, 6, 0, 3, 1, 2),  (4, 2, 5, 3, 9, 8, 6, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 7, 5, 8, 9, 6, 3, 4),   (0, 5, 3, 6, 7, 4, 1, 2),  (2, 6, 8, 3, 4, 9, 7, 5),  "arange")
#add_ts_case(["Ascend910A"], "int16", (2, 3, 8, 7, 6, 9, 4, 5),   (0, 1, 6, 5, 4, 7, 2, 3),  (2, 3, 4, 9, 6, 5, 8, 7),  "arange")
#add_ts_case(["Ascend910A"], "int16", (4, 19, 6, 7, 8, 3, 5, 2),  (2, 7, 4, 5, 6, 1, 3, 0),  (6, 2, 8, 3, 5, 19, 7, 4), "arange")
#add_ts_case(["Ascend910A"], "int16", (6, 19, 4, 5, 7, 2, 3, 8),  (4, 7, 2, 3, 5, 0, 1, 6),  (7, 8, 4, 5, 2, 6, 19, 3), "arange")
#add_ts_case(["Ascend910A"], "int16", (5, 2, 7, 3, 4, 8, 19, 6),  (3, 0, 5, 1, 2, 6, 7, 4),  (3, 5, 8, 2, 7, 19, 6, 4), "arange")
#add_ts_case(["Ascend910A"], "int16", (8, 4, 114),  (0, 2, 1),  (8, 114, 4), "arange")


##---------------------------------------------------------------------------------------------------------------------------------------------
##                                                           b32
##---------------------------------------------------------------------------------------------------------------------------------------------
#
#add_ts_case(["Ascend910A"], "int32", (9, 12, 21, 8),            (3, 2, 1, 0),               (8, 21, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 12, 22, 8),            (3, 2, 1, 0),               (8, 22, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 8, 14, 8),             (3, 2, 1, 0),               (8, 14, 8, 9),             "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 7, 14, 8),             (3, 2, 1, 0),               (8, 14, 7, 9),             "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 8, 22, 8),             (3, 2, 1, 0),               (8, 22, 8, 9),             "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 200, 8),               (2, 1, 0),                  (8, 200, 9),               "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 9, 200, 8),            (0, 3, 2, 1),               (3, 8, 200, 9),            "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 100, 8),               (2, 1, 0),                  (8, 100, 9),               "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 2, 16, 12),            (2, 0, 3, 1),               (16, 2, 12, 2),            "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 10, 12, 2),            (1, 3, 2, 0),               (10, 2, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int32", (9, 11, 12, 2),            (1, 3, 2, 0),               (11, 2, 12, 9),            "arange")
#add_ts_case(["Ascend910A"], "int32", (19, 11, 12, 2),           (1, 3, 2, 0),               (11, 2, 12, 19),           "arange")
#add_ts_case(["Ascend910A"], "int32", (32, 4, 3, 7),             (0, 3, 2, 1),               (32, 7, 3, 4),             "arange")
#add_ts_case(["Ascend910A"], "int32", (200, 300),                (1, 0),                     (300, 200),                "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 1240*2+9),             (1, 0),                     (1240*2+9, 3),             "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 4000),              (1, 2, 0),                  (3, 4000, 2),              "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 4000, 3),              (2, 1, 0),                  (3, 4000, 2),              "arange")
#add_ts_case(["Ascend910A"], "int32", (20, 3, 80, 5),            (1, 3, 0, 2),               (3, 5, 20, 80),            "arange")
#add_ts_case(["Ascend910A"], "int32", (10000, 2),                (1, 0),                     (2, 10000),                "arange")
#add_ts_case(["Ascend910A"], "int32", (50, 3, 30, 10),           (1, 3, 0, 2),               (3, 10, 50, 30),           "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 4000),              (2, 1, 0),                  (4000, 3, 2),              "arange")
#add_ts_case(["Ascend910A"], "int32", (20, 10, 9),               (2, 1, 0),                  (9, 10, 20),               "arange")
#add_ts_case(["Ascend910A"], "int32", (90, 20, 5),               (0, 2, 1),                  (90, 5, 20),               "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 20, 5),             (1, 3, 0, 2),               (3, 5, 2, 20),             "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 7, 3, 4, 20),          (3, 1, 4, 0, 2),            (4, 7, 20, 2, 3),          "arange")
#add_ts_case(["Ascend910A"], "int32", (7, 2, 3, 30, 5),          (3, 1, 4, 2, 0),            (30, 2, 5, 3, 7),          "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 4, 11, 7, 8, 2),       (5, 4, 3, 2, 1, 0),         (2, 8, 7, 11, 4, 3),       "arange")
#add_ts_case(["Ascend910A"], "int32", (5, 3, 4, 5, 3, 8, 2),     (6, 5, 0, 4, 3, 2, 1),      (2, 8, 5, 3, 5, 4, 3),     "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 7, 4, 5, 3, 8, 2),     (1, 6, 5, 4, 3, 0, 2),      (7, 2, 8, 3, 5, 3, 4),     "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 10, 7, 5, 11, 8, 2),   (2, 6, 5, 4, 1, 3, 0),      (7, 2, 8, 11, 10, 5, 3),   "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 10, 5, 11, 8, 2),      (5, 4, 3, 1, 0, 2),         (2, 8, 11, 10, 3, 5),      "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 4, 5, 11, 8, 2),       (5, 4, 3, 0, 2, 1),         (2, 8, 11, 3, 5, 4),       "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 2, 4, 2, 5, 11, 8, 3), (1, 3, 7, 6, 5, 0, 2, 4),   (2, 2, 3, 8, 11, 3, 4, 5), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 4, 5, 3, 8, 2),     (6, 4, 0, 5, 3, 2, 1),      (2, 3, 2, 8, 5, 4, 3),     "arange")
#add_ts_case(["Ascend910A"], "int32", (3,2,4,7,6,2),             (1,3,5,4,0,2),              (2,7,2,6,3,4),             "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 3, 2, 4, 5, 11, 8, 2),  (7, 2, 5, 0, 6, 4, 1, 3),  (2, 2, 11, 2, 8, 5, 3, 4), "arange")
#add_ts_case(["Ascend910A"], "int32", (3, 8, 4, 5, 6),            (4, 2, 0, 3, 1),           (6, 4, 3, 5, 8),           "arange")
#add_ts_case(["Ascend910A"], "int32", (6, 5, 3, 9, 4),            (3, 2, 0, 4, 1),           (9,3, 6,4, 5),             "arange")
#add_ts_case(["Ascend910A"], "int32", (2,65,16,3,4),              (2,4,1,0,3),               (16,4,65,2,3),             "arange")
#add_ts_case(["Ascend910A"], "int32", (8, 14, 7, 5),              (3, 2, 1, 0),              (5, 7, 14, 8),             "arange")
#add_ts_case(["Ascend910A"], "int32", (4,8,3,7,6),                (4,3,0,2,1),               (6,7,4,3,8),               "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 2,6,4,7,5),             (4,0,5,1,3,2),             (7,2,5,2,4,6),             "arange")
#add_ts_case(["Ascend910A"], "int32", (2,6,4,7,5),                (3,4,0,2,1),               (7,5,2,4,6),               "arange")
#add_ts_case(["Ascend910A"], "int32", (6,7,5,8),                  (3,1,0,2),                 (8,7,6,5),                 "arange")
#add_ts_case(["Ascend910A"], "int32", (4, 2, 7, 9, 2, 6),         (1, 4, 3, 5, 0, 2),        (2, 2, 9, 6, 4, 7),        "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 7, 2, 9, 6, 12),        (4, 0, 5, 3, 2, 1),        (6, 2, 12, 9, 2, 7),       "arange")
#add_ts_case(["Ascend910A"], "int32", (5, 58, 30, 2),             (2, 1, 3, 0),              (30, 58, 2, 5),            "arange")
#add_ts_case(["Ascend910A"], "int32", (5, 30, 100, 2),            (2, 1, 3, 0),              (100, 30, 2, 5),           "arange")
#add_ts_case(["Ascend910A"], "int32", (5, 6, 7, 2, 4, 8, 19, 3),  (3, 4, 5, 0, 2, 6, 7, 1),  (2, 4, 8, 5, 7, 19, 3, 6), "arange")
#add_ts_case(["Ascend910A"], "int32", (2, 4, 9, 7, 3, 8, 5, 6),   (0, 2, 7, 5, 1, 6, 3, 4),  (2, 9, 6, 8, 4, 5, 7, 3), "arange")
#add_ts_case(["Ascend910A"], "int32",  (4, 2, 33, 3, 51, 32), (0, 4, 1, 3, 2, 5), (4, 51, 2, 3, 33, 32), "arange")




##---------------------------------------------------------------------------------------------------------------------------------------------
##                                                           b64
##---------------------------------------------------------------------------------------------------------------------------------------------
#add_ts_case(["Ascend910A"], "int64", (9, 12, 21, 8),                  (3, 2, 1, 0),              (8, 21, 12, 9),              "arange")
#add_ts_case(["Ascend910A"], "int64", (5, 3, 4, 9, 2, 8, 6, 7),        (3, 1, 2, 7, 0, 6, 4, 5),  (9, 3, 4, 7, 5, 6, 2, 8),    "arange")
#add_ts_case(["Ascend910A"], "int64", (2, 4, 7, 8, 19, 3, 6, 5),       (0, 2, 5, 6, 7, 1, 4, 3),  (2, 7, 3, 6, 5, 4, 19, 8),   "arange")
#add_ts_case(["Ascend910A"], "int64", (4, 6, 8, 19, 2, 3, 5, 7),       (2, 4, 6, 7, 0, 1, 3, 5),  (8, 2, 5, 7, 4, 6, 19, 3),   "arange")
#add_ts_case(["Ascend910A"], "int64", (9, 12, 21, 8),                  (2, 1, 0, 3),              (21, 12, 9, 8),              "arange")
#add_ts_case(["Ascend910A"], "int64", (9, 12, 21, 7),                  (2, 1, 0, 3),              (21, 12, 9, 7),              "arange")

### nok





#arr = [2, 3, 4, 5, 6, 7, 8, 9]
#import itertools
#import random
#perm = list(itertools.permutations([0,1,2,3,4,5,6,7]))
#count = 0
#case_num = 0
#for p in perm:
#    j = list(p)
#    if (p[7] == 7):
#        continue
#    x = (arr[j[0]], arr[j[1]], arr[j[2]], arr[j[3]], arr[j[4]], arr[j[5]], arr[j[6]], arr[j[7]])
#    count = count + 1
#    y = (x[p[0]], x[p[1]], x[p[2]], x[p[3]], x[p[4]], x[p[5]], x[p[6]], x[p[7]]) 
#    r = random.randint(0,1000)
#    if r == 99:
#        case_num = case_num + 1
#        print("add case:", case_num, ", x:", x, ", perm :", p, ", y :", y)
#        add_ts_case(["Ascend910A"], "int16", x,  p, y, "random")
#        #break

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
#            add_ts_case(["Ascend910A"], "int32", x,  p, y, "random")
#            break

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
#            add_ts_case(["Ascend910A"], "float16", x,  p, y, "random")

#arr = [2, 3, 4, 33, 51, 7]
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
#            add_ts_case(["Ascend910A"], "float16", x,  p, y, "random")

#def test_transpose_a100(test_arg):
#    from impl.dynamic.transpose import transpose 
#    from te import platform as cce_conf
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
#ut_case.add_cust_test_func(test_func=test_transpose_a100)

if __name__ == '__main__':
    simulator_lib_path = "/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend310"], simulator_mode="ca", simulator_lib_path=simulator_lib_path)

