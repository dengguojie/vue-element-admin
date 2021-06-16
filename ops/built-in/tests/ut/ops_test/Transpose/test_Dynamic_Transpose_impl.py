#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from impl.util.platform_adapter import tbe_context
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import os
import time

#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(linewidth=10000)

ut_case = OpUT("Transpose", "impl.dynamic.transpose", "transpose")

def calc_expect_func(x, perm, y):
    x_val = x.get("value")
    p_val = perm.get("value")
    y_val = np.transpose(x_val, p_val)
    print("------------------actual---------------------")
    print(y.get("value"))
    print("------------------expect---------------------")
    print(y_val)
    return (y_val,)

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
        value = np.arange(0, vol, dtype = d_type).reshape(x)
    else:
        value = np.random.randint(100, size=vol, dtype=d_type).reshape(x)


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
                                   "case_name": "case_" + str(os.getpid())+ "_" + str((int)(time.time())),
                                   "precision_standard": precision_info.PrecisionStandard(0, 0)
                               })

add_ts_case(["Ascend910A", "Ascend310"], "bool", (7, 9, 30, 8),           (2, 1, 0, 3),           (30, 9, 7, 8),         "random")
add_ts_case(["Ascend910A", "Ascend310"], "bool", (7, 9, 30, 3),           (2, 1, 0, 3),           (30, 9, 7, 3),         "random")
#for i in range(1,64):
#    add_ts_case(["Ascend910A"], "bool", (7, 9, 30, i),           (2, 1, 0, 3),           (30, 9, 7, i),         "random")


ut_case.add_precision_case("Ascend910A",
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 200),
                                           "dtype": "uint8",
                                           "format": "ND",
                                           "ori_shape": (33, 200),
                                           "range": ((33, 33), (200, 200)),
                                           "run_shape": (33, 200),
                                           "value": np.random.randint(100, size=33*200, dtype="uint8").reshape(33, 200),
                                           "ori_format": "ND",
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2,),
                                           "run_shape": (2,),
                                           "dtype": "int32",
                                           "ori_shape": (2),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([1, 0]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (200, 33),
                                           "dtype": "uint8",
                                           "format": "ND",
                                           "ori_shape": (200, 33),
                                           "range": ((200, 200), (33, 33),),
                                           "run_shape": (200, 33),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       },
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

ut_case.add_precision_case("Ascend910A",
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 2000),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 2000),
                                           "range": ((1000, 1000), (2000, 2000),),
                                           "run_shape": (1000, 2000),
                                           "ori_format": "ND",
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2,),
                                           "run_shape": (2,),
                                           "dtype": "int32",
                                           "ori_shape": (2),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.arange(1, -1, -1), # [1,0]
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (2000, -1),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (2000, -1),
                                           "range": ((2000, 2000), (1000, 1000),),
                                           "run_shape": (2000, 1000),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       },
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

ut_case.add_precision_case("Ascend910A",
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 3, 4, 500, 601),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 3, 4, 500, 601),
                                           "range": ((2, 2), (3, 3), (4, 4), (500, 500), (601, 601),),
                                           "run_shape": (2, 3, 4, 500, 601),
                                           "ori_format": "ND",
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (5,),
                                           "run_shape": (5,),
                                           "dtype": "int32",
                                           "ori_shape": (5),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([0, 2, 1, 3, 4]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (-1, 4, 3, 500, 601),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 4, 3, 500, 601),
                                           "range": ((2, 2), (3, 3), (4, 4), (500, 500), (601, 601), ),
                                           "run_shape": (2, 4, 3, 500, 601),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       },
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

ut_case.add_precision_case("Ascend910A",
                           {
                               "params":
                                   [
                                       {
                                           "shape": (-1, 3, 4, 5, 6),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 3, 4, 5, 6),
                                           "range": ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6),),
                                           "run_shape": (2, 3, 4, 5, 6),
                                           "ori_format": "ND",
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (5,),
                                           "run_shape": (5,),
                                           "dtype": "int32",
                                           "ori_shape": (5),
                                           "ori_format" : "ND",
                                           "format": "ND",
                                           "value": np.array([0, 2, 1, 3, 4]),
                                           "value_need_in_tiling": True,
                                           "param_type": "input"
                                       },
                                       {
                                           "shape": (-1, 4, 3, 5, 6),
                                           "dtype": "float32",
                                           "format": "ND",
                                           "ori_shape": (-1, 4, 3, 5, 6),
                                           "range": ((2, 2), (4, 4), (3, 3), (5, 5), (6, 6), ),
                                           "run_shape": (2, 4, 3, 5, 6),
                                           "ori_format": "ND",
                                           "param_type": "output"
                                       },
                                   ],
                               "calc_expect_func": calc_expect_func,
                               "precision_standard": precision_info.PrecisionStandard(0, 0)
                           })

#ut_case.add_precision_case(["Ascend920A"],
#                           {
#                               "params":
#                                   [
#                                       {
#                                           "shape": (4, 255, 3, 8),
#                                           "dtype": "int32",
#                                           "format": "ND",
#                                           "ori_shape": (-1, 255, 3, 8),
#                                           "range": ((4, 4), (255, 255), (3, 3), (8, 8),),
#                                           "run_shape": (4, 255,3, 8),
#                                           "ori_format": "ND",
#                                           "param_type": "input",
#                                           #"value": np.arange(0, 256*8*4*20*16*16, dtype="int32").reshape(256, 8, 4, 20, 16, 16)
#                                       },
#                                       {
#                                           "shape": (4,),
#                                           "run_shape": (4,),
#                                           "dtype": "int32",
#                                           "ori_shape": (4),
#                                           "ori_format" : "ND",
#                                           "format": "ND",
#                                           "value": np.array([2, 1, 0, 3]),
#                                           "value_need_in_tiling": True,
#                                           "param_type": "input"
#                                       },
#                                       {
#                                           "shape": (3, 255, 4, 8),
#                                           "dtype": "int32",
#                                           "format": "ND",
#                                           "ori_shape": (3, 255, 4, 8),
#                                           "range": ((3, 3), (255, 255), (4, 4), (8, 8), ),
#                                           "run_shape": (3, 255, 4, 8),
#                                           "ori_format": "ND",
#                                           "param_type": "output"
#                                       },
#                                   ],
#                               "calc_expect_func": calc_expect_func,
#                               "precision_standard": precision_info.PrecisionStandard(0, 0)
#                           })
#
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

#ut_case.add_cust_test_func(test_func=test_transpose_920a)

if __name__ == '__main__':
    simulator_lib_path = "/usr/local/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910A", "Ascend920A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

