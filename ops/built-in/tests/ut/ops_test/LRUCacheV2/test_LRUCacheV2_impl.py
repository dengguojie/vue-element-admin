# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("LRUCacheV2", "impl.dynamic.lru_cache_v2", "lru_cache_v2")


def param_dict(shape, dtype):
    return {"shape": shape, "dtype": dtype, "format": "ND", "ori_shape": shape, "ori_format": "ND"}


def impl_lsit(shape_a, dtype_a, shape_b, dtype_b, shape_c, dtype_c, shape_d, dtype_d, pre_route_count=4):
    input_list = [param_dict(shape_a, dtype_a), param_dict(shape_b, dtype_b), param_dict(shape_c, dtype_c),
                  param_dict(shape_d, dtype_d), param_dict(shape_d, dtype_d)]
    output_list = [param_dict(shape_b, dtype_b), param_dict(shape_c, dtype_c), param_dict(shape_d, dtype_d),
                   param_dict(shape_a, dtype_a), param_dict(shape_a, dtype_a), param_dict([1, ], dtype_a)]
    param_list = [pre_route_count]

    return input_list + output_list + param_list


case1 = {"params": [
    {"shape": [-1], "dtype": "int32", "format": "ND", "ori_shape": [32], "ori_format": "ND", "ori_dtype": "int32",
     "range": [(1, 100000)]},
    {"shape": [1000, 256], "dtype": "float32", "format": "ND", "ori_shape": [1000, 256], "ori_format": "ND",
     "ori_dtype": "float32", "range": [(1000, 1000), (256, 256)]},
    {"shape": [131072], "dtype": "float32", "format": "ND", "ori_shape": [131072], "ori_format": "ND",
     "ori_dtype": "float32", "range": [(131072, 131072)]},
    {"shape": [512], "dtype": "int32", "format": "ND", "ori_shape": [512], "ori_format": "ND", "ori_dtype": "int32",
     "range": [(512, 512)]},
    {"shape": [1], "dtype": "int32", "format": "ND", "ori_shape": [1], "ori_format": "ND", "ori_dtype": "int32",
     "range": [(1, 1)]},
    {"shape": [1000, 256], "dtype": "float32", "format": "ND", "ori_shape": [1000, 256], "ori_format": "ND",
     "ori_dtype": "float32", "range": [(1000, 1000), (256, 256)]},
    {"shape": [131072], "dtype": "float32", "format": "ND", "ori_shape": [131072], "ori_format": "ND",
     "ori_dtype": "float32", "range": [(131072, 131072)]},
    {"shape": [512], "dtype": "int32", "format": "ND", "ori_shape": [512], "ori_format": "ND", "ori_dtype": "int32",
     "range": [(512, 512)]},
    {"shape": [1], "dtype": "int32", "format": "ND", "ori_shape": [1], "ori_format": "ND", "ori_dtype": "int32",
     "range": [(1, 1)]},
    {"shape": [1], "dtype": "int32", "format": "ND", "ori_shape": [1], "ori_format": "ND", "ori_dtype": "int32",
     "range": [(1, 1)]},
    {"shape": [1], "dtype": "int32", "format": "ND", "ori_shape": [1], "ori_format": "ND", "ori_dtype": "int32",
     "range": [(1, 1)]}, 1],
         "case_name": "dynamic_lru_case_1",
         "expect": "success",
         "support_expect": True}


err2 = {"params": impl_lsit([2048], "int32", [100, 4], "float32", [2048], "float32", [512], "int32", pre_route_count=3),
        "case_name": "erro_case_2",
        "expect": RuntimeError,
        "support_expect": False}

ut_case.add_case(["Ascend920A"], case1)
ut_case.add_case(["Ascend910A"], err2)

ut_case.run(["Ascend910A", "Ascend920A"])
