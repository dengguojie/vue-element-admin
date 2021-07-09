# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("LRUCacheV2", "impl.lru_cache_v2", "lru_cache_v2")

def param_dict(shape, dtype):

    return {"shape": shape, "dtype": dtype, "format": "ND", "ori_shape": shape, "ori_format": "ND"}

def impl_lsit(shape_a, dtype_a, shape_b, dtype_b, shape_c, dtype_c, shape_d, dtype_d, pre_route_count=4):
    input_list = [param_dict(shape_a, dtype_a), param_dict(shape_b, dtype_b),param_dict(shape_c, dtype_c),
    param_dict(shape_d, dtype_d), param_dict(shape_d, dtype_d)]
    output_list = [param_dict(shape_b, dtype_b),param_dict(shape_c, dtype_c),param_dict(shape_d, dtype_d),
    param_dict(shape_a, dtype_a),param_dict(shape_a, dtype_a),param_dict([1,], "int32")]
    param_list = [pre_route_count]

    return input_list + output_list + param_list


case1 = {"params": impl_lsit([32],"int32", [100, 4], "float32", [2048], "float32", [512], "int32"),
         "case_name": "case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": impl_lsit([2048],"int32", [100, 4], "float32", [2048], "float32", [512], "int32"),
         "case_name": "case_2",
         "expect": "success",
         "support_expect": True}


err1 = {"params": impl_lsit([2048],"int64", [100, 4], "float32", [2048], "float32", [512], "int32"),
         "case_name": "erro_case_1",
         "expect": RuntimeError,
         "support_expect": False}

err2 = {"params": impl_lsit([2048],"int32", [100, 4], "float32", [2048], "float32", [512], "int32",pre_route_count=3),
        "case_name": "erro_case_2",
        "expect": RuntimeError,
        "support_expect": False}


def test_lru_1981(test_arg):
    from impl.lru_cache_v2 import lru_cache_v2
    from te import platform as cce_conf
    cce_conf.cce_conf.te_set_version("Ascend920A", core_type="VectorCore")
    lru_cache_v2(*(case1["params"]))
    lru_cache_v2(*(case2["params"]))
    cce_conf.cce_conf.te_set_version(test_arg)

ut_case.add_case(["Ascend710",], err1)
ut_case.add_case(["Ascend710",], err2)
ut_case.add_cust_test_func(test_func=test_lru_1981)

if __name__ == '__main__':
    ut_case.run('Ascend910A')
    exit(0)

