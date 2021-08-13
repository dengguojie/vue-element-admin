#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from te import platform as cce_conf
from impl.top_k_pq_distance_merge import top_k_pq_distance_merge
from tbe.common.platform.platform_info import set_current_compile_soc_info

ut_case = OpUT("TopKPQDistanceMerge", "impl.top_k_pq_distance_merge", "top_k_pq_distance_merge")

def top_k_pq_distance_merge_001(test_arg):
    set_current_compile_soc_info("Ascend710")
    top_k_pq_distance_merge({"shape": (52,), "dtype": "float16", "format": "ND", "ori_shape": (52,),
                             "ori_format": "ND"},
                            {"shape": (52,), "dtype": "int32", "format": "ND", "ori_shape": (52,),
                             "ori_format": "ND"},
                            {"shape": (52,), "dtype": "int32", "format": "ND", "ori_shape": (52,),
                             "ori_format": "ND"},
                            {"shape": (26,), "dtype": "float16", "format": "ND", "ori_shape": (26,),
                             "ori_format": "ND"},
                            {"shape": (26,), "dtype": "int32", "format": "ND", "ori_shape": (26,),
                             "ori_format": "ND"},
                            {"shape": (26,), "dtype": "int32", "format": "ND", "ori_shape": (26,),
                             "ori_format": "ND"},
                            26)
    set_current_compile_soc_info(test_arg)

ut_case.add_cust_test_func(test_func=top_k_pq_distance_merge_001)

if __name__ == '__main__':
    ut_case.run("Ascend710")
