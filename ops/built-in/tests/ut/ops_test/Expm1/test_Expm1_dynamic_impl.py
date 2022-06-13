#!/usr/bin/env python
# -*- coding :UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Expm1", "impl.dynamic.expm1", "expm1")

case1 = {
    "params":[
        {"shape":(-1,), "ori_shape":(2,4), "dtype":"float16", "format":"ND", "ori_format":"ND", "range":((1,100),)},
        {"shape":(-1,), "ori_shape":(2,4), "dtype":"float16", "format":"ND", "ori_format":"ND", "range":((1,100),)},
    ],
    "case_name":"Expm1_1",
    "expect":"success",
    "support_expect":True
}

ut_case.add_case(["Ascend910A", "Ascend310P3", "Ascend610"], case1)

def reload_check_support(test_arg):
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))


ut_case.add_cust_test_func(test_func=reload_check_support)

if __name__=='__main__':
    ut_case.run("Ascend910A")
