#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Tanh", "impl.dynamic.tanh", "tanh")

case1 = {"params": [{"shape": (-2,), "ori_shape": (-2,), "range": ((1, None),), "format": "ND", "ori_format": "ND",
                     'dtype': "float32"},
                    {"shape": (-2,), "ori_shape": (-2,), "range": ((1, None),), "format": "ND", "ori_format": "ND",
                     'dtype': "float32"}],
         "case_name": "tanh_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)


def test_ln_import_lib(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))


ut_case.add_cust_test_func(test_func=test_ln_import_lib)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])