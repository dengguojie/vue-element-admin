'''
test code
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ApplyAdagradV2D", "impl.dynamic.apply_adagradv2_d", "apply_adagradv2_d")

case1 = {
    "params": [{
        "shape": (-1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND",
        "range": [(1, 100)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND",
        "range": [(1, 100)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND",
        "range": [(1, 100)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND",
        "range": [(1, 100)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND",
        "range": [(1, 100)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (2,),
        "ori_format": "ND",
        "range": [(1, 100)]
    }, 0.0001],
    "case_name": "apply_adagradv2_d_1",
    "expect": RuntimeError,
    "support_expect": True
}

case2 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100), (1, 100)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100), (1, 100)]
    }, {
        "shape": (1,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (1,),
        "ori_format": "ND",
        "range": [(1, 100)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100), (1, 100)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100), (1, 100)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (2, 4, 4),
        "ori_format": "ND",
        "range": [(1, 100), (1, 100), (1, 100)]
    }, 0.0001],
    "case_name": "apply_adagradv2_d_2",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
