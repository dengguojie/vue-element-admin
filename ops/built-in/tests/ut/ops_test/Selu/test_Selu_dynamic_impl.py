#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Selu", "impl.dynamic.selu", "selu")


def gen_dynamic_dict_list(_shape, _range, _dtype, _list_num, _format="ND"):
    _dict = {"shape": _shape, "dtype": _dtype, "format": _format, "ori_shape": _shape,"ori_format": _format, "range":_range}

    dict_list = [_dict for _ in range(_list_num)]
    return dict_list

case1 = {"params": [{"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
                    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 100)]},
                    ],
         "case_name": "selu_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
# ut_case.add_case(["Ascend310"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend310")
