#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("ConcatOffset", "impl.concat_offset", "op_select_format")


def gen_dynamic_dict_list(_shape, _range, _dtype, _list_num, _format="ND"):
    _dict = {"shape": _shape, "dtype": _dtype, "format": _format, "ori_shape": _shape,"ori_format": _format, "range":_range}

    dict_list = [_dict for _ in range(_list_num)]
    return dict_list

case1 = {"params": [{"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 10)]},
                    gen_dynamic_dict_list([-1], [(1, 16)], "int32", 10),
                    gen_dynamic_dict_list([-1], [(1, 16)], "int32", 10)
                    ],
         "case_name": "concat_offset_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "range":[(1, 10)]},
                    gen_dynamic_dict_list([7], [(1, 16)], "int32", 10),
                    gen_dynamic_dict_list([7], [(1, 16)], "int32", 10)
                    ],
         "case_name": "concat_offset_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910", "Ascend310"], case1)
ut_case.add_case(["Ascend910", "Ascend310"], case2)

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend310")

