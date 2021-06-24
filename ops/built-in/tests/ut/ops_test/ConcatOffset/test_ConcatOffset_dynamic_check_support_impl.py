#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ConcatOffset", "impl.concat_offset", "check_supported")


def gen_dynamic_dict_list(_shape, _range, _dtype, _list_num, _format="ND"):
    _dict = {"shape": _shape, "dtype": _dtype, "format": _format, "ori_shape": _shape,"ori_format": _format, "range":_range}

    dict_list = [_dict for _ in range(_list_num)]
    return dict_list


def test_op_check_supported_1(test_arg):
    from impl.dynamic.concat_offset import check_supported
    concat_dim = {'ori_shape': (-1), 'shape': (-1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    x = {'ori_shape': (-1, -1), 'shape': (2, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    y = {'ori_shape': (-1, -1), 'shape': (5, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(x, y, concat_dim) == False:
        raise Exception("Failed to call check_supported in concat_offset")


def test_get_op_support_info(test_arg):
    from impl.dynamic.concat_offset import get_op_support_info
    concat_dim = {'ori_shape': (-1), 'shape': (-1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    x = {'ori_shape': (-1, -1), 'shape': (2, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    y = {'ori_shape': (-1, -1), 'shape': (5, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    get_op_support_info(x, y, concat_dim)


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

ut_case.add_cust_test_func(test_func=test_get_op_support_info)
ut_case.add_cust_test_func(test_func=test_op_check_supported_1)
ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend310")
