#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("DynamicStridedSliceGrad", "impl.dynamic.strided_slice_grad", "strided_slice_grad")


def gen_ssg_case(shape, begin, end, strides, dy_shape, dtype, case_name_val, expect, input_format="ND"):
    dict_shape = {"shape": (len(shape),), "dtype": "int32",
                  "ori_shape": (len(shape),),
                  "ori_format": input_format, "format": input_format,
                  'range': [[1, 500]]}
    dict_begin = {"shape": (len(begin),), "dtype": "int32",
                  "ori_shape": (len(begin),),
                  "ori_format": input_format, "format": input_format,
                  'range': [[1, 500]]}
    dict_end = {"shape": (len(end),), "dtype": "int32",
                "ori_shape": (len(end),),
                "ori_format": input_format, "format": input_format,
                'range': [[1, 500]]}
    dict_strides = {"shape": (len(strides),), "dtype": "int32",
                    "ori_shape": (len(strides),),
                    "ori_format": input_format, "format": input_format,
                    'range': [[1, 500]]}

    dict_dy = {"shape": dy_shape, "dtype": dtype,
               "ori_shape": dy_shape,
               "ori_format": input_format, "format": input_format,
               'range': [[1, 500]] * len(dy_shape)}

    dict_out = {"shape": shape, "dtype": dtype,
                "ori_shape": shape,
                "ori_format": input_format, "format": input_format,
                'range': [[1, 500]] * len(dy_shape)}

    return {"params": [dict_shape,
                       dict_begin,
                       dict_end,
                       dict_strides,
                       dict_dy,
                       dict_out],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


dynamicrank = {"params": [
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", "dtype": "float16"},
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", "dtype": "float16"},
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", "dtype": "float16"},
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", "dtype": "float16"},
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", "dtype": "float16"},
    {"shape": (-2,), "ori_shape": (-2,), "format": "ND", "ori_format": "ND", "dtype": "float16"},
    None, None, None, None, None],
    "case_name": "strided_slice_grad_dynamic_rank_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

def test_op_check_supported_1(test_arg):
    from impl.dynamic.strided_slice_grad import check_supported
    shape = {'ori_shape': (), 'shape': (), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    begin = {'ori_shape': (-1, -1), 'shape': (2, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    end = {'ori_shape': (-1, -1), 'shape': (5, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    strides = {'ori_shape': (-1, -1), 'shape': (1, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    dy = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(shape, begin, end, strides, dy, output, new_axis_mask=5, shrink_axis_mask=5) == False:
        raise Exception("Failed to call check_supported in stridedslicegrad.")


def test_op_check_supported_2(test_arg):
    from impl.dynamic.strided_slice_grad import check_supported
    shape = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 25), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    begin = {'ori_shape': (-1, -1), 'shape': (2, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    end = {'ori_shape': (-1, -1), 'shape': (5, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    strides = {'ori_shape': (-1, -1), 'shape': (1, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    dy = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 2), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    if check_supported(shape, begin, end, strides, dy, output, new_axis_mask=3, shrink_axis_mask=3) == False:
        raise Exception("Failed to call check_supported in stridedslicegrad.")

def test_op_check_supported_3(test_arg):
    from impl.dynamic.strided_slice_grad import check_supported
    shape = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    begin = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    end = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    strides = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16', "const_value": (1, 1, 1, 3)}
    dy = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 256), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    output = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 768), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    if check_supported(shape, begin, end, strides, dy, output, new_axis_mask=0, shrink_axis_mask=0) == False:
        raise Exception("Failed to call check_supported in stridedslicegrad.")

def test_op_check_supported_4(test_arg):
    from impl.dynamic.strided_slice_grad import check_supported
    shape = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 25), 'ori_format': 'NCDHW', 'format': 'NCDHW',
            'dtype': 'float16', 'const_value': []}
    begin = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16',
             'const_value': [0, 0, 0, 0]}
    end = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16',
             'const_value': [1, 1, 1, 1]}
    strides = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16', "const_value": (1, 1, 1, 1)}
    dy = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 256), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    output = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 768), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    check_supported_result = check_supported(shape, begin, end, strides, dy,
                                             output, new_axis_mask=0, shrink_axis_mask=0)
    if check_supported_result[0]:
        raise Exception("Failed to call check_supported in stridedslicegrad.")

def test_op_check_supported_5(test_arg):
    from impl.dynamic.strided_slice_grad import check_supported
    shape = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 25), 'ori_format': 'NCDHW', 'format': 'NCDHW',
            'dtype': 'float16', 'const_value': [1, 300, 25]}
    begin = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16',
             'const_value': [0, 0, 0, 0]}
    end = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16',
             'const_value': [1, 1, 1, 1]}
    strides = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16', "const_value": (1, 1, 1, 1)}
    dy = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 256), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    output = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 768), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    check_supported_result = check_supported(shape, begin, end, strides, dy,
                                             output, new_axis_mask=0, shrink_axis_mask=0)
    if not check_supported_result[0]:
        raise Exception("Failed to call check_supported in stridedslicegrad.")

def test_op_check_supported_6(test_arg):
    from impl.dynamic.strided_slice_grad import check_supported
    shape = {'ori_shape': (-1, -1, -1), 'shape': (1, 300, 25), 'ori_format': 'NCDHW', 'format': 'NCDHW',
            'dtype': 'float16'}
    begin = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16',
             'const_value': [0, 0, 0, 0]}
    end = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16',
             'const_value': [1, 1, 1, 1]}
    strides = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16', "const_value": (1, 1, 1, 1)}
    dy = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 256), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    output = {'ori_shape': (-1, -1, -1, -1), 'shape': (32, 32, 32, 768), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    check_supported_result = check_supported(shape, begin, end, strides, dy,
                                             output, new_axis_mask=0, shrink_axis_mask=0)
    if not check_supported_result[0]:
        raise Exception("Failed to call check_supported in stridedslicegrad.")


ut_case.add_cust_test_func(test_func=test_op_check_supported_1)
ut_case.add_cust_test_func(test_func=test_op_check_supported_2)
ut_case.add_cust_test_func(test_func=test_op_check_supported_3)
ut_case.add_cust_test_func(test_func=test_op_check_supported_4)
ut_case.add_cust_test_func(test_func=test_op_check_supported_5)
ut_case.add_cust_test_func(test_func=test_op_check_supported_6)

ut_case.add_case(["Ascend910A"],
                 gen_ssg_case((1, 300, 25), (2, 1), (5, 3), (1, 1), (1, 300, 2), "float16", "case_1",
                              "success"))

ut_case.add_case(["Ascend910A"], dynamicrank)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
