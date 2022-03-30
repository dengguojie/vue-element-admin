"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic Pad ut case
"""
from op_test_frame.ut import OpUT
from tbe.common.platform import set_current_compile_soc_info
from impl.dynamic.resize import op_select_format

ut_case = OpUT("Resize", "impl.dynamic.resize", "op_select_format")


def tensor_dict(tensor_ori_shape, tensor_ori_format, tensor_type,
                tensor_format=None, is_output=False, const_value=None):
    """
    return a dict
    """
    if tensor_format is None:
        tensor_format = tensor_ori_format

    gen_dict = dict()
    gen_dict["ori_shape"] = tensor_ori_shape
    gen_dict["ori_format"] = tensor_ori_format
    gen_dict["dtype"] = tensor_type
    gen_dict["shape"] = tensor_ori_shape
    gen_dict["format"] = tensor_format
    gen_dict["range"] = [(1, 100000)] * len(tensor_ori_shape)
    param_type = "output" if is_output else "input"
    gen_dict["param_type"] = param_type
    gen_dict["param_type"] = param_type
    if const_value is not None:
        gen_dict["const_value"] = const_value
    return gen_dict


def test_op_select_format(test_arg):
    """
    test_op_select_format

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    x1_shape = [1, 16, 3, 7]
    x2_shape = [1, 1, 4, 4, 4]
    x_dtype = "float32"
    ret = op_select_format(
        tensor_dict(x1_shape, "NCHW", x_dtype),
        tensor_dict([1], "ND", "float"),
        tensor_dict([1], "ND", "float"),
        tensor_dict([1], "ND", "int32"),
        tensor_dict(x1_shape, "NCHW", x_dtype, is_output=True),
    )
    assert ret.find("NC1HWC0") != -1

    ret = op_select_format(
        tensor_dict(x2_shape, "NCDHW", x_dtype),
        tensor_dict([1], "ND", "float"),
        tensor_dict([1], "ND", "float"),
        tensor_dict([1], "ND", "int32"),
        tensor_dict(x2_shape, "NCDHW", x_dtype, is_output=True),
    )
    assert ret.find("NDC1HWC0") != -1
    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(test_func=test_op_select_format)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
