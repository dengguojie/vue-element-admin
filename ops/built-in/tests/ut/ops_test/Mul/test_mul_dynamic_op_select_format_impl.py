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

Dynamic Mul ut case for op_select_format
"""
from op_test_frame.ut import OpUT
from tbe.common.platform import set_current_compile_soc_info
from impl.util.util_tensor_dict import TensorClass
from impl.util.util_tensor_dict import get_format_for_broardcast
from impl.dynamic.mul import op_select_format

# pylint: disable=invalid-name
ut_case = OpUT("Mul", None, None)


# pylint: disable=unused-argument,too-many-arguments
def get_special_shape(ori_shape, ori_format, dst_format, align_num=16):
    """
    get_special_shape
    """

    def _ceil_div(dim):
        return (dim + align_num - 1) // align_num

    dst_shape = []
    if dst_format in ("FRACTAL_NZ",):
        dst_shape = ori_shape[:-2] + [_ceil_div(ori_shape[-1]), _ceil_div(ori_shape[-2]), align_num, align_num]
    if dst_format in ("NC1HWC0",):
        foramt_dim = dict(zip(list(ori_format), ori_shape))
        dst_shape = [foramt_dim["N"], _ceil_div(foramt_dim["C"]), foramt_dim["H"], foramt_dim["W"], align_num]

    dst_shape_len = len(dst_shape)

    return dst_shape if dst_shape_len != 0 else ori_shape


def tensor_dict(tensor_ori_shape,
                tensor_ori_format,
                tensor_type,
                tensor_format=None,
                is_output=False,
                const_value=None):
    """
    return a dict
    """
    if tensor_format is None:
        tensor_format = tensor_ori_format
    tensor_shape = get_special_shape(tensor_ori_shape, tensor_ori_format, tensor_format)

    gen_dict = dict()
    gen_dict["ori_shape"] = tensor_ori_shape
    gen_dict["ori_format"] = tensor_ori_format
    gen_dict["dtype"] = tensor_type
    gen_dict["shape"] = tensor_shape
    gen_dict["format"] = tensor_format
    gen_dict["range"] = [(1, 100000)] * len(tensor_shape)
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
    x0_shape = [16, 16, 3, 7]
    x1_shape = [16, 16, 3, 7]
    x_dtype = "float32"
    ret = op_select_format(
        tensor_dict(x0_shape, "NCHW", x_dtype),
        tensor_dict(x1_shape, "NCHW", x_dtype),
        tensor_dict(x0_shape, "NCHW", x_dtype, is_output=True),
    )
    assert ret.find("NC1HWC0") != -1
    assert ret.find("FRACTAL_Z") != -1
    assert ret.find("FRACTAL_NZ") != -1

    x0_shape = [-1, 16, -1, -1]
    x1_shape = [16, 16, -1, -1]
    x_dtype = "float32"
    ret = op_select_format(
        tensor_dict(x0_shape, "NCHW", x_dtype),
        tensor_dict(x1_shape, "NCHW", x_dtype),
        tensor_dict(x0_shape, "NCHW", x_dtype, is_output=True),
    )
    assert ret.find("NC1HWC0") != -1
    assert ret.find("FRACTAL_Z") == -1
    assert ret.find("FRACTAL_NZ") == -1

    x0_shape = [-1, 16, -1, -1]
    x1_shape = [16, -1, -1]
    x_dtype = "float32"
    ret = op_select_format(
        tensor_dict(x0_shape, "NCHW", x_dtype),
        tensor_dict(x1_shape, "NCHW", x_dtype),
        tensor_dict(x0_shape, "NCHW", x_dtype, is_output=True),
    )
    assert ret.find("NC1HWC0") == -1
    assert ret.find("FRACTAL_Z") == -1
    assert ret.find("FRACTAL_NZ") == -1

    x0_shape = [16, 16, -1, -1]
    x1_shape = [16, 16, -1, -1]
    x_dtype = "float32"
    ret = op_select_format(
        tensor_dict(x0_shape, "NCHW", x_dtype),
        tensor_dict(x1_shape, "NCHW", x_dtype),
        tensor_dict(x0_shape, "NCHW", x_dtype, is_output=True),
    )
    assert ret.find("NC1HWC0") != -1
    assert ret.find("FRACTAL_Z") == -1
    assert ret.find("FRACTAL_NZ") == -1

    x0_shape = [16, 16, -1, -1, -1]
    x1_shape = [16, 16, -1, -1, -1]
    x_dtype = "float32"
    ret = op_select_format(
        tensor_dict(x0_shape, "NCDHW", x_dtype),
        tensor_dict(x1_shape, "NCDHW", x_dtype),
        tensor_dict(x0_shape, "NCDHW", x_dtype, is_output=True),
    )
    assert ret.find("NC1HWC0") == -1
    assert ret.find("FRACTAL_Z") == -1
    assert ret.find("FRACTAL_NZ") == -1

    x0_shape = [16, 16, 16, 16, 16]
    x1_shape = [16, 16, 16, 16, 16]
    x_dtype = "float32"
    ret = op_select_format(
        tensor_dict(x0_shape, "NCDHW", x_dtype),
        tensor_dict(x1_shape, "NCDHW", x_dtype),
        tensor_dict(x0_shape, "NCDHW", x_dtype, is_output=True),
    )
    assert ret.find("FRACTAL_Z") != -1
    assert ret.find("FRACTAL_Z_3D") != -1
    assert ret.find("FRACTAL_NZ") != -1
    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
