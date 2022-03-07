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
from impl.dynamic.pad import check_supported

ut_case = OpUT("Pad", "impl.dynamic.pad", "check_supported")


# pylint: disable=unused-argument
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


def tensor_dict(tensor_ori_shape, tensor_ori_format, tensor_type,
                tensor_format=None, is_output=False, const_value=None):
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


def test_check_supported(test_arg):
    """
    test_check_supported

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    x1_shape = [1, 16, 3, 7]
    padding1_shape = [4, 2]
    x1_dtype = "float32"
    ret1, _ = check_supported(
        tensor_dict(x1_shape, "NCHW", x1_dtype),
        tensor_dict(padding1_shape, "NCHW", "int32", const_value=[16, 16, 16, 16, 16, 16, 16, 16]),
        tensor_dict(x1_shape, "NCHW", x1_dtype, is_output=True),
    )
    assert ret1

    x2_shape = [1, 3204, 256]
    padding2_shape = [3, 2]
    x2_dtype = "float32"
    ret2, _ = check_supported(
        tensor_dict(x2_shape, "NCHW", x2_dtype),
        tensor_dict(padding2_shape, "NCHW", "int32", const_value=[16, 16, 16, 16, 16, 16]),
        tensor_dict(x2_shape, "NCHW", x2_dtype, is_output=True),
    )
    assert not ret2

    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(test_func=test_check_supported)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
