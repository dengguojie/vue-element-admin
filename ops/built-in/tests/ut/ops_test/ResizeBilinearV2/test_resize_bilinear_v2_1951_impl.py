# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ut for resize
"""
import tbe
from tbe.common.platform import set_current_compile_soc_info
from op_test_frame.ut import OpUT
from impl.dynamic.resize_bilinear_v2 import resize_bilinear_v2


ut_case = OpUT("ResizeBilinearV2", "impl.dynamic.resize_bilinear_v2", "resize_bilinear_v2")


def get_special_shape(ori_shape, ori_format, dst_format, align_num=16):
    """
    get_special_shape
    """
    def _ceil_div(dim):
        return (dim + align_num - 1) // align_num

    dst_shape = []
    if dst_format in ("FRACTAL_NZ",):
        dst_shape = ori_shape[:-2] + [
            _ceil_div(ori_shape[-1]),
            _ceil_div(ori_shape[-2]), align_num, align_num
        ]
    if dst_format in ("NC1HWC0",):
        foramt_dim = dict(zip(list(ori_format), ori_shape))
        dst_shape = [foramt_dim["N"],
                     _ceil_div(foramt_dim["C"]),
                     foramt_dim["H"],
                     foramt_dim["W"],
                     align_num]

    dst_shape_len = len(dst_shape)

    return dst_shape if dst_shape_len != 0 else ori_shape


def tensor_dict(tensor_ori_shape, tensor_ori_format, tensor_type, tensor_format=None, is_output=False):
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
    return gen_dict


def test_1951_fp16_small_shape(test_arg):
    """
    test_1951_fp16_small_shape

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    set_current_compile_soc_info('Ascend710')
    image_shape = [-1, 16, 16, 16]
    image_dtype = "float16"
    size = [16, 16]
    output_shape = [image_shape[0], image_shape[1], size[0], size[1]]

    with tbe.common.context.op_context.OpContext("dynamic"):
        resize_bilinear_v2(tensor_dict(image_shape, "NCHW", image_dtype, tensor_format="NC1HWC0"),
                           tensor_dict([2], "NCHW", "int32"),
                           tensor_dict(output_shape, "NCHW", "float32", tensor_format="NC1HWC0", is_output=True))

    set_current_compile_soc_info(test_arg)


def test_1951_fp32_small_shape(test_arg):
    """
    test_1951_fp16_small_shape

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    set_current_compile_soc_info('Ascend710')
    image_shape = [-1, 17, 32, 22]
    image_dtype = "float32"
    size = [16, 16]
    output_shape = [image_shape[0], image_shape[1], size[0], size[1]]

    with tbe.common.context.op_context.OpContext("dynamic"):
        resize_bilinear_v2(tensor_dict(image_shape, "NCHW", image_dtype, tensor_format="NC1HWC0"),
                           tensor_dict([2], "NCHW", "int32"),
                           tensor_dict(output_shape, "NCHW", "float32", tensor_format="NC1HWC0", is_output=True),
                           True, False, "test_1951_fp32_small_shape_tf")
        resize_bilinear_v2(tensor_dict(image_shape, "NCHW", image_dtype, tensor_format="NC1HWC0"),
                           tensor_dict([2], "NCHW", "int32"),
                           tensor_dict(output_shape, "NCHW", "float32", tensor_format="NC1HWC0", is_output=True),
                           True, True, "test_1951_fp32_small_shape_tt")
        resize_bilinear_v2(tensor_dict(image_shape, "NCHW", image_dtype, tensor_format="NC1HWC0"),
                           tensor_dict([2], "NCHW", "int32"),
                           tensor_dict(output_shape, "NCHW", "float32", tensor_format="NC1HWC0", is_output=True),
                           False, False, "test_1951_fp32_small_shape_ff")
    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(test_func=test_1951_fp16_small_shape)
ut_case.add_cust_test_func(test_func=test_1951_fp32_small_shape)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run('Ascend910A')
