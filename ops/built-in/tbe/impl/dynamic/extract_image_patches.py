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
# ============================================================================
"""
extract_image_patches
"""
# 'pylint: disable=too-many-lines
import math
import tbe as mytbe
from tbe.dsl.base import operation
from tbe.dsl.compute import common
from tbe.common.utils.errormgr import error_manager_vector
from impl.im2col_common_func import im2col_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.util_common import check_load3d_w_out_1_support


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    BLOCK_SIZE = 16
    BLOCK_SIZE_INT8 = 32
    DOUBLE_BUFFER = 2
    FP16_SIZE = 2
    INT8_SIZE = 1
    MAX_INT32_VALUE = 2**31 - 1
    SIZE_L1 = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


def param_check(ksizes, strides, dilates, padding, fmap_h, fmap_w, fmap_c0, type_size, kernel_name):
    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    out_h, padding_h_top, padding_h_bottom = \
        common.tf_get_windowed_output_size_verbose_v2(
            fmap_h, kernel_h, dilate_h, stride_h, padding)
    out_w, padding_w_before, padding_w_after = common.tf_get_windowed_output_size_verbose_v2(
        fmap_w, kernel_w, dilate_w, stride_w, padding)
    if out_h <= 0 or out_w <= 0:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "out_h and out_w", ">0",
                                                           str([out_h, out_w]))
    if padding_h_top >= 256 or padding_h_bottom >= 256:
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "padding_h_top and padding_h_bottom", "<256",
                                                           str([padding_h_top, padding_h_bottom]))
    if padding_w_before >= 256 or padding_w_after >= 256:
        error_manager_vector.raise_err_input_value_invalid(kernel_name,
                                                           "padding_w_before and padding_w_after", "<256",
                                                           str([padding_w_before, padding_w_after]))

    if not check_load3d_w_out_1_support() and out_h != 1 and out_w == 1:
        if fmap_w + padding_w_before + padding_w_after - ((kernel_w - 1) * dilate_w + 1) < stride_w:
            error_result = fmap_w + padding_w_before + padding_w_after - ((kernel_w - 1) * dilate_w + 1)
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "Platform cloud and DC",
                "fmap_w + pad_l + pad_r - ((kernel_w - 1) * dilate_w + 1) >= stride_w",
                str(error_result))

    # min cut_h
    dilated_kernel_h = (kernel_h - 1) * dilate_h + 1
    cut_h_col = (Constant.BLOCK_SIZE // math.gcd(out_w, Constant.BLOCK_SIZE) - 1) * \
                stride_h + 1 + dilated_kernel_h // 2
    if cut_h_col > fmap_h:
        cut_h_col = fmap_h

    cut_w_row_s = (Constant.BLOCK_SIZE - 1) * stride_w + 1
    cut_h_row_s = ((cut_w_row_s - 1) // fmap_w + 1) * stride_h + 1
    min_cut_h = min(cut_h_col, cut_h_row_s)

    if min_cut_h * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER > Constant.SIZE_L1:
        error_manager_vector.raise_err_specific_reson(
            kernel_name, "Input size is too large load to L1, while cut h, need size: %d" %
                         (min_cut_h * fmap_w * fmap_c0 * type_size * Constant.DOUBLE_BUFFER))


# 'pylint:disable=unused-argument,too-many-arguments,too-many-locals
@mytbe.common.register.register_param_generalization("ExtractImagePatches")
def extract_image_patch_generalization(fmap, c_in_real, ksizes, strides, dilates,
                                       padding, output_res, generalize_config=None):
    """
    extract image patch generalization
    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of fmap
    c_in_real: real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    output_res: output tensor
    generalize_config: dict
        single item under "keep_rank" mode and multiple under "all_shape"

    Returns
    -------
    result:standardized fmap ksizes strides dilates padding output_res
    """
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    data_format = fmap.get("ori_format")
    if data_format == "NCHW":
        c_dim_index = 1
        h_dim_index = 2
        w_dim_index = 3
        c_dim = fmap["shape"][c_dim_index]
    elif data_format == "NHWC":
        h_dim_index = 1
        w_dim_index = 2
        c_dim_index = 3
        c_dim = fmap["shape"][c_dim_index]
    elif data_format == "NC1HWC0":
        h_dim_index = 2
        w_dim_index = 3
        c_dim = c_in_real

    result = []
    h_dim = fmap["shape"][h_dim_index]
    w_dim = fmap["shape"][w_dim_index]
    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    shape_x = (-1, h_dim, w_dim, c_dim)
    range_x = [(1, -1), (h_dim, h_dim), (w_dim, w_dim), (c_dim, c_dim)]

    def static_shape_to_range(raw_static_shape):
        static_range = [(shape_i, shape_i) for shape_i in raw_static_shape]
        return static_range

    fmap["shape"], fmap["ori_shape"] = shape_x, shape_x
    fmap["range"], fmap["ori_range"] = range_x, range_x

    ksizes_dict = {"shape": ksizes, "ori_shape": ksizes, "range": [], "ori_range": []}
    ksizes_range = static_shape_to_range(ksizes_dict.get("shape"))
    ksizes_dict["range"], ksizes_dict["ori_range"] = ksizes_range, ksizes_range

    strides_dict = {"shape": strides, "ori_shape": strides, "range": [], "ori_range": []}
    strides_range = static_shape_to_range(strides_dict.get("shape"))
    strides_dict["range"], strides_dict["ori_range"] = strides_range, strides_range

    dilates_dict = {"shape": dilates, "ori_shape": dilates, "range": [], "ori_range": []}
    dilates_range = static_shape_to_range(dilates_dict.get("shape"))
    dilates_dict["range"], dilates_dict["ori_range"] = dilates_range, dilates_range

    padding_dict = {"shape": padding, "ori_shape": padding, "range": [], "ori_range": []}
    padding_range = static_shape_to_range(padding_dict.get("shape"))
    padding_dict["range"], padding_dict["ori_range"] = padding_range, padding_range

    output_res["shape"], output_res["ori_shape"] = output_res["shape"], output_res["shape"]
    output_res_range = static_shape_to_range(output_res.get("shape"))
    output_res["range"], output_res["ori_range"] = output_res_range, output_res_range

    result.append([fmap, ksizes_dict, strides_dict, dilates_dict, padding_dict, output_res])
    return result


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
def extract_image_patches_compute(fmap, c_in_real, ksizes, strides, dilates, padding,
                                  kernel_name="extract_image_patches"):
    """
    ops compute

    Parameters
    ----------
    fmap : TVM tensor
        the placeholder of fmap
    c_in_real : real c size of input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str kernel name

    Returns
    -------
    output_res
    workspace_res
    workspace_shape
    """
    # fmap's format is NC1HWC0
    fmap_shape = fmap.shape

    fmap_h = fmap_shape[2].value
    fmap_w = fmap_shape[3].value

    _, kernel_h, kernel_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilate_h, dilate_w, _ = dilates

    out_h, padding_h_before, padding_h_after = common.tf_get_windowed_output_size_verbose_v2(
        fmap_h, kernel_h, dilate_h, stride_h, padding)
    out_w, padding_w_before, padding_w_after = common.tf_get_windowed_output_size_verbose_v2(
        fmap_w, kernel_w, dilate_w, stride_w, padding)

    pads = (padding_h_before, padding_h_after, padding_w_before, padding_w_after)
    ksize = (kernel_h, kernel_w)
    stride = (stride_h, stride_w)
    dilate = (dilate_h, dilate_w)

    output_res, workspace_res, workspace_shape = im2col_compute(fmap, c_in_real, ksize, stride, dilate, pads, out_h,
                                                                out_w)

    return output_res, workspace_res, workspace_shape


def _classify(images, ksizes, strides, padding):
    ins = ["common"]
    images_range_0 = images.get("range")[0]
    images_range_0_l = 0 if images_range_0[0] is None else int(images_range_0[0])
    if isinstance(images_range_0[1], int):
        images_range_0_r = Constant.MAX_INT32_VALUE if images_range_0[1] is None else int(images_range_0[1]) + 1
    else:
        images_range_0_r = Constant.MAX_INT32_VALUE
    if 128 in range(images_range_0_l, images_range_0_r) and list(ksizes) == [1, 2, 2, 1] and \
            list(strides) == [1, 2, 2, 1] and padding == "VALID":
        ins.append("special")
    return ins


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals,too-many-branches
@register_operator("ExtractImagePatches", pattern="ExtractImagePatches")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def extract_image_patches(images, y, ksizes, strides, dilates, padding, kernel_name="extract_image_patches"):
    """
    calculating data

    Parameters
    ----------
    images : dict
        shape and dtype of input, only support float16
    y : dict
        shape and dtype of output, should be same shape and type as input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    kernel_name : str
        kernel name, default value is "extract_image_patches"

    Returns
    -------
    None
    """
    shape_input_4d = images.get("ori_shape")
    dtype_input = images.get("dtype")
    dtype_input = dtype_input.lower()
    if dtype_input in ('int8', 'uint8'):
        align_block_size = Constant.BLOCK_SIZE_INT8
        type_size = Constant.INT8_SIZE
    else:
        align_block_size = Constant.BLOCK_SIZE
        type_size = Constant.FP16_SIZE

    data_format = images.get('ori_format')
    format_list = ('NHWC', 'NCHW')
    if data_format not in format_list:
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", format_list, data_format)
    if len(ksizes) != 4 or len(strides) != 4 or len(dilates) != 4:
        error_manager_vector.raise_err_check_params_rules(kernel_name, 'input params invalide',
                                                          ['ksizes', 'strides', 'dilates'], [ksizes, strides, dilates])
    # NCHW -> NHWC
    if data_format == 'NCHW':
        shape_input_4d = (shape_input_4d[0], shape_input_4d[2], shape_input_4d[3], shape_input_4d[1])
        ksizes = (ksizes[0], ksizes[2], ksizes[3], ksizes[1])
        strides = (strides[0], strides[2], strides[3], strides[1])
        dilates = (dilates[0], dilates[2], dilates[3], dilates[1])

    dim_0 = tbe.var("dim_0")
    multi_core_factor_0 = tbe.var("multi_core_factor_0")

    ins = _classify(images, ksizes, strides, padding)
    schedules, tensors = [], []
    for i in ins:
        if i == "special":
            dim_0 = 128
            multi_core_factor_0 = 16
        operation.get_context().add("mode", i)

        _, fmap_h, fmap_w, fmap_c = shape_input_4d
        fmap_c1 = (fmap_c + align_block_size - 1) // align_block_size
        fmap_c0 = align_block_size
        shape_input = (dim_0, fmap_c1, fmap_h, fmap_w, fmap_c0)
        param_check(ksizes, strides, dilates, padding, fmap_h, fmap_w, fmap_c0, type_size, kernel_name)

        data_input = tvm.placeholder(shape_input, name="data", dtype=dtype_input)
        with tbe.compute():
            output_res, workspace_res, _ = extract_image_patches_compute(data_input, fmap_c, ksizes, strides, dilates,
                                                                         padding, kernel_name)

            if fmap_c % align_block_size == 0:
                tensor_list = [data_input, output_res]
            else:
                tensor_list = [data_input, output_res, workspace_res]
            tensors.append(tensor_list)

        operation.get_context().add("multi_core_factor_0", multi_core_factor_0)
        with tvm.target.cce():
            sch = tbe.auto_schedule(output_res)
            schedules.append(sch)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
