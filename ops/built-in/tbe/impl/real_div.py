# Copyright 2019 Huawei Technologies Co., Ltd
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
real_div
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base

SIZE_SIXTEEN = 16


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name
def _is_last_two_axis_16_multiple(shape):
    """
    check whether divided by 16.

    Parameters
    ----------
    shape: list or tuple

    Returns:
    -------
    None
    """
    if shape[-1] == 0 or shape[-2] == 0:
        expected_value = "not equal to 0"
        real_value = "equal to 0"
        error_manager_vector.raise_err_input_value_invalid("real_div", "shape[-1] and shape[-2]", expected_value,
                                                           real_value)

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


# 'pylint: disable=too-many-locals,too-many-statements,too-many-boolean-expressions
def op_select_format(input_x, input_y, output_z, kernel_name="real_div"):
    """
    select format dynamically\n
    op_select_format support desc:

    1.when input x's ori_shape is 4, and bias's shape is not 1.\n
    The Op Bias can support
    ND/ND = ND,
    NC1HWC0/NC1HWC0 = NC1HWC0.

        for example:
        inputs:
            x        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
            bias     ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
        outputs:
            y        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"

    2.In other scenes, all input(x, bias) only support ND.

        for example:
        inputs:
            x        ori shape = [2] ori_format = "ND"
            bias     ori shape = [2] ori_format = "ND"
        outputs:
            y        ori shape = [2] ori_format = "ND"
    """
    shape_x = input_x.get("ori_shape")
    shape_y = input_y.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)
    format_x = input_x.get("ori_format")
    format_y = input_y.get("ori_format")

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]

    x_is_which_format = {
        "is_5d": len(shape_x) == 5 and format_x in format_5d_list,
        "is_4d": len(shape_x) == 4 and format_x in format_4d_list,
        "is_scalar": len(shape_x) == 1 and shape_x[0] == 1
    }
    y_is_which_format = {
        "is_5d": len(shape_y) == 5 and format_y in format_5d_list,
        "is_4d": len(shape_y) == 4 and format_y in format_4d_list,
        "is_scalar": len(shape_y) == 1 and shape_y[0] == 1
    }

    x_info = {"shape": shape_x, "format": format_x, "dim_n": 1, "dim_c": 1}
    y_info = {"shape": shape_y, "format": format_y, "dim_n": 1, "dim_c": 1}

    if (x_is_which_format["is_4d"] and y_is_which_format["is_4d"]) or \
            (len(shape_x) == 5 and len(shape_y) == 5 and format_x == format_y and format_x in format_5d_list):
        x_info["dim_c"] = shape_x[format_x.index("C")]
        x_info["dim_n"] = shape_x[format_x.index("N")]
        y_info["dim_c"] = shape_y[format_y.index("C")]
        y_info["dim_n"] = shape_y[format_y.index("N")]
    if len(shape_y) == 1 and x_is_which_format["is_4d"]:
        x_info["dim_c"] = shape_x[format_x.index("C")]
        x_info["dim_n"] = shape_x[format_x.index("N")]
    if len(shape_x) == 1 and y_is_which_format["is_4d"]:
        y_info["dim_c"] = shape_y[format_y.index("C")]
        y_info["dim_n"] = shape_y[format_y.index("N")]

    format_support_flag = {
        ("ND", "ND", "ND"): 1,
        ("FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ"): 0,
        ("NDC1HWC0", "NDC1HWC0", "NDC1HWC0"): 0,
        ("FRACTAL_Z_3D", "FRACTAL_Z_3D", "FRACTAL_Z_3D"): 0,
        ("NC1HWC0", "NC1HWC0", "NC1HWC0"): 0,
        ("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z"): 0,
        ("NC1HWC0", "ND", "NC1HWC0"): 0,
        ("FRACTAL_Z", "ND", "FRACTAL_Z"): 0,
        ("ND", "NC1HWC0", "NC1HWC0"): 0,
        ("ND", "FRACTAL_NZ", "FRACTAL_NZ"): 0,
        ("FRACTAL_NZ", "ND", "FRACTAL_NZ"): 0,
        ("ND", "FRACTAL_Z", "FRACTAL_Z"): 0
    }

    # FRACTAL_NZ    /     FRACTAL_NZ
    # NDC1HWC0      /     NDC1HWC0
    # FRACTAL_Z_3D  /     FRACTAL_Z_3D
    # NC1HWC0       /     NC1HWC0
    # FRACTAL_Z     /     FRACTAL_Z
    _is_support_same_formats(x_info, y_info, y_is_which_format, x_is_which_format, format_support_flag)

    # FRACTAL_NZ    /     ND          ->    FRACTAL_NZ       /     ND
    # ND            /     FRACTAL_NZ  ->    ND               /     FRACTAL_NZ
    # ND & scalar   /     FRACTAL_NZ  ->    ND               /     FRACTAL_NZ
    # FRACTAL_NZ    /     ND & scalar ->    FRACTAL_NZ       /     ND
    # 5HD           /     ND          ->    5HD              /     ND
    # 4D            /     scalar      ->    FRACTAL_Z & 5HD  /     ND
    # scalar        /     4D          ->    ND               /     FRACTAL_Z & 5HD
    _is_support_diff_formats(x_info, y_info, x_is_which_format, y_is_which_format, format_4d_list, format_support_flag)

    # gen format and dtype
    format_list_input0 = [format_tuple[0] for format_tuple in format_support_flag if format_support_flag[format_tuple]]
    format_list_input1 = [format_tuple[1] for format_tuple in format_support_flag if format_support_flag[format_tuple]]
    format_list_output = [format_tuple[2] for format_tuple in format_support_flag if format_support_flag[format_tuple]]

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_list = ["float16"]
    else:
        dtype_list = ["float16", "float32"]
    dtype_total = []
    for dtype in dtype_list:
        dtype_total = dtype_total + [dtype] * len(format_list_output)
    len_dtype_list = len(dtype_list)
    format_list_input0 = format_list_input0 * len_dtype_list
    format_list_input1 = format_list_input1 * len_dtype_list
    format_list_output = format_list_output * len_dtype_list
    unknownshape_format_list = ["ND"] * len(dtype_total)

    if -1 in shape_x or -1 in shape_y:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x1",
                                               datatype=",".join(dtype_total),
                                               format=",".join(format_list_input0),
                                               unknownshape_format=",".join(unknownshape_format_list))
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x2",
                                               datatype=",".join(dtype_total),
                                               format=",".join(format_list_input1),
                                               unknownshape_format=",".join(unknownshape_format_list))
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=",".join(dtype_total),
                                                format=",".join(format_list_output),
                                                unknownshape_format=",".join(unknownshape_format_list))
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x1",
                                               datatype=",".join(dtype_total),
                                               format=",".join(format_list_input0))
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x2",
                                               datatype=",".join(dtype_total),
                                               format=",".join(format_list_input1))
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="y",
                                                datatype=",".join(dtype_total),
                                                format=",".join(format_list_output))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _is_support_same_formats(x_info, y_info, y_is_which_format, x_is_which_format, format_support_flag):
    # FRACTAL_NZ/FRACTAL_NZ
    all_append_format_fractal_nz_cond_list = []
    all_append_format_fractal_nz_cond_list.append(
        (len(x_info["shape"]) >= 2 and len(y_info["shape"]) >= 2 and x_info["shape"][-2:] == y_info["shape"][-2:])
        and (_is_last_two_axis_16_multiple(y_info["shape"])))
    format_support_flag[("FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ")] = any(all_append_format_fractal_nz_cond_list)

    # NDC1HWC0/NDC1HWC0
    all_append_format_ndc1hwc0_cond_list = []
    all_append_format_ndc1hwc0_cond_list.append(
        (x_is_which_format["is_5d"] and y_is_which_format["is_5d"] and x_info["format"] == y_info["format"])
        and (x_info["dim_c"] % 16 == 0 and y_info["dim_c"] % 16 == 0))
    format_support_flag[("NDC1HWC0", "NDC1HWC0", "NDC1HWC0")] = any(all_append_format_ndc1hwc0_cond_list)

    # FRACTAL_Z_3D/FRACTAL_Z_3D
    all_append_format_fractal_z_3d_cond_list = []
    all_append_format_fractal_z_3d_cond_list.append(
        (x_is_which_format["is_5d"] and y_is_which_format["is_5d"] and x_info["format"] == y_info["format"])
        and (x_info["dim_c"] % 16 == 0 and y_info["dim_c"] % 16 == 0)
        and (x_info["dim_n"] % 16 == 0 and y_info["dim_n"] % 16 == 0))
    format_support_flag[("FRACTAL_Z_3D", "FRACTAL_Z_3D", "FRACTAL_Z_3D")] = \
        any(all_append_format_fractal_z_3d_cond_list)

    # NC1HWC0/NC1HWC0
    _is_support_5d_5d_5d(x_info, y_info, x_is_which_format, y_is_which_format, format_support_flag)

    # FRACTAL_Z/FRACTAL_Z
    _is_support_fz_fz_fz(x_info, y_info, x_is_which_format, y_is_which_format, format_support_flag)


def _is_support_diff_formats(x_info, y_info, x_is_which_format, y_is_which_format, format_4d_list, format_support_flag):
    # FRACTAL_NZ/ND -> FRACTAL_NZ/ND
    # ND/FRACTAL_NZ -> ND/FRACTAL_NZ
    # if inputs are [shape_0, shape_1] == [shape_x, shape_y], is_x_y is True, else False
    _is_any_fractal_nz_res = _is_any_fractal_nz(x_info["shape"], y_info["shape"], is_x_y=True)
    if _is_any_fractal_nz_res is not None:
        format_0, format_1, format_2 = _is_any_fractal_nz_res
        format_support_flag[(format_0, format_1, format_2)] = 1
    # ND/FRACTAL_NZ & scalar/FRACTAL_NZ -> ND/FRACTAL_NZ
    # FRACTAL_NZ/ND & FRACTAL_NZ/scalar -> FRACTAL_NZ/ND
    _is_any_fractal_nz_res = _is_any_fractal_nz(y_info["shape"], x_info["shape"], is_x_y=False)
    if _is_any_fractal_nz_res is not None:
        format_0, format_1, format_2 = _is_any_fractal_nz_res
        format_support_flag[(format_0, format_1, format_2)] = 1

    # 5HD/ND -> 5HD/ND
    if len(x_info["shape"]) == 1 and x_info["format"] in format_4d_list and x_info["shape"][0] % 16 == 0 \
            and y_is_which_format["is_scalar"]:
        format_support_flag[("NC1HWC0", "ND", "NC1HWC0")] = 1

    # 4D/scalar --> FRACTAL_Z/ND & 5HD/ND
    if x_is_which_format["is_4d"] and y_is_which_format["is_scalar"]:
        format_support_flag[("NC1HWC0", "ND", "NC1HWC0")] = 1
        format_support_flag[("FRACTAL_Z", "ND", "FRACTAL_Z")] = 1
    # scalar/4D --> ND/FRACTAL_Z & ND/5HD
    if y_is_which_format["is_4d"] and x_is_which_format["is_scalar"]:
        if y_info["dim_c"] % 16 == 0:
            format_support_flag[("ND", "NC1HWC0", "NC1HWC0")] = 1
        if y_info["dim_c"] % 16 == 0 and y_info["dim_n"] % 16 == 0:
            format_support_flag[("ND", "FRACTAL_Z", "FRACTAL_Z")] = 1


# 'pylint: disable=too-many-locals,too-many-branches,too-many-boolean-expressions
def _is_support_5d_5d_5d(x_info, y_info, x_is_which_format, y_is_which_format, format_support_flag):
    format_x, shape_x, x_cdim = x_info["format"], x_info["shape"], x_info["dim_c"]
    format_y, shape_y, y_cdim = y_info["format"], y_info["shape"], y_info["dim_c"]
    if len(shape_x) == 1 and len(shape_y) == 1 and shape_x[0] % 16 == 0 and shape_y[0] % 16 == 0:
        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
    if len(shape_y) == 1 and x_is_which_format["is_4d"] and shape_y[0] % 16 == 0 and x_cdim % 16 == 0:
        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
    if len(shape_x) == 1 and y_is_which_format["is_4d"] and shape_x[0] % 16 == 0 and y_cdim % 16 == 0:
        format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
    if x_is_which_format["is_4d"] and y_is_which_format["is_4d"]:
        if x_cdim % 16 == 0 and y_cdim % 16 == 0:
            if format_x == format_y == "NCHW":
                if (shape_x[1] == shape_y[1] or shape_x[1] == 16 or shape_y[1] == 16) \
                        or (shape_x[0] == shape_y[0] or shape_x[0] == 1 or shape_y[0] == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
            if format_x == format_y in ("HWCN", "NHWC"):
                if shape_x[0] == shape_y[0] and (shape_x[1] == 1 or shape_y[1] == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if shape_x[1] == shape_y[1] and (shape_x[0] == 1 or shape_y[0] == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if shape_x[0] == shape_y[0] and shape_x[1] == shape_y[1]:
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if (shape_x[1] == shape_x[0] == 1) or (shape_y[0] == shape_y[1] == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
                if (shape_x[0] == shape_y[1] == 1) or (shape_x[1] == shape_y[0] == 1):
                    format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1


def _is_support_fz_fz_fz(x_info, y_info, x_is_which_format, y_is_which_format, format_support_flag):
    all_append_format_fractal_z_cond_list = []
    all_append_format_fractal_z_cond1 = ((x_is_which_format["is_4d"] and y_is_which_format["is_4d"])
                                         and (x_info["dim_c"] % 16 == 0 and x_info["dim_n"] % 16 == 0)
                                         and (y_info["dim_c"] % 16 == 0 and y_info["dim_n"] % 16 == 0)
                                         and (x_info["format"] == y_info["format"] == "NHWC")
                                         and (list(x_info["shape"]) == list(y_info["shape"])))
    all_append_format_fractal_z_cond_list.append(all_append_format_fractal_z_cond1)
    all_append_format_fractal_z_cond2 = ((x_is_which_format["is_4d"] and y_is_which_format["is_4d"])
                                         and (x_info["dim_c"] % 16 == 0 and x_info["dim_n"] % 16 == 0)
                                         and (y_info["dim_c"] % 16 == 0 and y_info["dim_n"] % 16 == 0)
                                         and (x_info["format"] == y_info["format"] == "NCHW")
                                         and (list(x_info["shape"]) == list(y_info["shape"])))
    all_append_format_fractal_z_cond_list.append(all_append_format_fractal_z_cond2)
    all_append_format_fractal_z_cond3 = ((x_is_which_format["is_4d"] and y_is_which_format["is_4d"])
                                         and (x_info["dim_c"] % 16 == 0 and x_info["dim_n"] % 16 == 0)
                                         and (y_info["dim_c"] % 16 == 0 and y_info["dim_n"] % 16 == 0)
                                         and (x_info["format"] == y_info["format"] == "HWCN")
                                         and (x_info["shape"][0] * x_info["shape"][1]
                                              == y_info["shape"][0] * y_info["shape"][1]))
    all_append_format_fractal_z_cond_list.append(all_append_format_fractal_z_cond3)
    format_support_flag[("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z")] = any(all_append_format_fractal_z_cond_list)


# 'pylint: disable=too-many-locals,too-many-statements,too-many-boolean-expressions
def _is_any_fractal_nz(shape_0, shape_1, is_x_y):
    """
    Check whether is any FRACTAL_NZ format, and return the format tuple

    Parameters
    ----------
    shape_0: list or tuple
    shape_1: list or tuple
    is_x_y： bool
        if inputs are [shape_0, shape_1] == [shape_x, shape_y], is_x_y is True, else False

    Returns
    -------
    Str
    "FRACTAL_NZ", "ND", "FRACTAL_NZ" or
    "ND", "FRACTAL_NZ", "FRACTAL_NZ"
    """
    # FRACTAL_NZ/ND -> FRACTAL_NZ/ND
    # ND/FRACTAL_NZ -> ND/FRACTAL_NZ
    if len(shape_0) >= 2 and len(shape_1) >= 2:
        if _is_last_two_axis_16_multiple(shape_0) and (not _is_last_two_axis_16_multiple(shape_1)):
            if is_x_y:
                return "FRACTAL_NZ", "ND", "FRACTAL_NZ"
            else:
                return "ND", "FRACTAL_NZ", "FRACTAL_NZ"

    # ND/FRACTAL_NZ & scalar/FRACTAL_NZ -> ND/FRACTAL_NZ
    # FRACTAL_NZ/ND & FRACTAL_NZ/scalar -> FRACTAL_NZ/ND
    if (len(shape_0) == 1 and len(shape_1) >= 2 and shape_0[-1] == shape_1[-1]) or \
            (len(shape_0) == 1 and len(shape_1) >= 2 and shape_0[-1] == 1):
        if is_x_y and _is_last_two_axis_16_multiple(shape_1):
            return "ND", "FRACTAL_NZ", "FRACTAL_NZ"
        if not is_x_y:
            return "FRACTAL_NZ", "ND", "FRACTAL_NZ"
    return None


def _check_format(x, y):
    """
    funtion to check format

    Parameters
    ----------
    x: dict
        dict of x, include keys(shape and dtype).
    y: dict
        dict of x, include keys(shape and dtype).

    Returns:
    -------
    format_pattern: int
    """
    format_pattern = 0
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = shape_util.scalar2tensor_one(shape1)
    shape2 = shape_util.scalar2tensor_one(shape2)
    check_list = [["FRACTAL_NZ", "ND"], ["ND", "FRACTAL_NZ"], ["FRACTAL_NZ", "NHWC"], ["NHWC", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NCHW"], ["NCHW", "FRACTAL_NZ"]]
    if list_format == check_list[0] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[1] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[2] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[3] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[4] and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[5] and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2

    return format_pattern


def _infer_shape(format_pattern, x, y):
    """
    funtion to infer shape

    Parameters
    ----------
    format_pattern: int
    x: dict
        dict of x, include keys(shape and dtype).
    y: dict
        dict of x, include keys(shape and dtype).

    Returns:
    -------
    shape_x: shape of x
    shape_y: shape of y
    """
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    if format_pattern == 1:
        ori_shape_x, shape_y, shape_max = shape_util.broadcast_shapes(ori_shape_x,
                                                                      shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
        if shape_y[-2] == 1 and shape_y[-1] == ori_shape_x[-1]:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-3] = 1
            shape_y[-1] = shape_x[-1]
            shape_y[-4] = shape_x[-4]

        elif shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-4] = 1
            shape_y[-2] = shape_x[-2]
            shape_y[-3] = shape_x[-3]

        elif shape_y[-2] == shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)

    elif format_pattern == 2:
        shape_x, ori_shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                      ori_shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
        if shape_x[-2] == 1 and shape_x[-1] == ori_shape_y[-1]:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-3] = 1
            shape_x[-1] = shape_y[-1]
            shape_x[-4] = shape_y[-4]

        elif shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-4] = 1
            shape_x[-2] = shape_y[-2]
            shape_x[-3] = shape_y[-3]

        elif shape_x[-2] == shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)

    return shape_x, shape_y


@tbe_platform.fusion_manager.fusion_manager.register("real_div")
def real_div_compute(x1, x2, y, kernel_name="real_div"):
    """
    calculating data's realdiv, c = a / b

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is real_div

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                              shape_y,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")
    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)
    res = tbe.vdiv(data_x, data_y)

    return res


# 'pylint: disable=too-many-locals,too-many-statements,too-many-boolean-expressions
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def real_div(x1, x2, y, kernel_name="real_div"):
    """
    algorithm: real_div
    calculating data's real_div, c = a / b

    Parameters
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is real_div

    Returns
    -------
    None
    """
    format_pattern = _check_format(x1, x2)
    shape_x, shape_y = _infer_shape(format_pattern, x1, x2)
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)
    para_check.check_shape(shape_x, param_name="x1")
    para_check.check_shape(shape_y, param_name="x2")

    check_tuple = ("float16", "float32")
    input_data_type = x1.get("dtype").lower()
    para_check.check_dtype(input_data_type, check_tuple, param_name="x1")
    input_data_type_x2 = x2.get("dtype").lower()
    para_check.check_dtype(input_data_type_x2, check_tuple, param_name="x2")

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                              shape_y,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=input_data_type)

    res = real_div_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": (data_x, data_y, res)}

    tbe.cce_build_code(schedule, config)
