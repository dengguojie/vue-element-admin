# Copyright 2020 Huawei Technologies Co., Ltd
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
add
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base
from impl.util import util_common

# constant, value is 16
SIZE_SIXTEEN = 16


# pylint: disable=too-many-nested-blocks
def _can_division_sixteen(shape):
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
        error_manager_vector.raise_err_input_value_invalid("add", "shape[-1] and shape[-2]",
                                                           expected_value, real_value)

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


# pylint: disable=invalid-name
def _can_broadcast(shape1, shape2):
    """
    check whether can broadcast or no.

    Parameters
    ----------
    shape1: list or tuple
    shape2: list or tuple

    Returns:
    -------
    None
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1

    _dv = len(shape1) - len(shape2)
    shape2 = [1] * _dv + shape2

    for shape1_i, shape2_i in zip(shape1, shape2):
        if not shape1_i == shape2_i and shape1_i != 1 and shape2_i != 1:
            return False

    return True


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-locals,too-many-branches,unused-variable
# pylint: disable=too-many-statements,too-many-boolean-expressions,consider-using-enumerate
def op_select_format(input_x, input_y, output_z, kernel_name="add"):
    """
    select format dynamically
    op_select_format support desc:
        1. when input x's ori_shape is 4, and bias's shape is not 1.
           The Op Bias can support
           ND + ND = ND,
           NC1HWC0 + NC1HWC0 = NC1HWC0,

           for example:
           inputs:
             x        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
             bias     ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"
           outputs:
             y        ori shape = [16, 16, 16, 16, 16] ori_format = "NC1HWC0"

        2. In other scenes, all input(x, bias) only support ND,
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

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_list = ["float16", "int32", "int8", "uint8"]
    else:
        dtype_list = ["float16", "float32", "int32", "int8", "uint8"]

    format_x = input_x.get("ori_format")
    format_y = input_y.get("ori_format")

    dtype_total = []
    format_nd = ["ND"]
    format_list = ["ND"]
    format_nz = ["FRACTAL_NZ"]
    format_5hd = ["NC1HWC0"]
    # 2dims add (3|4)dims,fe regards 2dims as HW,actually is WC
    if (len(shape_x) == 2 and (len(shape_y) == 3 or len(shape_y) == 4)) \
            or (len(shape_y) == 2 and (len(shape_x) == 3 or len(shape_x) == 4)):
        format_5hd = []

    len_format_list = len(dtype_list)
    add_nd_nz = False
    add_nz_nd = False
    if len(shape_x) == 1 and len(shape_y) >= 2 and shape_x[-1] == shape_y[-1]:
        for i in range(0, len(shape_y)):
            if shape_x[0] == shape_y[i] and shape_x[0] % 16 == 0:
                add_nd_nz = True
                break
    if len(shape_y) == 1 and len(shape_x) >= 2 and shape_x[-1] == shape_y[-1]:
        for i in range(0, len(shape_x)):
            if shape_y[0] == shape_x[i] and shape_y[0] % 16 == 0:
                add_nz_nd = True
                break

    if (len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list) or \
            (len(shape_x) == 5 and len(shape_y) == 5 and format_x == format_y and format_x in format_5d_list):
        x_cdim = shape_x[format_x.index("C")]
        x_wdim = shape_x[format_x.index("W")]
        x_hdim = shape_x[format_x.index("H")]
        x_ndim = shape_x[format_x.index("N")]
        y_cdim = shape_y[format_y.index("C")]
        y_wdim = shape_y[format_y.index("W")]
        y_hdim = shape_y[format_y.index("H")]
        y_ndim = shape_y[format_y.index("N")]

    if (len(shape_y) == 1 and len(shape_x) == 4) and format_x in format_4d_list:
        x_cdim = shape_x[format_x.index("C")]
        x_ndim = shape_x[format_x.index("N")]

    if (len(shape_x) == 1 and len(shape_y) == 4) and format_y in format_4d_list:
        y_cdim = shape_y[format_y.index("C")]
        y_ndim = shape_y[format_y.index("N")]

    # NDC1HWC0 FRACTAL_Z_3D
    if len(shape_x) == 5 and len(shape_y) == 5 and format_x == format_y and format_x in format_5d_list:
        if list(shape_x) == list(shape_y):
            format_list.append("NDC1HWC0")
        elif x_cdim == y_cdim:
            format_list.append("NDC1HWC0")
    if len(shape_x) == 5 and len(shape_y) == 5 and format_x == format_y and format_x in format_5d_list:
        if list(shape_x) == list(shape_y):
            format_list.append("FRACTAL_Z_3D")
        elif x_cdim == y_cdim and x_ndim == y_ndim:
            format_list.append("FRACTAL_Z_3D")

    # ND+ND NZ+NZ 5HD+5HD FZ+FZ
    if len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:]:
        format_list.append("FRACTAL_NZ")
        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list:
            if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                if format_x == format_y == "NCHW" and (x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16 == 1) \
                        and (x_ndim == y_ndim or x_ndim == 1 or y_ndim == 1):
                    format_list += format_5hd
                if format_x == format_y == "HWCN":
                    if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                        format_list += format_5hd
                    if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                        format_list += format_5hd
                    if x_wdim == y_wdim and x_hdim == y_hdim:
                        format_list += format_5hd
                    if (x_wdim == x_hdim == 1) or (y_hdim == y_wdim == 1):
                        format_list += format_5hd
                    if (x_hdim == y_wdim == 1) or (x_wdim == y_hdim == 1):
                        format_list += format_5hd
                if format_x == format_y == "NHWC":
                    if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                        format_list += format_5hd
                    if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                        format_list += format_5hd
                    if x_ndim == y_ndim and x_hdim == y_hdim:
                        format_list += format_5hd
                    if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                        format_list += format_5hd
                    if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                        format_list += format_5hd
            if x_cdim % 16 == 0 and y_cdim % 16 == 0 and y_ndim % 16 == 0 and x_ndim % 16 == 0:
                if (format_x == format_y == "NHWC" and list(shape_x) == list(shape_y)) or \
                        (format_x == format_y == "NCHW" and list(shape_x) == list(shape_y)):
                    format_list.append("FRACTAL_Z")
                if format_x == format_y == "HWCN" and x_wdim * x_hdim == y_wdim * y_hdim:
                    format_list.append("FRACTAL_Z")
            if list(shape_x) == list(shape_y):
                format_list += format_5hd
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        format_list_input0 = format_list
        format_list_input1 = format_list
        format_list_output = format_list
        unknownshape_format_list = ["ND"] * len(dtype_total)

    # NZ+ND,ND+ND,5HD+5HD,FZ+FZ,ND+NZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2 and ((_can_division_sixteen(shape_x) and
                                                       not _can_division_sixteen(shape_y)) or
                                                      (not _can_division_sixteen(shape_x) and
                                                       _can_division_sixteen(shape_y))):
        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list:
            if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                if x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16 == 1:
                    format_list += format_5hd
            if x_cdim % 16 == 0 and x_ndim % 16 == 0 and y_cdim % 16 == 0 and y_ndim % 16 == 0:
                if format_x == format_y == "NCHW" and x_hdim * x_wdim == y_hdim * y_wdim and x_cdim == y_cdim:
                    if x_ndim == y_ndim:
                        format_list.append("FRACTAL_Z")
                    if (x_ndim // 16 == 1 and y_ndim % 16 == 0) or (y_ndim // 16 == 1 and x_ndim % 16 == 0):
                        format_list.append("FRACTAL_Z")
                if format_x == format_y == "NHWC" and x_hdim * x_wdim == y_hdim * y_wdim and \
                        x_ndim == y_ndim and x_cdim == y_cdim:
                    format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nz * len_format_list
        format_list1 = format_list + format_nd * len_format_list
        if _can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y):
            format_list_input0 = format_list0
            format_list_input1 = format_list1
            format_list_output = format_list0
        else:
            format_list_input0 = format_list1
            format_list_input1 = format_list0
            format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)

    elif add_nd_nz or add_nz_nd:
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nz * len_format_list
        format_list1 = format_list + format_nd * len_format_list
        if len(shape_y) == 1 and len(shape_x) == 4 and format_x in format_4d_list:
            if shape_y[0] % 16 == 0 and x_cdim % 16 == 0:
                format_list0 = format_list + format_5hd * len_format_list
                format_list1 = format_list + format_5hd * len_format_list
        if add_nz_nd:
            format_list_input0 = format_list0
            format_list_input1 = format_list1
            format_list_output = format_list0
        else:
            format_list_input0 = format_list1
            format_list_input1 = format_list0
            format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)

    # 5HD+scalar,ND+ND,FZ+scalar
    elif len(shape_x) >= 2 and len(shape_y) == 1 and shape_y[0] == 1:
        if len(shape_x) == 4 and len(
                shape_y) == 1 and format_x in format_4d_list:
            if x_cdim % 16 == 0:
                format_list += format_5hd
            if x_cdim % 16 == 0 and x_ndim % 16 == 0:
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nd * len_format_list
        format_list1 = format_nd * len(format_list) + format_nd * len_format_list
        format_list_input0 = format_list0
        format_list_input1 = format_list1
        format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)

    # ND+ND,scalar+5HD,scalar+FZ
    elif len(shape_y) >= 2 and len(shape_x) == 1 and shape_x[0] == 1:
        if len(shape_x) == 1 and len(shape_y) == 4 and format_y in format_4d_list:
            if y_cdim % 16 == 0:
                format_list += format_5hd
            if y_cdim % 16 == 0 and y_ndim % 16 == 0:
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        format_list = format_list * len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * 1
        format_list0 = format_list + format_nd * len_format_list
        format_list1 = format_nd * len(format_list) + format_nd * len_format_list
        format_list_input0 = format_list1
        format_list_input1 = format_list0
        format_list_output = format_list0
        unknownshape_format_list = ["ND"] * len(dtype_total)
    # ND+ND,5HD+5HD
    else:
        if len(shape_x) == 1 and len(shape_y) == 1 and shape_x[0] % 16 == 0 and shape_y[0] % 16 == 0:
            format_list += format_5hd

        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list:
            if format_x == format_y == "NCHW" or format_x == format_y == "HWCN" or format_x == format_y == "NHWC":
                if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                    if (x_cdim // 16 == 1 or y_cdim // 16 == 1) or (x_cdim == y_cdim):
                        if x_ndim == y_ndim:
                            if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                                format_list += format_5hd
                            if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                                format_list += format_5hd
                            if x_hdim == y_hdim and x_wdim == y_wdim:
                                format_list += format_5hd
                            if (x_wdim == x_hdim == 1) or (y_wdim == y_hdim == 1):
                                format_list += format_5hd
                            if (x_hdim == 1 and y_wdim == 1) or (x_wdim == 1 and y_hdim == 1):
                                format_list += format_5hd
                        if x_hdim == y_hdim:
                            if x_ndim == y_ndim and (x_wdim == 1 or y_wdim == 1):
                                format_list += format_5hd
                            if x_wdim == y_wdim and (x_ndim == 1 or y_ndim == 1):
                                format_list += format_5hd
                            if x_ndim == y_ndim and x_wdim == y_wdim:
                                format_list += format_5hd
                            if (x_ndim == x_wdim == 1) or (y_ndim == y_wdim == 1):
                                format_list += format_5hd
                            if (x_ndim == 1 and y_wdim == 1) or (x_wdim == 1 and y_ndim == 1):
                                format_list += format_5hd
                        if x_wdim == y_wdim:
                            if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                                format_list += format_5hd
                            if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                                format_list += format_5hd
                            if x_ndim == y_ndim and x_hdim == y_hdim:
                                format_list += format_5hd
                            if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                                format_list += format_5hd
                            if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                                format_list += format_5hd
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        len_format_list = len(dtype_list)
        format_list = format_list * len_format_list
        format_list_input0 = format_list
        format_list_input1 = format_list
        format_list_output = format_list
        unknownshape_format_list = ["ND"] * len(dtype_total)

    if _can_broadcast(shape_x, shape_y) and len(shape_x) != len(shape_y):
        x_format = input_x.get("ori_format")
        y_format = input_y.get("ori_format")
        if x_format == "NHWC" or y_format == "NHWC":
            if len(shape_x) > 4 or len(shape_y) > 4:
                formats = ["ND"]
            else:
                formats = format_5hd
            for item in formats:
                dtype_total = dtype_total + dtype_list
                format_list_input0 = format_list_input0 + [item] * len(dtype_list)
                format_list_input1 = format_list_input1 + [item] * len(dtype_list)
                format_list_output = format_list_output + [item] * len(dtype_list)
            unknownshape_format_list = ["ND"] * len(dtype_total)

    if -1 in shape_x or -1 in shape_y:
        input0 = util_select_op_base.gen_param(classify="input0", name="x1", datatype=",".join(dtype_total),
                                               format=",".join(format_list_input0),
                                               unknownshape_format=",".join(unknownshape_format_list))
        input1 = util_select_op_base.gen_param(classify="input1", name="x2", datatype=",".join(dtype_total),
                                               format=",".join(format_list_input1),
                                               unknownshape_format=",".join(unknownshape_format_list))
        output0 = util_select_op_base.gen_param(classify="output0", name="y", datatype=",".join(dtype_total),
                                                format=",".join(format_list_output),
                                                unknownshape_format=",".join(unknownshape_format_list))
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="x1", datatype=",".join(dtype_total),
                                               format=",".join(format_list_input0))
        input1 = util_select_op_base.gen_param(classify="input1", name="x2", datatype=",".join(dtype_total),
                                               format=",".join(format_list_input1))
        output0 = util_select_op_base.gen_param(classify="output0", name="y", datatype=",".join(dtype_total),
                                                format=",".join(format_list_output))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _add_check_format(x, y):
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
    check_list = [["FRACTAL_NZ", "ND"], ["ND", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NHWC"], ["NHWC", "FRACTAL_NZ"],
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
        ori_shape_x, shape_y, shape_max = shape_util.broadcast_shapes(ori_shape_x, shape_y,
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


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("add")
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output of the data's add
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                                              param_name_input2="input_y")

    x_dtype = input_x.dtype.lower()
    if x_dtype in ("uint8", "int8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    input_x = tbe.broadcast(input_x, shape_max)
    input_y = tbe.broadcast(input_y, shape_max)
    res = tbe.vadd(input_x, input_y)

    if x_dtype in ("uint8", "int8"):
        res = util_common.uint8_int8_overflow_proc(res, x_dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def add(input_x, input_y, output_z, kernel_name="add"):
    """
    algorithm: add
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x : dict
        shape and dtype of first input, only support float16, float32, int32, int8, uint8
    input_y : dict
        shape and dtype of second input, only support float16, float32, int32, int8, uint8
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is add

    Returns
    -------
    None
    """
    # format_pattern = 1  Nz and vector
    # format_pattern = 2  vector and Nz
    # format_pattern = 0  Nz scalar  Nz Nz  ND ND
    format_pattern = _add_check_format(input_x, input_y)
    shape_x, shape_y = _infer_shape(format_pattern, input_x, input_y)
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")

    check_tuple = ("float16", "float32", "int32", "int8", "uint8")
    input_data_type = input_x.get("dtype").lower()
    para_check.check_dtype(input_data_type, check_tuple, param_name="input_x")

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                                              param_name_input2="input_y")
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_data_type)

    res = add_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"print_ir": False, "name": kernel_name,
              "tensor_list": (data_x, data_y, res)}
    tbe.cce_build_code(schedule, config)
