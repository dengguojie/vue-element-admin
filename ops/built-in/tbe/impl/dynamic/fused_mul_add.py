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
fused_mul_add
"""
from impl import constant_util
from impl.util import util_select_op_base
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.platform_adapter import register_operator
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import update_shape_for_other_format


def _division_sixteen(shape):
    """
    check be div by sixteen
    """
    if len(shape) < 2:
        if shape[-1] == 0:
            error_detail = 'value of shape is illegal, shape[-1] == 0'
            error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        error_detail = 'value of shape is illegal, shape[-1] == %s, shape[-2] == %s' % (shape[-1], shape[-2])
        error_manager_vector.raise_err_specific_reson("fused_mul_add", error_detail)

    return shape[-1] % constant_util.SIZE_SIXTEEN == 0 and shape[-2] % constant_util.SIZE_SIXTEEN == 0


def split_bind(shape0, shape1):
    """
    check can be split together
    """
    if len(shape0) == 0 or len(shape1) == 0:
        return False
    if shape0[0] == 1 or shape1[0] == 1:
        return False
    if len(shape0) != len(shape1):
        return False
    if shape0[0] == shape1[0]:
        return True
    return False


def get_split_matrix(input0_shape, input1_shape, input2_shape):
    """
    get axis split matrix
    """
    axis_split_matrix = None
    if split_bind(input0_shape, input1_shape):
        input_slice_list = [[0, [0], [-1], [-1]], [1, [0], [-1], [-1]]]
        if split_bind(input1_shape, input2_shape):
            input_slice_list.append([2, [0], [-1], [-1]])
        split_0 = [SplitInput(*input_slice_list), SplitOutput([0, [0]])]
        axis_split_matrix = [split_0]

    elif split_bind(input0_shape, input2_shape):
        input_slice_list = [[0, [0], [-1], [-1]], [2, [0], [-1], [-1]]]
        split_0 = [SplitInput(*input_slice_list), SplitOutput([0, [0]])]
        axis_split_matrix = [split_0]
    
    elif split_bind(input1_shape, input2_shape):
        input_slice_list = [[0, [0], [-1], [-1]], [1, [0], [-1], [-1]]]
        split_0 = [SplitInput(*input_slice_list), SplitOutput([0, [0]])]
        axis_split_matrix = [split_0]
    
    return axis_split_matrix


def get_op_support_info(input0, input1, input2, output,
                        kernel_name="fused_mul_add"):
    """
    get_op_support_info
    """
    input0_shape = list(input0.get('shape'))
    input1_shape = list(input1.get('shape'))
    input2_shape = list(input2.get('shape'))

    input_list = [input0, input1, input2]
    input_shape_list = [input0_shape, input1_shape, input2_shape]
    input_len_list = [len(input0_shape), len(input1_shape), len(input2_shape)]
    maxlen_idx = input_len_list.index(max(input_len_list))

    axis_split_matrix = None
    axis_reduce_list = None

    if input_len_list[maxlen_idx] != 0:
        for _idx, _input in enumerate(input_list):
            if _idx != maxlen_idx and input_len_list[_idx] != 0:
                input_shape_list[_idx] = \
                update_shape_for_other_format(_input['shape'], 
                                              _input['format'].upper(),
                                              _input['ori_shape'],
                                              input_list[maxlen_idx]['format'].upper())

        axis_split_matrix = get_split_matrix(*input_shape_list)
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(input0, input1, input2, output,
                     kernel_name="fused_mul_add"):
    """
    _division_sixteen : judge whether the last two dimensions are divided by 16
    scalar2tensor_one : convert scalar to tensor
    """
    shape_0 = input0.get("ori_shape")
    shape_1 = input1.get("ori_shape")
    shape_2 = input2.get("ori_shape")

    shape_0 = shape_util.scalar2tensor_one(shape_0)
    shape_1 = shape_util.scalar2tensor_one(shape_1)
    shape_2 = shape_util.scalar2tensor_one(shape_2)

    if _division_sixteen(shape_0) and not _division_sixteen(shape_1) \
            and not _division_sixteen(shape_2):
        # Nz+ND+ND
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input2 = util_select_op_base.gen_param(classify="input2", name="x3",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                                                format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")

    elif _division_sixteen(shape_0) and not _division_sixteen(shape_1) \
            and _division_sixteen(shape_2):
        # Nz+ND+Nz
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input2 = util_select_op_base.gen_param(classify="input2", name="x3",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                                                format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")

    elif not _division_sixteen(shape_0) and _division_sixteen(shape_1) \
            and not _division_sixteen(shape_2):
        # ND+NZ+ND
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        input2 = util_select_op_base.gen_param(classify="input2", name="x3",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                                                format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")

    elif not _division_sixteen(shape_0) and _division_sixteen(shape_1) \
            and _division_sixteen(shape_2):
        # ND+NZ+NZ
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        input2 = util_select_op_base.gen_param(classify="input2", name="x3",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                                                format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")

    elif not _division_sixteen(shape_0) and not _division_sixteen(shape_1) \
            and _division_sixteen(shape_2):
        # ND+ND+NZ
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input2 = util_select_op_base.gen_param(classify="input2", name="x3",
                                               datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                                                format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
    else:
        # ND+ND
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype="float16,float16,float16,float16,\
                                     float,float,float,float,\
                                     int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype="float16,float16,float16,float16,\
                                     float,float,float,float,\
                                     int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")
        input2 = util_select_op_base.gen_param(classify="input2", name="x3",
                                               datatype="float16,float16,float16,float16,\
                                     float,float,float,float,\
                                     int32,int32,int32,int32",
                                               format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float16,float16,float16,\
                                      float,float,float,float,\
                                      int32,int32,int32,int32",
                                                format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


def _check_format(format_input0, format_input1, format_input2):
    """
    check the format_list
    """
    list_format = [format_input0, format_input1, format_input2]

    nd_format = {"ND", "NHWC", "NCHW", "HWCN"}
    standard_format = []

    for item in list_format:
        if item in nd_format:
            standard_format.append("ND")
        else:
            standard_format.append(item)

    list_pattern = [
        ["FRACTAL_NZ", "ND", "ND"],
        ["ND", "FRACTAL_NZ", "ND"],
        ["ND", "ND", "FRACTAL_NZ"],
        ["FRACTAL_NZ", "ND", "FRACTAL_NZ"]
    ]
    if standard_format in list_pattern:
        format_pattern = list_pattern.index(standard_format) + 1
    else:
        format_pattern = 0

    return format_pattern


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("FusedMulAdd", op_mode="dynamic", support_fusion=True)
def fused_mul_add_compute(data_input0, data_input1, data_input2,
                          output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fuesd_mul_add"

    Returns
    -------
    output tensor
    """
    # mul
    data_input0, data_input1 = _shape_broadcast(data_input0, data_input1)
    mul_result = tbe.vmul(data_input0, data_input1)

    # add
    mul_result, data_input2 = _shape_broadcast(mul_result, data_input2)
    res = tbe.vadd(mul_result, data_input2)

    return res


@register_operator("FusedMulAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fused_mul_add(input0, input1, input2,
                  output, kernel_name="fused_mul_add"):
    """
    function: fused for mul+add

    Parameters
    ----------
    input0: dict
         the dict of input of mul, support float16,float32,int32
    input1: dict
         the dict of input of mul, support float16,float32,int32
    input2: dict
         the dict of input of add, support float16,float32,int32
    output: dict
         the dict of output of add, support float16,float32,int32
    kernel_name: str
        cce kernel name, default value is fused_mul_add

    Returns
    -------
    None
    """
    # check dtype
    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()
    check_dtype_list = ["float32", "float16", "int32"]
    para_check.check_dtype(dtype_input0, check_dtype_list, param_name="input0")
    para_check.check_dtype(dtype_input1, check_dtype_list, param_name="input1")
    para_check.check_dtype(dtype_input2, check_dtype_list, param_name="input2")

    # check format
    format_input0 = input0.get("format").upper()
    format_input1 = input1.get("format").upper()
    format_input2 = input2.get("format").upper()
    _check_format(format_input0, format_input1, format_input2)

    # classify
    ins = classify([input0, input1, input2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input0, _input1, _input2) in ins:
        with tbe.compute():
            shape_input0, shape_input1, shape_input2 = shape_util.variable_shape([_input0, _input1, _input2])

            data_input0 = tvm.placeholder(shape_input0, name="data_input0", dtype=dtype_input0)
            data_input1 = tvm.placeholder(shape_input1, name="data_input1", dtype=dtype_input1)
            data_input2 = tvm.placeholder(shape_input2, name="data_input2", dtype=dtype_input2)

            res = fused_mul_add_compute(data_input0, data_input1, data_input2, output, kernel_name)

            tensor_list = [data_input0, data_input1, data_input2, res]
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
