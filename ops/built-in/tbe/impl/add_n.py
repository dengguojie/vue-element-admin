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
add_n
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import shape_util
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.util import util_compute
from tbe.dsl import broadcast


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("add_n")
def add_n_compute_for_fusion(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    # if fused with batch_matmul, split batch dim in batchmatmul
    batchmatmul_flag = False
    for i, data_n in enumerate(datas):
        batchmatmul_flag = util_compute.check_batchmatmul_fuse(data_n)
        if batchmatmul_flag:
            if "para_name" in data_n.op.attrs:
                para_name = data_n.op.attrs["para_name"].value
                para_name += "_addn"
            else:
                para_name = "addn"
            batch_shape = shape_util.shape_to_list(data_n.op.attrs["batch_shape"])
            batchmatmul_node = data_n
            batchmatmul_idx = i
            shape_max = batch_shape + shape_util.shape_to_list(data_n.shape)[-4:]
            break
    nz_addn = []
    for i, data_n in enumerate(datas):
        if batchmatmul_flag and i != batchmatmul_idx:
            data_n = broadcast(data_n, shape_max)
            data_n = util_compute.batchmatmul_elem_reshape(batchmatmul_node, data_n, batch_shape, para_name + str(i))
            nz_addn.append(data_n)

    if not batchmatmul_flag:
        res = datas[0]
        for i, data_n in enumerate(datas):
            if i == 0:
                continue
            res = tbe.vadd(res, data_n)
        return res

    for i, nz_add in enumerate(nz_addn):
        batchmatmul_node = tbe.vadd(batchmatmul_node, nz_add)
    return batchmatmul_node


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
def add_n_compute(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    data_type = datas[0].dtype
    has_covert_float32 = (data_type == "float16" and
                          tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"))

    first_data = datas[0] if not has_covert_float32 else tbe.cast_to(datas[0], "float32")

    res = first_data
    for i, data_n in enumerate(datas):
        if i == 0:
            continue
        temp_data = data_n if not has_covert_float32 else tbe.cast_to(data_n, "float32")
        res = tbe.vadd(res, temp_data)

    if has_covert_float32:
        res = tbe.cast_to(res, "float16")
    return res


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def add_n(inputs, output, tensor_num, kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects, each with same shape and type.
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """
    input_num = len(inputs)
    if input_num < 2:
        expected_value = "greater than or equal to 2"
        real_value = "less than 2"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "the length of inputs",
                                                           expected_value, real_value)

    if input_num != tensor_num:
        expected_value = tensor_num
        real_value = input_num
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "the length of inputs",
                                                           expected_value, real_value)

    shape_0 = inputs[0].get("shape")
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape_0)

    check_list = ("float16", "float32", "int32")
    data = []
    for i, input_dict in enumerate(inputs):
        shape_input = input_dict.get("shape")
        if list(shape_0) != list(shape_input):
            expected_value = list(shape_input)
            real_value = list(shape_0)
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "shape of input",
                                                               expected_value, real_value)
        para_check.check_shape(shape_input, param_name="inputs")
        dtype_input = input_dict.get("dtype").lower()
        para_check.check_dtype(dtype_input, check_list, param_name="inputs")
        data.append(tvm.placeholder(fuseshape, name="data_%d" % i, dtype=dtype_input))

    res = add_n_compute(data, output, tensor_num, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    data.append(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": data}

    tbe.cce_build_code(schedule, config)
