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
ascend_dequant
"""
import functools
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from te.utils import shape_util
from impl import ascend_quant_util


# pylint: disable=locally-disabled,too-many-arguments, unused-argument
# pylint: disable=invalid-name,too-many-locals,unnecessary-lambda
# pylint: disable=len-as-condition,too-many-branches,too-many-statements
def _check_params(x, deq_scale, sqrt_mode, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr
    """
    x_shape = x.get("shape")
    deq_shape = deq_scale.get("shape")

    x_format = x.get("format")
    deq_format = deq_scale.get("format")

    x_dtype = x.get("dtype").lower()
    deq_dtype = deq_scale.get("dtype").lower()
    x_format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_format(x_format, x_format_list, param_name="x")
    para_check.check_format(deq_format, ("NC1HWC0",), param_name="deq_scale")

    if x_format == "NC1HWC0":
        para_check.check_shape(x_shape, min_rank=5, max_rank=5, param_name="x")
    if x_format == "FRACTAL_NZ":
        para_check.check_shape(x_shape, min_rank=4, param_name="x")

    para_check.check_shape(deq_shape, min_rank=5, max_rank=5, param_name="deq_scale")

    if deq_shape[0] != 1 or deq_shape[2] != 1 or deq_shape[3] != 1:
        detail = "deq_scale shape must be 1 in n,h,w"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "deq_scale", detail)

    para_check.check_dtype(x_dtype, ("int32",), param_name="x")

    deq_dtype_check = "float16"
    if _is_support_v200_instruction():
        deq_dtype_check = "uint64"

    para_check.check_dtype(deq_dtype, (deq_dtype_check,), param_name="deq_scale")

    para_check.check_shape(x_shape, param_name="x")
    para_check.check_shape(deq_shape, param_name="deq_scale")

    if deq_dtype == "uint64" and sqrt_mode:
        rule = "when deq_scale dtype is uint64, sqrt_mode only support False"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "sqrt_mode", sqrt_mode)


def _is_support_v200_instruction():
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in (
            "Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS",
            "SD3403") or tbe_platform.api_check_support("tik.vgatherb"):
        return True
    return False


def _is_support_a100_instruction():
    if tbe_platform.api_check_support("tik.vgatherb"):
        return True
    return False


def _matmul_vdeq_cast_compute(x, deq_scale, x_shape, c1_index, tensor_flag, relu_flag, is_v200_flag):
    """
    generate lambda func
    """
    n_dim = len(x_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        new_indice = [0] * 5
        if tensor_flag:
            new_indice[4] = indice[c0_index]
            new_indice[1] = indice[c1_index]
        if is_v200_flag:
            if tensor_flag:
                func = tvm.vdeq_cast(x(*indice), deq_scale(*new_indice), dtype="float16", do_relu=relu_flag)
            else:
                func = tvm.deq_cast(x(*indice), deq_scale(*new_indice), dtype="float16")
        else:
            func = x(*indice).astype("float16") * deq_scale(*new_indice)
        return func

    return lambda_func


def _matmul_compute(x, x_shape, deq_scale, sqrt_mode, relu_flag, shape_matmul_origin, c1_index, tensor_flag):
    """
    dequant for matmul
    """
    if _is_support_v200_instruction():
        if tensor_flag:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, x_shape, c1_index, tensor_flag, relu_flag,
                                                            True),
                                  name="dequant", tag="dequant_vector")
        else:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, x_shape, c1_index, tensor_flag, relu_flag,
                                                            True),
                                  name="dequant", tag="dequant_scale")
    else:
        if tensor_flag:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, x_shape, c1_index, tensor_flag, relu_flag,
                                                            False),
                                  name="dequant", tag="dequant_vector")
        else:
            res_f16 = tvm.compute(x_shape,
                                  _matmul_vdeq_cast_compute(x, deq_scale, x_shape, c1_index, tensor_flag, relu_flag,
                                                            False),
                                  name="dequant", tag="dequant")
        if sqrt_mode:
            if tensor_flag:
                res_f16 = tvm.compute(x_shape,
                                      _matmul_vdeq_cast_compute(res_f16, deq_scale, x_shape, c1_index, tensor_flag,
                                                                relu_flag, False),
                                      name="dequant_sqrt", tag="dequant_vector_sqrt")
            else:
                res_f16 = tvm.compute(x_shape,
                                      _matmul_vdeq_cast_compute(res_f16, deq_scale, x_shape, c1_index, tensor_flag,
                                                                relu_flag, False),
                                      name="dequant_sqrt", tag="dequant_sqrt")

        if relu_flag:
            res_f16 = tvm.compute(x_shape, lambda *indices: tvm.relu(res_f16[indices]),
                                  name="dequant_relu", tag="dequant_relu")
    if not ascend_quant_util.is_nz_format(x):
        # convert fractal_z to ND
        res_out = tvm.compute(shape_matmul_origin, lambda i, j: res_f16[j // 16, i // 16, i % 16, j % 16],
                              name="dequant_ND", tag="dequant_ND", attrs={"format": "NC1HWC0"})
    else:
        # nz format
        res_out = tvm.compute(x_shape, lambda *i: res_f16[i], name="dequant_NZ", tag="dequant_NZ",
                              attrs={"format": "FRACTAL_NZ"})
    if "ori_shape" in x.op.attrs:
        res_out.op.attrs["ori_shape"] = x.op.attrs["ori_shape"]
    return res_out


def _vector_dequant_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode, conv_flag):
    """
    dequant for vector in v100
    """
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"])
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]

        if relu_flag:
            res_f16 = tvm.compute(
                align_shape,
                lambda batch, cout1, howo, cout0:
                    tvm.relu(x.op.input_tensors[0](
                        0 if group == 1 else cout1 // cout1_opt, batch,
                        cout1 if group == 1 else cout1 % cout1_opt, howo, cout0).astype("float16") *
                        deq_scale(0, cout1, 0, 0, cout0)),
                name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})
        else:
            res_f16 = tvm.compute(
                align_shape,
                lambda batch, cout1, howo, cout0:
                    x.op.input_tensors[0](
                        0 if group == 1 else cout1 // cout1_opt, batch,
                        cout1 if group == 1 else cout1 % cout1_opt, howo, cout0).astype("float16") *
                        deq_scale(0, cout1, 0, 0, cout0),
                name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value//2 # remove padded column and pad
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            x_shape = res_shape_nchw_after_removepad
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0:
                                      res_f16(batch, cout1, howo*2, cout0),
                                  name='dequant_remove_padded_column',
                                  tag='dequant_remove_padded_column')
        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")
    else:
        if relu_flag:
            res_f16 = tvm.compute(align_shape,
                                  lambda i, j, k, l:
                                      tvm.relu(x(i, j, k, l).astype("float16") * deq_scale(0, j, 0, 0, l)),
                                  name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})
        else:
            res_f16 = tvm.compute(align_shape,
                                  lambda i, j, k, l: x(i, j, k, l).astype("float16") * deq_scale(0, j, 0, 0, l),
                                  name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")

    if sqrt_mode:
        res = tvm.compute(x_shape,
                          lambda i, j, k, l: (res(i, j, k, l) * deq_scale(0, j, 0, 0, l)),
                          name="dequant2", tag="dequant2_vector")
    return res


def _scalar_dequant_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode, conv_flag):
    """
    dequant for scale in v100
    """
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"])
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]
        res_f16 = tvm.compute(
            align_shape,
            lambda batch, cout1, howo, cout0:
                x.op.input_tensors[0](
                    0 if group == 1 else cout1 // cout1_opt, batch,
                    cout1 if group == 1 else cout1 % cout1_opt, howo, cout0).astype("float16") *
                    deq_scale(0, 0, 0, 0, 0),
            name="dequant1", tag="dequant1_scale")

        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value//2 # remove padded column
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0:
                                      res_f16(batch, cout1, howo*2, cout0),
                                  name='dequant_remove_padded_column',
                                  tag='dequant_remove_padded_column')

        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")

        x_shape = res_shape_nchw_after_removepad

    else:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, k, l: (x(i, j, k, l).astype("float16") * deq_scale(0, 0, 0, 0, 0)),
                              name="dequant1", tag="dequant1_scale")
        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")


    if relu_flag:
        res = tvm.compute(x_shape, lambda *indices: tvm.relu(res(*indices)), name="dequant_relu", tag="dequant_relu")
    if sqrt_mode:
        res = tvm.compute(x_shape,
                          lambda i, j, k, l: (res(i, j, k, l) * deq_scale(0, 0, 0, 0, 0)),
                          name="dequant2", tag="dequant2_scale")

    return res


def _vector_dequant_v200(x, x_shape, align_shape, deq_scale, relu_flag, conv_flag):
    """
    dequant for vector in v200
    """
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"]) if not _is_support_a100_instruction() else False
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]

        res_f16 = tvm.compute(
            align_shape,
            lambda batch, cout1, howo, cout0:
                tvm.vdeq_cast(
                    x.op.input_tensors[0](
                        0 if group == 1 else cout1 // cout1_opt, batch,
                        cout1 if group == 1 else cout1 % cout1_opt, howo, cout0),
                        deq_scale(0, cout1, 0, 0, cout0), dtype="float16", do_relu=relu_flag),
            name="dequant", tag="dequant_vector")

        if not _is_support_a100_instruction() and x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value//2 # remove padded column and pad
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0:
                                      res_f16(batch, cout1, howo*2, cout0),
                                  name = 'dequant_remove_padded_column',
                                  tag = 'dequant_remove_padded_column')

        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")
    else:
        res_f16 = tvm.compute(
            align_shape,
            lambda i, j, k, l:
                tvm.vdeq_cast(x(i, j, k, l), deq_scale(0, j, 0, 0, l), dtype="float16", do_relu=relu_flag),
            name="dequant", tag="dequant_vector")
        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")

    return res


def _scalar_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode):
    """
    dequant for vector in v100
    """
    if relu_flag:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, a, k, l:
                              tvm.relu(x(i, j // 2, j % 2, k, l).astype("float16") * deq_scale(0, 0, 0, 0, 0)),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})

    else:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, a, k, l:
                              x(i, j // 2, j % 2, k, l).astype("float16") * deq_scale(0, 0, 0, 0, 0),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

    align_shape[3] = x_shape[3].value

    if not sqrt_mode:
        res = tvm.compute(align_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                          tag="dequant_remove_pad", attrs={"sqrt_flag": 0})
    else:
        res_sqrt = tvm.compute(align_shape,
                               lambda i, j, a, k, l: (res_f16(i, j, a, k, l) * deq_scale(0, 0, 0, 0, 0)),
                               name="dequant2", tag="dequant2_vector")

        res = tvm.compute(align_shape, lambda *indice: res_sqrt(*indice), name="dequant2_remove_pad",
                          tag="dequant2_remove_pad", attrs={"sqrt_flag": 1})
    return res


def _vector_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode):
    """
    dequant for vector in v100
    """
    if relu_flag:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, a, k, l:
                              tvm.relu(x(i, j // 2, j % 2, k, l).astype("float16") * deq_scale(0, j, 0, 0, l)),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 1})
    else:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, a, k, l:
                              x(i, j // 2, j % 2, k, l).astype("float16") * deq_scale(0, j, a, 0, l),
                              name="dequant1", tag="dequant1_vector", attrs={"relu_flag": 0})

    align_shape[3] = x_shape[3].value

    if not sqrt_mode:
        res = tvm.compute(align_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                          tag="dequant_remove_pad", attrs={"sqrt_flag": 0})
    else:
        res_sqrt = tvm.compute(align_shape,
                               lambda i, j, a, k, l: (res_f16(i, j, a, k, l) * deq_scale(0, j, a, 0, l)),
                               name="dequant2", tag="dequant2_vector")

        res = tvm.compute(align_shape,
                          lambda *indice: res_sqrt(*indice), name="dequant2_remove_pad",
                          tag="dequant2_remove_pad", attrs={"sqrt_flag": 1})
    return res


def _vector_depthwise_fused_v200(x, x_shape, align_shape, deq_scale, relu_flag):
    """
    depthwise dequant for vector in v200
    """
    res_f16 = tvm.compute(align_shape,
                          lambda i, j, a, k, l:
                          tvm.vdeq_cast(x(i, j // 2, j % 2, k, l), deq_scale(0, j, 0, 0, l), dtype="float16",
                                        do_relu=relu_flag),
                          name="dequant1", tag="dequant1_vector", attrs={"relu_flag": relu_flag})

    align_shape[3] = x_shape[3].value

    res = tvm.compute(align_shape, lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                      tag="dequant_remove_pad", attrs={"sqrt_flag": 0})

    return res


def _scalar_depthwise_fused_v200(x, x_shape, align_shape, deq_scale, relu_flag):
    """
    depthwise dequant for vector in v200
    """
    res_f16 = tvm.compute(align_shape,
                          lambda i, j, a, k, l:
                          tvm.deq_cast(x(i, j // 2, j % 2, k, l), deq_scale(0, 0, 0, 0, 0), dtype="float16"),
                          name="dequant1", tag="dequant1_scale")

    align_shape[3] = x_shape[3].value

    res = tvm.compute(align_shape, lambda *indice: res_f16(*indice), name="dequant_remove_pad",
                      tag="dequant_remove_pad", attrs={"sqrt_flag": 0})

    return res


def _scalar_dequant_v200(x, x_shape, align_shape, deq_scale, conv_flag):
    """
    dequant for scale in v200
    """
    if conv_flag:
        invalid_data_rm_flag = int(x.op.attrs["invalid_data_rm_flag"])
        group = x.op.input_tensors[0].shape[0].value
        cout1_opt = x.op.input_tensors[0].shape[2].value
        res_shape_nchw_after_removepad = x.op.attrs["conv_shape"]

        res_f16 = tvm.compute(
            align_shape,
            lambda batch, cout1, howo, cout0:
                tvm.deq_cast(x.op.input_tensors[0](0 if group == 1 else cout1 // cout1_opt, batch,
                                                   cout1 if group == 1 else cout1 % cout1_opt, howo, cout0),
                             deq_scale(0, 0, 0, 0, 0), dtype="float16"),
            name="dequant", tag="dequant_scale")

        if x.op.attrs["remove_padded_column_in_next_op"].value == 1:
            remove_padded_column_shape = align_shape
            remove_padded_column_shape[-2] = res_shape_nchw_after_removepad[-2].value//2 # remove padded column
            res_shape_nchw_after_removepad = x.op.attrs["true_conv_shape"]
            res_f16 = tvm.compute(remove_padded_column_shape,
                                  lambda batch, cout1, howo, cout0:
                                      res_f16(batch, cout1, howo*2, cout0),
                                  name = 'dequant_remove_padded_column',
                                  tag = 'dequant_remove_padded_column')

        if invalid_data_rm_flag:
            res = tvm.compute(res_f16.shape,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='invalid_dequant_rmpad',
                              tag="invalid_dequant_rmpad")
        else:
            res = tvm.compute(res_shape_nchw_after_removepad,
                              lambda batch, cout1, howo, cout0:
                              res_f16(batch, cout1, howo, cout0),
                              name='dequant_remove_pad',
                              tag="dequant_remove_pad")
    else:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, k, l: tvm.deq_cast(x(i, j, k, l), deq_scale(0, 0, 0, 0, 0), dtype="float16"),
                              name="dequant", tag="dequant_scale")
        res = tvm.compute(x_shape,
                          lambda *indice: res_f16(*indice), name="dequant_remove_pad", tag="dequant_remove_pad")
    return res


@tbe_platform.fusion_manager.fusion_manager.register("ascend_dequant")
def ascend_dequant_compute(x, deq_scale, y, sqrt_mode=False, relu_flag=False, kernel_name="ascend_dequant"):
    """
    int32 -> fp16

    Parameters:
    ----------
    x: the placeholder of input
    deq_scale: the placeholder of deq_scale
    y: the dict of output
    sqrt_mode: the sqrt mode, when true the result to do sqrt
    relu_flag: the relu mode, when true the result to do relu
    kernel_name: cce kernel name, default value is "ascend_dequant"

    Returns:
    -------
    res : the result of ascend_dequant
    """
    conv_flag = 0
    if len(x.op.input_tensors) and ('mad1' in x.op.input_tensors[0].name or \
            'convolution_c_col_bias' in x.op.input_tensors[0].name):
        conv_flag = 1

    def shape_to_list(shape):
        """
        trans shape to list shape
        """
        tmp = []
        for i in shape:
            tmp.append(i.value)
        return tmp

    x_shape = x.shape
    deq_shape = deq_scale.shape
    x_shape_list = shape_to_list(x_shape)
    deq_shape_list = shape_to_list(deq_shape)
    ori_shape_deq = deq_scale.op.attrs["ori_shape"]
    ori_shape_deq_list = shape_util.shape_to_list(ori_shape_deq)
    deq_dim = functools.reduce(lambda x, y: x * y, ori_shape_deq_list[:])
    tensor_flag = False
    if deq_dim > 1:
        tensor_flag = True

    if conv_flag:
        x_input_shape = shape_util.shape_to_list(x.op.input_tensors[0].shape)
        align_shape = [x_input_shape[1],
                       x.shape[1],
                       x_input_shape[3],
                       x_input_shape[4]]
    else:
        align_shape = x_shape_list.copy()

    if x.op.tag != "depthwise_conv2d":
        align_shape[2] = (align_shape[2] + 15) // 16 * 16

    if x.op.tag in ("matmul", "matmul_gemv", "matmul_gevm", "gemm"):
        shape_matmul_origin = x.op.attrs["shape"]
        c1_index = len(x_shape) - 4
        res = _matmul_compute(x, x_shape, deq_scale, sqrt_mode, relu_flag, shape_matmul_origin, c1_index, tensor_flag)
        return res
    if x.op.tag == "depthwise_conv2d":
        align_shape[4] = 16
        align_shape[3] = (x_shape_list[3] + 15) // 16 * 16
        align_shape[2] = 1
        if deq_shape_list[1] == 1:
            tensor_dict = {}
            tensor_dict["mad_ubuf"] = x.op.input_tensors[0]
            if x.op.attrs['bias_flag'].value == 1:
                tensor_dict["flag_is_dequant_bias"] = True
                tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
                tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
                tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
                tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
                tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
            else:
                tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
            tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
            tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
            tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
            tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]

            tensor_dict["temp"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
            if tensor_dict["temp"].op.input_tensors:
                tensor_dict["fmap"] = tensor_dict["temp"].op.input_tensors[0]
            else:
                tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
            x_ori_shape = tensor_dict["fmap"].op.attrs["ori_shape"]
            x_ori_format = tensor_dict["fmap"].op.attrs["ori_format"]
            x_ori_shape_list = shape_util.shape_to_list(x_ori_shape)
            if x_ori_format == "NCHW":
                align_shape[1] = (x_ori_shape_list[1] + 15) // 16
            else:
                align_shape[1] = (x_ori_shape_list[3] + 15) // 16
        else:
            align_shape[1] = (deq_shape_list[1] * deq_shape_list[4]) // 16
        align_shape[0] = x_shape_list[0]

        if tensor_flag:
            if _is_support_v200_instruction():
                res = _vector_depthwise_fused_v200(x, x_shape, align_shape, deq_scale, relu_flag)
            else:
                res = _vector_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode)
        else:
            if _is_support_v200_instruction():
                res = _scalar_depthwise_fused_v200(x, x_shape, align_shape, deq_scale, relu_flag)
            else:
                res = _scalar_depthwise_fused_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode)

        return res

    if tensor_flag:
        if _is_support_v200_instruction() or _is_support_a100_instruction():
            res = _vector_dequant_v200(x, x_shape, align_shape, deq_scale, relu_flag, conv_flag)
        else:
            res = _vector_dequant_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode, conv_flag)
    else:
        if _is_support_v200_instruction() or _is_support_a100_instruction():
            res = _scalar_dequant_v200(x, x_shape, align_shape, deq_scale, conv_flag)
        else:
            res = _scalar_dequant_v100(x, x_shape, align_shape, deq_scale, relu_flag, sqrt_mode, conv_flag)

    return res


def _dequant_v200_v2(x_l0c, deq_ub, align_shape, x_shape, relu_flag, tensor_flag):
    """
    dequant for vector in v200
    """
    if tensor_flag:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, k, l:
                              tvm.vdeq_cast(x_l0c(i, j, k, l), deq_ub(0, j, 0, l), dtype="float16", do_relu=relu_flag),
                              name="dequant_to_fp16", tag="dequant_vector")
    else:
        res_f16 = tvm.compute(align_shape,
                              lambda i, j, k, l: tvm.deq_cast(x_l0c(i, j, k, l), deq_ub(0, 0, 0, 0), dtype="float16"),
                              name="dequant_to_fp16", tag="dequant_scale")
    is_scalar = 1
    if tensor_flag:
        is_scalar = 0
    res = tvm.compute(x_shape,
                      lambda *indice: res_f16(*indice), name="res", tag="dequant_res",
                      attrs={"is_scalar": is_scalar})

    return res


def _vector_dequant_v100_v2(x_l0c, deq_ub, align_shape, x_shape, relu_flag, sqrt_mode):
    """
    dequant for vector in v100
    """
    if relu_flag:
        res = tvm.compute(align_shape,
                          lambda i, j, k, l: tvm.relu(x_l0c(i, j, k, l).astype("float16") * deq_ub(0, j, 0, l)),
                          name="dequant_to_fp16")
    else:
        res = tvm.compute(align_shape,
                          lambda i, j, k, l: x_l0c(i, j, k, l).astype("float16") * deq_ub(0, j, 0, l),
                          name="dequant_to_fp16")

    if sqrt_mode:
        res = tvm.compute(x_shape,
                          lambda i, j, k, l: (res(i, j, k, l) * deq_ub(0, j, 0, l)), name="dequant_sqrt")

    res = tvm.compute(x_shape,
                      lambda *indice: res(*indice), name="res", tag="dequant_res",
                      attrs={"sqrt_mode": sqrt_mode,
                             "relu_mode": relu_flag,
                             "is_scalar": 0})

    return res


def _scalar_dequant_v100_v2(x_l0c, deq_ub, align_shape, x_shape, relu_flag, sqrt_mode):
    """
    dequant for scale in v100
    """
    res = tvm.compute(align_shape,
                      lambda i, j, k, l: (x_l0c(i, j, k, l).astype("float16") * deq_ub(0, 0, 0, 0)),
                      name="dequant_to_fp16")

    if sqrt_mode:
        res = tvm.compute(x_shape,
                          lambda i, j, k, l: (res(i, j, k, l) * deq_ub(0, 0, 0, 0)), name="dequant_sqrt")

    if relu_flag:
        res = tvm.compute(x_shape,
                          lambda *indices: tvm.relu(res(*indices)), name="dequant_relu")

    res = tvm.compute(x_shape, lambda *indice: res(*indice), name="res", tag="dequant_res",
                      attrs={"sqrt_mode": sqrt_mode,
                             "relu_mode": relu_flag,
                             "is_scalar": 1})
    return res


def ascend_dequant_compute_v2(x, deq_scale, y, sqrt_mode=False, relu_flag=False, kernel_name="ascend_dequant"):
    """
    int32 -> fp16

    Parameters:
    ----------
    x: the placeholder of input
    deq_scale: the placeholder of dequant num
    offset: the placeholder of offset num
    y: the dict of output
    sqrt_mode: the sqrt mode when true the result to do sqrt
    relu_flag: the relu mode when true the result to do relu
    kernel_name: cce kernel name, default value is "ascend_dequant"

    Returns:

    res : the result of ascend_dequant
    -------
    None
    """
    ori_shape_deq = deq_scale.op.attrs["ori_shape"]
    ori_shape_deq_list = shape_util.shape_to_list(ori_shape_deq)
    deq_dim = functools.reduce(lambda x, y: x * y, ori_shape_deq_list[:])
    tensor_flag = False
    if deq_dim > 1:
        tensor_flag = True

    align_shape = shape_util.shape_to_list(x.shape)
    align_shape[-2] = (align_shape[-2] + 15) // 16 * 16

    x_ub = tvm.compute(x.shape, lambda *i: x(*i), name="x_ub", tag="dequant_x_ub")
    deq_ub = tvm.compute(deq_scale.shape, lambda *i: deq_scale(*i), name='deq_ub', tag="dequant_deq_ub")
    x_l0c = tvm.compute(align_shape, lambda *i: x_ub(*i), name="x_l0c", tag="dequant_x_l0c")

    if tensor_flag:
        if _is_support_v200_instruction():
            res = _dequant_v200_v2(x_l0c, deq_ub, align_shape, x.shape, relu_flag, tensor_flag)
        else:
            res = _vector_dequant_v100_v2(x_l0c, deq_ub, align_shape, x.shape, relu_flag, sqrt_mode)
    else:
        if _is_support_v200_instruction():
            res = _dequant_v200_v2(x_l0c, deq_ub, align_shape, x.shape, relu_flag, tensor_flag)
        else:
            res = _scalar_dequant_v100_v2(x_l0c, deq_ub, align_shape, x.shape, relu_flag, sqrt_mode)
    return res


def get_op_support_info(x, deq_scale, y, sqrt_mode=False, relu_mode=False, kernel_name="ascend_dequant"):
    """
    get split info
    """
    return ascend_quant_util.get_quant_support_info(x)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ascend_dequant(x, deq_scale, y, sqrt_mode=False, relu_mode=False, kernel_name="ascend_dequant"):
    """
    int32 -> fp16

    Parameters:
    ----------
    x: the dict of input
    deq_scale: the dict of dequant num
    offset: the dict of offset num
    y: the dict of output
    sqrt_mode: the sqrt mode when true the result to do sqrt
    relu_flag: the relu mode when true the result to do relu
    kernel_name: cce kernel name, default value is "ascend_dequant"

    Returns:
    -------
    None
    """
    _check_params(x, deq_scale, sqrt_mode, kernel_name)

    shape_x = x.get("shape")
    shape_deq = deq_scale.get("shape")

    dtype_x = x.get("dtype")
    dtype_deq = deq_scale.get("dtype")
    x_format = x.get("format")
    ori_shape_deq = deq_scale.get("ori_shape")
    attr = {"ori_shape": ori_shape_deq}

    if x_format == "NC1HWC0":
        # n, C1, H*W, C0
        shape_x = [shape_x[0], shape_x[1], shape_x[2] * shape_x[3], shape_x[4]]
        shape_deq = [shape_deq[0], shape_deq[1], shape_deq[2] * shape_deq[3], shape_deq[4]]
    else:
        # C1,N1,N0,C0 change to 1,C1,N1*N0,C0 equivalence N,C1,H*W,C0
        x_batch = 1
        if len(shape_x) > 4:
            x_batch = functools.reduce(lambda x, y: x * y, shape_x[:-4])
        shape_x = [x_batch, shape_x[-4], shape_x[-3] * shape_x[-2], shape_x[-1]]
        shape_deq = [shape_deq[0], shape_deq[1], shape_deq[2] * shape_deq[3], shape_deq[4]]

    input_x = tvm.placeholder(shape_x, dtype_x, "x")
    input_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr)

    with tvm.target.cce():
        res = ascend_dequant_compute_v2(input_x, input_deq, y, sqrt_mode, relu_mode, kernel_name)
        sch = tbe.auto_schedule(res)
        config = {"name": kernel_name,
                  "tensor_list": [input_x, input_deq, res]}
        tbe.cce_build_code(sch, config)
