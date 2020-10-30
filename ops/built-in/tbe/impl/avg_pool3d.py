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
avg_pool3d
"""
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


def _check_window_rule(ksize, strides, pads):
    """
    avg_pool3d_check_window_rule

    Parameters
    ----------
    ksize: kernel size
    strides: stride

    Returns
    -------
    None
    """
    if len(ksize) != 5:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d', 'ksize', 5, 5, len(ksize))

    if len(strides) != 5:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d', 'strides', 5, 5, len(strides))

    if len(pads) != 6:
        error_manager_vector.raise_err_input_param_range_invalid('avg_pool3d', 'pads', 6, 6, len(pads))


def _check_global_rule(input_shape, ksize, pads):
    ksize_h = ksize[2]
    ksize_w = ksize[3]
    fmap_h = input_shape[3]
    fmap_w = input_shape[4]
    if ksize_h != fmap_h or ksize_w != fmap_w:
        error_manager_vector.raise_err_check_params_rules('avg_pool3d', 'only support slide on D dimension now',
                                                          'ksize', ksize)

    if pads[0] != 0 or pads[1] != 0 or pads[2] != 0 or pads[3] != 0 or pads[4] != 0 or pads[5] != 0:
        error_manager_vector.raise_err_pad_mode_invalid('avg_pool3d', 'valid', 'other')


# pylint: disable=too-many-arguments,unused-argument,invalid-name
def _avg_pool3d_check_rule(input_shape, input_dtype, ksize, strides, pads, data_format, kernel_name):
    """
    _avg_pool3d_check_rule

    Parameters
    ----------
    input_shape: input shape
    input_dtype: output dtype
    ksize: kernel size
    strides: strides
    pads: zero paddings on both sides
    data_format: must be "NDHWC"
    kernel_name: kernel name

    Returns
    -------
    None
    """
    para_check.check_shape(input_shape)
    para_check.check_dtype(input_dtype, ("float16", ))
    para_check.check_kernel_name(kernel_name)
    para_check.check_format(data_format, ("NDHWC", ))
    _check_window_rule(ksize, strides, pads)
    _check_global_rule(input_shape, ksize, pads)


# pylint: disable=too-many-locals,too-many-arguments,unused-argument,invalid-name
def avg_pool3d_compute(x,
                       y,
                       ksize,
                       strides,
                       pads,
                       ceil_mode=False,
                       count_include_pad=True,
                       divisor_override=0,
                       data_format="NDHWC",
                       kernel_name="avg_pool3d"):
    """
    avg_pool3d compute

    Parameters
    ----------
    x: input tensor dict
    y: output tensor dict
    ksize: kernel size
    strides: strides
    padding: padding mode, str
    data_format: must be "NDHWC"
    kernel_name: kernel name

    Returns
    -------
    output tensor
    """
    shape = x.shape

    ksize_d = ksize[1]
    stride_d = strides[1]
    size_kernel = (ksize[1] * ksize[2] * ksize[3])

    cast_type = "float16"
    if tbe_platform.intrinsic_check_support("Intrinsic_vadd", "float32"):
        cast_type = "float32"

    # copy gm to ub
    tensor_in_ub = tvm.compute(shape, lambda *i: x[i], name="tensor_in_ub")

    tensor_in_ub_cast = tvm.compute(shape, lambda *i: tensor_in_ub(*i).astype(cast_type), name="tensor_in_ub_cast")

    d_axis = tvm.reduce_axis((0, ksize_d), "d_sum")
    hw_axis = tvm.reduce_axis((0, shape[3]), "hw_sum")
    origin_d = shape[1]
    reduced_d = 1 + (origin_d - ksize_d) // stride_d
    shape_d_hw = (shape[0], reduced_d, shape[2], 1, shape[4])
    tensor_d_hw = tvm.compute(shape_d_hw,
                              lambda n, d, c1, hw, c0: tvm.sum(
                                  tensor_in_ub_cast[n, d * stride_d + d_axis, c1, hw_axis, c0], axis=[d_axis, hw_axis]),
                              name="tensor_d_hw")

    tensor_a = tvm.compute(
        shape_d_hw,
        lambda n, d, c1, hw, c0: tensor_d_hw[n, d, c1, hw, c0] * tvm.const(1.0 / size_kernel, dtype=cast_type),
        name="tensor_a")

    res_cast = tvm.compute(shape_d_hw, lambda *i: tensor_a(*i).astype("float16"), name="res_cast")

    res = tvm.compute(shape_d_hw, lambda *i: res_cast[i], name='res')
    return res


def _tiling_param(shape, ksize, strides, core_num):
    d = shape[1]
    hw = shape[3]
    c0 = shape[4]
    ksize_d = ksize[1]
    stride_d = strides[1]
    d_out = 1 + (d - ksize_d) // stride_d

    total_ub_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    factor_c1 = 1
    if hw * 5 * c0 * 4 > total_ub_bytes:
        # split reduce hw axis
        factor_dout = 1
        factor_rd = 1
        factor_rhw = total_ub_bytes // (5 * c0 * 4)
    elif ksize_d * hw * 5 * c0 * 4 > total_ub_bytes:
        # split reduce d axis
        factor_dout = 1
        factor_rd = total_ub_bytes // (hw * 5 * c0 * 4)
        factor_rhw = hw
    elif d_out * ksize_d * hw * c0 * 5 * 4 > total_ub_bytes:
        # split d_out axis
        factor_dout = total_ub_bytes // (ksize_d * hw * c0 * 5 * 4)
        factor_rd = ksize_d
        factor_rhw = hw
    else:
        # do not split any axis
        factor_dout = d_out
        factor_rd = ksize_d
        factor_rhw = hw

    return factor_c1, factor_dout, factor_rd, factor_rhw


def _avg_pool3d_schedule(res, sch, ksize, strides):
    res_cast = res.op.input_tensors[0]
    tensor_a = res_cast.op.input_tensors[0]
    tensor_d_hw = tensor_a.op.input_tensors[0]
    tensor_in_ub_cast = tensor_d_hw.op.input_tensors[0]
    tensor_in_ub = tensor_in_ub_cast.op.input_tensors[0]

    input_shape = [int(i) for i in tensor_in_ub.shape]
    # set scope
    sch[tensor_in_ub].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_in_ub_cast].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_d_hw].set_scope(tbe_platform.scope_ubuf)
    sch[tensor_a].set_scope(tbe_platform.scope_ubuf)
    sch[res_cast].set_scope(tbe_platform.scope_ubuf)

    core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    ax_res_n = res.op.axis[0]
    ax_res_do = res.op.axis[1]
    ax_res_c1 = res.op.axis[2]
    ax_res_hw = res.op.axis[3]
    ax_res_c0 = res.op.axis[4]

    ax_dhw_n = tensor_d_hw.op.axis[0]
    ax_dhw_do = tensor_d_hw.op.axis[1]
    ax_dhw_c1 = tensor_d_hw.op.axis[2]
    ax_dhw_hw = tensor_d_hw.op.axis[3]
    ax_dhw_c0 = tensor_d_hw.op.axis[4]
    ax_dhw_rd = tensor_d_hw.op.reduce_axis[0]
    ax_dhw_rhw = tensor_d_hw.op.reduce_axis[1]

    factor_c1, factor_dout, factor_reduce_d, factor_reduce_hw = _tiling_param(input_shape, ksize, strides, core_num)

    reduce_hw_o, reduce_hw_i = sch[tensor_d_hw].split(ax_dhw_rhw, factor=factor_reduce_hw)
    reduce_d_o, reduce_d_i = sch[tensor_d_hw].split(ax_dhw_rd, factor=factor_reduce_d)
    dhw_do_o, dhw_do_i = sch[tensor_d_hw].split(ax_dhw_do, factor=factor_dout)

    sch[tensor_d_hw].reorder(ax_dhw_n, ax_dhw_c1, ax_dhw_hw, dhw_do_o, reduce_d_o, reduce_hw_o, dhw_do_i, reduce_d_i,
                             reduce_hw_i, ax_dhw_c0)

    ax_res_c1_o, ax_res_c1_i = sch[res].split(ax_res_c1, factor=factor_c1)
    ax_res_do_o, ax_res_do_i = sch[res].split(ax_res_do, factor=factor_dout)

    sch[res].reorder(ax_res_n, ax_res_c1_o, ax_res_do_o, ax_res_c1_i, ax_res_do_i, ax_res_hw, ax_res_c0)

    ax_fused = sch[res].fuse(ax_res_n, ax_res_c1_o)

    block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(ax_fused, block)

    sch[tensor_in_ub].compute_at(sch[tensor_d_hw], reduce_hw_o)
    sch[tensor_in_ub_cast].compute_at(sch[tensor_d_hw], reduce_hw_o)
    sch[res_cast].compute_at(sch[res], ax_res_do_o)
    sch[tensor_a].compute_at(sch[res], ax_res_do_o)
    sch[tensor_d_hw].compute_at(sch[res], ax_res_do_o)

    if tensor_in_ub.dtype != tensor_in_ub_cast.dtype:
        sch[tensor_in_ub_cast].emit_insn(sch[tensor_in_ub_cast].op.axis[0], tbe_platform.CAST)
        sch[res_cast].emit_insn(sch[res_cast].op.axis[0], tbe_platform.CAST)
    sch[tensor_in_ub].emit_insn(sch[tensor_in_ub].op.axis[0], tbe_platform.DMA_COPY)
    sch[tensor_d_hw].emit_insn(dhw_do_i, tbe_platform.REDUCE_SUM)
    sch[tensor_a].emit_insn(sch[tensor_a].op.axis[0], tbe_platform.MUL)
    sch[res].emit_insn(ax_res_c1_i, tbe_platform.DMA_COPY)


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def avg_pool3d(x,
               y,
               ksize,
               strides,
               pads,
               ceil_mode=False,
               count_include_pad=True,
               divisor_override=0,
               data_format="NDHWC",
               kernel_name="avg_pool3d"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data,
        only support float16, shape is 5 dims, format is NDC1HWC0

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avg_pool3d,
            only support avg_pool3d in D or H or W

    strides : list or tuple, the stride of avg_pool3d window,
              only support avg_pool3d in D or H or W

    pads : list or tuple, count of padding zero or d,h,w axis

    ceil_mode: when True, will use ceil instead of floor in the formula to
               compute the output shape

    count_include_pad: when True, will include the zero-padding in the
                       averaging calculation.

    divisor_override: if specified, it will be used as divisor, otherwise size
                      of the pooling region will be used.

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "avg_pool3d"

    Returns
    -------
    None

    Notice
    -------
    Only support global model currently.
    """
    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()

    if len(ksize) == 1:
        ksize = (1, ksize[0], ksize[0], ksize[0], 1)
    elif len(ksize) == 3:
        ksize = (1, ksize[0], ksize[1], ksize[2], 1)

    if len(strides) == 1:
        strides = (1, strides[0], strides[0], strides[0], 1)
    elif len(strides) == 3:
        strides = (1, strides[0], strides[1], strides[2], 1)

    _avg_pool3d_check_rule(input_shape, input_dtype, ksize, strides, pads, data_format, kernel_name)

    # create tensor_in
    input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * input_shape[4], input_shape[5])
    tensor_in = tvm.placeholder(input_shape, name="tensor_in", dtype=input_dtype)

    res = avg_pool3d_compute(tensor_in, y, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override,
                             data_format, kernel_name)

    # schedule
    sch = tvm.create_schedule(res.op)

    _avg_pool3d_schedule(res, sch, ksize, strides)

    with tbe_platform.build_config:
        tvm.build(sch, [tensor_in, res], "cce", name=kernel_name)
