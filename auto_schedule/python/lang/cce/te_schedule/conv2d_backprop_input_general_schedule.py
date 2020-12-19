# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
conv2d backprop input general schedule.
"""
from functools import reduce
from te import tvm
from te.domain.tiling.get_tiling import get_tiling
from te.lang.cce.boost_schedule_kit import Compare
from te.lang.cce.boost_schedule_kit import ScheduleAgent
from te.lang.cce.te_compute.conv2d_backprop_input_general_compute import DeConvPattern
from te.lang.cce.te_compute.cube_util import shape_to_list
from te.lang.cce.te_compute.cube_util import GroupDictKeys
from te.lang.cce.te_schedule.util import L1CommonParam
from te.platform import cce_conf
from te.platform import cce_params
from te.tvm.schedule import InferBound
from te.tvm.schedule import ScheduleOps
from te.utils.error_manager import error_manager_util

# default false
DEBUG_MODE = False  # pylint: disable=C0302
# Don't modify, used in log_util
DX_SUPPORT_TAG_LOG_PREFIX = "#Conv2DBackpropInput only support#"
# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16
OUT_OF_ORDER_SHIFT_BIT = 13


def _raise_dx_general_err(msg):
    """
    In op Conv2DBackpropInput_general, [%s] % (msg)
    msg for discribe the error info
    the error info only for Conv2DBackpropInput_general's developers
    """
    args_dict = {"errCode": "E60108", "reason": msg}
    msg = error_manager_util.get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def _print_ir_conv(process, sch):
    """
    print ir for input sch

    Parameter:
    --------------------------------------------------------------
    :param process: tag
    :param sch: schedule
    :return: IR process
    ---------------------------------------------------------------
    """

    if DEBUG_MODE or process == "debug":
        start = process + " IR start"
        end = process + " IR end\n"
        print(start)
        sch = sch.normalize()
        bounds = InferBound(sch)
        stmt = ScheduleOps(sch, bounds, True)
        print(stmt)
        print(end)


def _ceil(divisor_a, divisor_b):
    """
    round up function

    Paramater:
    :param divisor_a: int.
    :param divisor_b: int.
    :return: int
    """
    if divisor_b == 0:
        _raise_dx_general_err("division by zero")
    return (divisor_a + divisor_b - 1) // divisor_b


def _align(x_1, x_2):
    """
    Get minimum y: y >= x_1 and y % x_2 == 0
    :param x_1:
    :param x_2:
    :return: minimum y: y >= x_1 and y % x_2 == 0
    """
    if x_2 == 0:
        _raise_dx_general_err("division by zero")
    return (x_1 + x_2 - 1) // x_2 * x_2


def general_schedule(
    tensor, sch_list, tiling_case=None, var_range=None
):  # pylint:disable=R0914,R0915
    """
    auto_schedule for cce AI-CORE.
    For now, only one convolution operation is supported.

    Parameters
    ----------
    res : tvm.tensor

    sch_list: use sch_list[0] to return conv schedule

    tiling_case: fix tiling for dynamic shape

    var_range: var_range for dynamic shape

    Returns
    -------
    True for sucess, False for no schedule
    """
    c_ddr = tensor
    sch = sch_list[0]
    tiling = None
    out_of_order = None

    def _set_output_mem():
        if out_mem == "fuse_flag":
            if "addr_type" in c_ddr.op.attrs:
                res_addr_type = c_ddr.op.attrs["addr_type"].value
            else:
                res_addr_type = 0
            output_memory_type = [res_addr_type]
            if res_addr_type == 1:
                sch[c_ddr].set_scope(cce_params.scope_cbuf_fusion)
        else:
            if out_mem == 1:
                sch[c_ddr].set_scope(cce_params.scope_cbuf_fusion)
            output_memory_type = [out_mem]

        return output_memory_type

    fusion_para = DeConvPattern.fusion_para_map
    l1_fusion_type = int(fusion_para.get("l1_fusion_type"))
    input_mem = [fusion_para.get("input_memory_type")]
    out_mem = fusion_para.get("output_memory_type")
    valid_shape = fusion_para.get("valid_shape")
    slice_offset = fusion_para.get("slice_offset")
    out_mem = _set_output_mem()
    fmap_l1_addr_flag = fusion_para.get("fmap_l1_addr_flag")
    fmap_l1_valid_size = fusion_para.get("fmap_l1_valid_size")
    cube_vector_split = cce_conf.get_soc_spec("CUBE_VECTOR_SPLIT")

    def _config_dynamic_mode(var_range):
        if not var_range:
            return None
        if "batch_n" in var_range and len(var_range) == 1:
            return "dynamic_batch"
        for var in ("dedy_h", "dedy_w", "dx_h", "dx_w"):
            if var in var_range:
                return "dynamic_hw"
        return None

    def _fetch_tensor_info(dynamic_mode):  # pylint:disable=R0914,R0912,R0915
        def _get_vadd_tensors(vadd_res_tensor):
            left_tensor = vadd_res_tensor.op.input_tensors[0]
            right_tensor = vadd_res_tensor.op.input_tensors[1]
            if left_tensor.op.tag == "conv2d_backprop_input":
                c_ub_cut = left_tensor
                vadd_tensor = right_tensor
            else:
                c_ub_cut = right_tensor
                vadd_tensor = left_tensor
            return c_ub_cut, vadd_tensor

        def _checkout_quant_fusion():
            temp_tensor = deconv_res
            if deconv_res.op.tag == "quant":
                tensor_attr["n0_32_flag"] = True
                tensor_attr["q_mode"] = deconv_res.op.attrs["round_mode"].value
            while temp_tensor.op.input_tensors:
                if temp_tensor.op.tag in ("dequant_vector", "dequant1_vector"):
                    tensor_map["c_ub"] = temp_tensor
                    tensor_attr["deq_vector"] = True
                    tensor_attr["quant_fuse"] = True
                    tensor_map["deq"] = temp_tensor.op.input_tensors[1]
                elif temp_tensor.op.tag in ("dequant_scale", "dequant1_scale"):
                    tensor_map["c_ub"] = temp_tensor
                    tensor_attr["quant_fuse"] = True
                    tensor_map["deq"] = temp_tensor.op.input_tensors[1]
                    tensor_attr["deq_vector"] = False
                elif (
                    len(temp_tensor.op.input_tensors) == 2
                    and "elewise" in temp_tensor.op.tag
                ):
                    input_tensor = temp_tensor.op.input_tensors[0]
                    if not input_tensor.op.input_tensors:
                        temp_tensor = temp_tensor.op.input_tensors[1]
                        continue
                elif (
                    "dequant_remove_pad" in temp_tensor.op.tag
                    and deconv_res.op.tag != temp_tensor.op.tag
                ):
                    sch[temp_tensor].compute_inline()
                temp_tensor = temp_tensor.op.input_tensors[0]

        def _fetch_elewise_fusion():
            ub_list = []
            input_tensor_list = []
            temp_tensor = deconv_res
            c_ub_res = sch.cache_write(deconv_res, cce_params.scope_ubuf)
            while temp_tensor.op.input_tensors:
                if len(temp_tensor.op.input_tensors) == 2:
                    input_tensor = temp_tensor.op.input_tensors[0]
                    input_tensor_des = (
                        c_ub_res if temp_tensor == deconv_res else temp_tensor
                    )
                    if not input_tensor.op.input_tensors:
                        temp_tensor = temp_tensor.op.input_tensors[1]
                    else:
                        input_tensor = temp_tensor.op.input_tensors[1]
                        temp_tensor = temp_tensor.op.input_tensors[0]
                    input_tensor_list.append([input_tensor, input_tensor_des])
                    continue
                if temp_tensor.op.tag == "conv2d_backprop_input":
                    c_ub_cut = temp_tensor
                    break
                if temp_tensor == deconv_res:
                    ub_list.append(c_ub_res)
                else:
                    ub_list.append(temp_tensor)
                temp_tensor = temp_tensor.op.input_tensors[0]
            tensor_attr["elewise_fuse"] = True
            tensor_map["ub_list"] = ub_list
            tensor_map["input_tensor_list"] = input_tensor_list
            return c_ub_cut

        def _fetch_quant_info():
            ub_list = []
            input_cache_buffer = []
            temp_tensor = deconv_res
            while temp_tensor.op.input_tensors:
                if (
                    len(temp_tensor.op.input_tensors) == 2
                    and "elewise" in temp_tensor.op.tag
                ):
                    input_tensor = temp_tensor.op.input_tensors[0]
                    c_ub_res = temp_tensor
                    if deconv_res.op.tag == temp_tensor.op.tag:
                        c_ub_res = sch.cache_write(deconv_res, cce_params.scope_ubuf)
                    if not input_tensor.op.input_tensors:
                        input_cache_buffer.append([input_tensor, c_ub_res])
                        temp_tensor = temp_tensor.op.input_tensors[1]
                        continue
                    input_tensor = temp_tensor.op.input_tensors[1]
                    input_cache_buffer.append([input_tensor, c_ub_res])
                elif temp_tensor.op.name in (
                    "cast_i8_ub",
                    "scale_sqrt_ub",
                    "offset_ub",
                    "reform_by_vmuls",
                    "reform_by_vadds"
                ):
                    ub_list.append(temp_tensor)
                elif (
                    temp_tensor.op.name in ("dequant2", "dequant_relu")
                    or "elewise" in temp_tensor.op.tag
                ):
                    if deconv_res.op.tag == temp_tensor.op.tag:
                        c_ub_res = sch.cache_write(deconv_res, cce_params.scope_ubuf)
                        ub_list.append(c_ub_res)
                    else:
                        ub_list.append(temp_tensor)
                elif temp_tensor.op.name == "input_ub":
                    if temp_tensor.op.attrs["c_out"].value % 2:
                        ub_list.append(temp_tensor)
                    else:
                        sch[temp_tensor].compute_inline()
                temp_tensor = temp_tensor.op.input_tensors[0]
            tensor_map["input_tensor"] = input_cache_buffer
            tensor_map["ub_list"] = ub_list

        def _fill_tensor_map(c_col, dynamic_mode, tensor_map):  # pylint: disable=R0915
            a_col = c_col.op.input_tensors[0]  # im2col_fractal in L0A
            b_col = c_col.op.input_tensors[1]  # weight_transform in L0B
            b_ddr = b_col.op.input_tensors[0]  # weight in ddr

            if not dynamic_mode:
                # im2col_row_major in L1
                a_col_before = a_col.op.input_tensors[0]
            else:
                a_col_before = a_col
                kernel_h, kernel_w = (
                    a_col.op.attrs["kernel_h"],
                    a_col.op.attrs["kernel_w"]
                )
                tensor_attr["kernel_h"] = int(kernel_h)
                tensor_attr["kernel_w"] = int(kernel_w)
            padding = shape_to_list(a_col_before.op.attrs["padding"])
            dilations = shape_to_list(a_col_before.op.attrs["dilation"])

            tensor_map["c_col"] = c_col
            tensor_map["a_col"] = a_col
            tensor_map["a_col_before"] = a_col_before
            tensor_map["b_col"] = b_col
            tensor_map["b_ddr"] = b_ddr
            tensor_attr["padding"] = padding
            tensor_attr["dilations"] = dilations

            def _fill_a_tensormap_dynamic():
                if (
                    a_col_before.op.input_tensors[0].op.tag == "dy_l1"
                    or a_col_before.op.input_tensors[0].op.tag == "dy_l1_cut"
                ):
                    a_l1 = a_col_before.op.input_tensors[0]
                    a_filling = a_l1.op.input_tensors[0]
                    vn_tensor = a_filling.op.input_tensors[1]
                    a_zero = vn_tensor.op.input_tensors[0]
                    a_one = vn_tensor.op.input_tensors[1]
                    stride_h, stride_w = list(
                        i.value for i in a_filling.op.attrs["stride_expand"]
                    )
                    a_ddr = a_filling.op.input_tensors[0]  # dEdY in ddr

                    tensor_map["a_l1"] = a_l1
                    tensor_map["a_filling"] = a_filling
                    tensor_map["vn_tensor"] = vn_tensor
                    tensor_map["a_one"] = a_one
                    tensor_map["a_zero"] = a_zero

                    a_ub = sch.cache_read(a_ddr, cce_params.scope_ubuf, [a_filling])
                    sch[a_zero].set_scope(cce_params.scope_ubuf)
                    sch[a_one].set_scope(cce_params.scope_ubuf)
                    sch[vn_tensor].set_scope(cce_params.scope_ubuf)
                    sch[a_filling].set_scope(cce_params.scope_ubuf)
                    sch[a_l1].set_scope(cce_params.scope_cbuf)
                    tensor_map["a_ub"] = a_ub
                else:
                    a_ddr = a_col_before.op.input_tensors[0]  # dEdY in ddr
                    stride_h = 1
                    stride_w = 1
                    a_l1 = sch.cache_read(a_ddr, cce_params.scope_cbuf, [a_col_before])
                    tensor_map["a_l1"] = a_l1
                tensor_map["a_ddr"] = a_ddr
                tensor_attr["stride_h"] = stride_h
                tensor_attr["stride_w"] = stride_w

            def _fill_a_tensormap():  # pylint: disable=R0915
                def _fill_a_dilate():
                    a_l1 = a_col_before.op.input_tensors[0]
                    a_filling = a_l1.op.input_tensors[0]
                    stride_h, stride_w = list(
                        i.value for i in a_filling.op.attrs["stride_expand"]
                    )
                    a_zero = a_filling.op.input_tensors[1]  # dEdY_zero in ub
                    tensor_map["a_filling"] = a_filling
                    tensor_map["a_zero"] = a_zero
                    # generate a_zero in ub
                    sch[a_zero].set_scope(cce_params.scope_ubuf)
                    sch[a_filling].set_scope(cce_params.scope_ubuf)
                    tensor_map["a_l1"] = a_l1
                    sch[a_l1].set_scope(cce_params.scope_cbuf)
                    if valid_shape:
                        read_select = a_filling.op.input_tensors[0]
                        a_ddr = read_select.op.input_tensors[0]
                        if l1_fusion_type == 0 and input_mem[0] == 0:
                            sch[read_select].set_scope(cce_params.scope_ubuf)
                            tensor_map["a_ub"] = read_select
                        elif input_mem[0] == 1:
                            sch[read_select].set_scope(cce_params.scope_ubuf)
                            tensor_map["a_ub"] = read_select
                            sch[a_ddr].set_scope(cce_params.scope_cbuf_fusion)
                        elif l1_fusion_type == 1:
                            sch[read_select].set_scope(cce_params.scope_cbuf_fusion)
                            tensor_map["a_l1_full"] = read_select
                            a_ub = sch.cache_read(
                                read_select, cce_params.scope_ubuf, [a_filling]
                            )
                            tensor_map["a_ub"] = a_ub
                    else:
                        a_ddr = a_filling.op.input_tensors[0]
                        if input_mem[0] == 0 and l1_fusion_type != 1:
                            a_ub = sch.cache_read(
                                a_ddr, cce_params.scope_ubuf, [a_filling]
                            )
                        elif input_mem[0] == 1:
                            a_ub = sch.cache_read(
                                a_ddr, cce_params.scope_ubuf, [a_filling]
                            )
                            sch[a_ddr].set_scope(cce_params.scope_cbuf_fusion)
                        elif l1_fusion_type == 1:
                            a_l1_full = sch.cache_read(
                                a_ddr, cce_params.scope_cbuf_fusion, [a_filling]
                            )
                            tensor_map["a_l1_full"] = a_l1_full
                            a_ub = sch.cache_read(
                                a_l1_full, cce_params.scope_ubuf, [a_filling]
                            )
                        tensor_map["a_ub"] = a_ub
                    if tensor_map.get("a_l1_full") is not None:
                        a_l1_full = tensor_map.get("a_l1_full")
                        al1_shape = a_l1_full.shape
                        sch[a_l1_full].buffer_align(
                            (1, 1),
                            (1, 1),
                            (al1_shape[2], al1_shape[2]),
                            (al1_shape[3], al1_shape[3]),
                            (1, 1)
                        )
                    tensor_map["a_ddr"] = a_ddr
                    tensor_attr["stride_h"] = stride_h
                    tensor_attr["stride_w"] = stride_w

                def _fill_a():
                    if valid_shape and input_mem[0] == 0:
                        a_l1 = a_col_before.op.input_tensors[0]
                        a_ddr = a_l1.op.input_tensors[0]
                        stride_h = 1
                        stride_w = 1
                        sch[a_l1].set_scope(cce_params.scope_cbuf_fusion)
                    elif a_col_before.op.input_tensors[0].op.tag == "dy_l1_modify":
                        a_l1 = a_col_before.op.input_tensors[0]
                        a_ddr = a_l1.op.input_tensors[0]
                        stride_h = 1
                        stride_w = 1
                        sch[a_l1].set_scope(cce_params.scope_cbuf)
                    else:
                        a_ddr = a_col_before.op.input_tensors[0]
                        stride_h = 1
                        stride_w = 1
                        if l1_fusion_type != -1:
                            a_l1 = sch.cache_read(
                                a_ddr, cce_params.scope_cbuf_fusion, [a_col_before]
                            )
                        else:
                            a_l1 = sch.cache_read(
                                a_ddr, cce_params.scope_cbuf, [a_col_before]
                            )
                        if input_mem[0] == 1:
                            sch[a_ddr].set_scope(cce_params.scope_cbuf_fusion)
                    tensor_map["a_l1"] = a_l1
                    if input_mem[0] == 1 or l1_fusion_type == 1:
                        al1_shape = a_l1.shape
                        sch[a_l1].buffer_align(
                            (1, 1),
                            (1, 1),
                            (al1_shape[2], al1_shape[2]),
                            (al1_shape[3], al1_shape[3]),
                            (1, 1)
                        )
                    tensor_map["a_ddr"] = a_ddr
                    tensor_attr["stride_h"] = stride_h
                    tensor_attr["stride_w"] = stride_w

                if (
                    a_col_before.op.input_tensors[0].op.tag == "dy_l1"
                    or a_col_before.op.input_tensors[0].op.tag == "dy_l1_cut"
                ):
                    _fill_a_dilate()
                else:
                    _fill_a()

            if dynamic_mode:
                _fill_a_tensormap_dynamic()
            else:
                _fill_a_tensormap()
            b_l1 = sch.cache_read(b_ddr, cce_params.scope_cbuf, [b_col])
            tensor_map["b_l1"] = b_l1
            # dataflow management
            sch[b_col].set_scope(cce_params.scope_cb)
            sch[a_col_before].set_scope(cce_params.scope_cbuf)
            sch[a_col].set_scope(cce_params.scope_ca)
            sch[c_col].set_scope(cce_params.scope_cc)
            if not cube_vector_split:
                sch[c_ub].set_scope(cce_params.scope_ubuf)

            return tensor_map

        def _bias_tensor_setscope():
            # when add bias in ub
            bias_add_vector = tensor_map.get("bias_add_vector")
            if bias_add_vector is not None:
                sch[bias_add_vector].set_scope(cce_params.scope_ubuf)
                bias_tensor = bias_add_vector.op.input_tensors[1]
                bias_ub = sch.cache_read(
                    bias_tensor, cce_params.scope_ubuf, [bias_add_vector]
                )
                tensor_map["bias_ub"] = bias_ub

            # when add bias in l0c
            if tensor_map.get("c_add_bias") is not None:
                c_add_bias = tensor_map.get("c_add_bias")
                bias_l0c = tensor_map.get("bias_l0c")
                bias_ub_brc = tensor_map.get("bias_ub_brc")
                sch[c_add_bias].set_scope(cce_params.scope_cc)
                sch[bias_l0c].set_scope(cce_params.scope_cc)
                sch[bias_ub_brc].set_scope(cce_params.scope_ubuf)

        def _bias_tensor():
            if c_ub.op.input_tensors[0].name == "c_add_bias":
                c_add_bias = c_ub.op.input_tensors[0]
                bias_l0c = c_add_bias.op.input_tensors[0]
                c_col = c_add_bias.op.input_tensors[1]
                bias_ub_brc = bias_l0c.op.input_tensors[0]
                tensor_bias = bias_ub_brc.op.input_tensors[0]
                bias_ub = sch.cache_read(
                    tensor_bias, cce_params.scope_ubuf, [bias_ub_brc]
                )
                tensor_map["c_add_bias"] = c_add_bias
                tensor_map["bias_l0c"] = bias_l0c
                tensor_map["bias_ub_brc"] = bias_ub_brc
                tensor_map["tensor_bias"] = tensor_bias
                tensor_map["bias_ub"] = bias_ub
            else:
                if cube_vector_split:
                    c_col = c_ddr.op.input_tensors[0]
                else:
                    c_col = c_ub.op.input_tensors[0]
            return c_col

        def _ubtensor_setscope(ub_tensor_list, input_tensor_list, fusion_num, deq_list):
            for ub_tensor in ub_tensor_list:
                sch[ub_tensor].set_scope(cce_params.scope_ubuf)
                if "dequant2" in ub_tensor.op.name:
                    deq_list.append(ub_tensor)
                if "dequant" in ub_tensor.op.name and deconv_res.op.tag != "quant":
                    fusion_num += 1
            if input_tensor_list and deconv_res.op.tag != "quant":
                fusion_num += 1
            for input_tensor_mem in input_tensor_list:
                sch[input_tensor_mem[1]].set_scope(cce_params.scope_ubuf)
                input_ub = sch.cache_read(
                    input_tensor_mem[0], cce_params.scope_ubuf, input_tensor_mem[1]
                )
                input_tensor_mem[0] = input_ub
            return fusion_num

        def _tensor_setscope():
            fusion_param = 0
            if deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool":
                fusion_param = 1 / 16
                if "elewise_binary_add" in deconv_res.op.input_tensors[1].op.tag:
                    fusion_param += 1
                    sch[vadd_res].set_scope(cce_params.scope_ubuf)
                    vadd_tensor_ub = sch.cache_read(
                        vadd_tensor, cce_params.scope_ubuf, [vadd_res]
                    )
                    tensor_map["vadd_tensor_ub"] = vadd_tensor_ub
                sch[c_ub_cut].set_scope(cce_params.scope_ubuf)
                mask_ub = sch.cache_read(mask, cce_params.scope_ubuf, [deconv_res])
                c_ub_drelu = sch.cache_write(deconv_res, cce_params.scope_ubuf)
                tensor_map["mask_ub"] = mask_ub
                tensor_map["c_ub_drelu"] = c_ub_drelu
            elif "requant_remove_pad" in deconv_res.op.tag:
                deq = tensor_map.get("deq")
                sch[tensor_map["c_ub"]].set_scope(cce_params.scope_ubuf)
                deq_ub = sch.cache_read(deq, cce_params.scope_ubuf, tensor_map["c_ub"])
                tensor_map["deq"] = deq_ub
                sch[tensor_map["data_transfer"]].set_scope(cce_params.scope_ubuf)
                sch[tensor_map["c_ub"]].compute_inline()
                sch[tensor_map["c_ub"]].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
                tensor_map["c_ub"] = tensor_map["data_transfer"]
            elif tensor_attr.get("quant_fuse"):
                deq_list = [
                    tensor_map["c_ub"]
                ]
                sch[tensor_map["c_ub"]].set_scope(cce_params.scope_ubuf)
                ub_list = tensor_map["ub_list"]
                input_tensor = tensor_map["input_tensor"]
                if deconv_res.op.tag == "quant":
                    fusion_param = 4
                else:
                    fusion_param = 0
                fusion_param = _ubtensor_setscope(
                    ub_list, input_tensor, fusion_param, deq_list
                )
                deq = tensor_map.get("deq")
                deq_ub = sch.cache_read(deq, cce_params.scope_ubuf, deq_list)
                tensor_map["deq"] = deq_ub
                if input_tensor and deconv_res.op.tag != "quant":
                    fusion_param += 1
                for input_tensor_mem in input_tensor:
                    sch[input_tensor_mem[1]].set_scope(cce_params.scope_ubuf)
                    input_ub = sch.cache_read(
                        input_tensor_mem[0], cce_params.scope_ubuf, input_tensor_mem[1]
                    )
                    input_tensor_mem[0] = input_ub
                fusion_param += 0.125
            elif "elewise" in deconv_res.op.tag:
                fusion_param = 0
                sch[tensor_map["c_ub"]].set_scope(cce_params.scope_ubuf)
                for ub_tensor in tensor_map["ub_list"]:
                    sch[ub_tensor].set_scope(cce_params.scope_ubuf)
                for input_tensor in tensor_map["input_tensor_list"]:
                    fusion_param = 1
                    sch[input_tensor[1]].set_scope(cce_params.scope_ubuf)
                    input_ub = sch.cache_read(
                        input_tensor[0], cce_params.scope_ubuf, input_tensor[1]
                    )
                    input_tensor[0] = input_ub
                if "bias_add_vector" in tensor_map:
                    fusion_param += 0.125
                sch[tensor_map["c_ub_cut"]].compute_inline()
            return fusion_param

        tensor_map = {}
        tensor_attr = {}
        if c_ddr.op.tag == "write_select":
            tensor_attr["write_select"] = True
        if deconv_res.op.tag == "requant_remove_pad":
            tensor_attr["n0_32_flag"] = True
            tensor_attr["quant_fuse"] = True
            tensor_map["data_transfer"] = deconv_res.op.input_tensors[0]
            tensor_map["c_ub"] = tensor_map["data_transfer"].op.input_tensors[0]
            tensor_map["deq"] = tensor_map["c_ub"].op.input_tensors[1]
            c_ub_ddr = tensor_map["c_ub"].op.input_tensors[0]
            c_ub = c_ub_ddr.op.input_tensors[0]
            output_shape = list(i.value for i in c_ub_ddr.op.attrs["output_shape"])
            group_dict = c_ub_ddr.op.attrs["group_dict"]
            tensor_attr["group_dict"] = group_dict
            tensor_attr["output_shape"] = output_shape
            tensor_map["c_ub_cut"] = c_ub_ddr
            sch[c_ub_ddr].compute_inline()
            sch[c_ub].compute_inline()
            sch[c_ub_ddr].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
        elif "quant" in deconv_res.op.tag or "elewise" in deconv_res.op.tag:
            _checkout_quant_fusion()
            if tensor_attr.get("quant_fuse"):
                _fetch_quant_info()
                c_ub_ddr = tensor_map["c_ub"].op.input_tensors[0]
                c_ub = c_ub_ddr.op.input_tensors[0]
                output_shape = list(i.value for i in c_ub_ddr.op.attrs["output_shape"])
                group_dict = c_ub_ddr.op.attrs["group_dict"]
                tensor_attr["group_dict"] = group_dict
                tensor_attr["output_shape"] = output_shape
                tensor_map["c_ub_cut"] = c_ub_ddr
                sch[c_ub_ddr].compute_inline()
                sch[c_ub].compute_inline()
                sch[c_ub_ddr].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
        if tensor_attr.get("quant_fuse"):
            pass
        elif deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool":
            mask = deconv_res.op.input_tensors[0]
            if "elewise_binary_add" in deconv_res.op.input_tensors[1].op.tag:
                vadd_res = deconv_res.op.input_tensors[1]
                c_ub_cut, vadd_tensor = _get_vadd_tensors(vadd_res)
                tensor_map["vadd_res"] = vadd_res
            else:
                c_ub_cut = deconv_res.op.input_tensors[1]
            c_ub = c_ub_cut.op.input_tensors[0]
            output_shape = list(i.value for i in c_ub_cut.op.attrs["output_shape"])

            group_dict = c_ub_cut.op.attrs["group_dict"]
            tensor_attr["group_dict"] = group_dict
            tensor_map["c_ub_cut"] = c_ub_cut
            tensor_map["c_ub"] = c_ub
            tensor_attr["output_shape"] = output_shape
        elif "elewise" in deconv_res.op.tag:
            c_ub_cut = _fetch_elewise_fusion()
            c_ub = c_ub_cut.op.input_tensors[0]
            if c_ub.op.name == "bias_add_vector":
                tensor_map["bias_add_vector"] = c_ub
                c_ub = c_ub.op.input_tensors[0]
            output_shape = list(i.value for i in c_ub_cut.op.attrs["output_shape"])
            group_dict = c_ub_cut.op.attrs["group_dict"]
            tensor_attr["group_dict"] = group_dict
            tensor_map["c_ub_cut"] = c_ub_cut
            tensor_map["c_ub"] = c_ub
            tensor_attr["output_shape"] = output_shape
        elif deconv_res.op.tag == "conv2d_backprop_input":
            c_ub = deconv_res.op.input_tensors[0]
            output_shape = shape_to_list(deconv_res.op.attrs["output_shape"])
            group_dict = deconv_res.op.attrs["group_dict"]
            tensor_attr["group_dict"] = group_dict
            if c_ub.op.name == "bias_add_vector":
                tensor_map["bias_add_vector"] = c_ub
                c_ub = c_ub.op.input_tensors[0]
            tensor_map["c_ub"] = c_ub
            tensor_attr["output_shape"] = output_shape
        else:
            _raise_dx_general_err(
                DX_SUPPORT_TAG_LOG_PREFIX
                + " dx or dx+drelu or dx+elewise or dx+vadd+drelu"
            )

        c_col = _bias_tensor()
        tensor_map = _fill_tensor_map(c_col, dynamic_mode, tensor_map)
        _bias_tensor_setscope()

        tensor_attr["fusion_param"] = _tensor_setscope()

        return tensor_map, tensor_attr

    def _get_fusion_type():
        fusion_type = 0
        # the single deconv fusion is 1 for fp16, 2 for int8
        if deconv_res.op.tag == "conv2d_backprop_input":
            if a_ddr.dtype == "int8":
                fusion_type = 2
            else:
                fusion_type = 1
        # deonv+add+drelu fusion type is 4, deonv+drelu fusion type is 8
        elif deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool":
            if "elewise_binary_add" in deconv_res.op.input_tensors[1].op.tag:
                fusion_type = 4
            else:
                fusion_type = 8
        # deconv+requant is 7
        elif deconv_res.op.tag == "requant_remove_pad":
            fusion_type = 7
        # deconv+dequant is 5 and quant is 6
        elif tensor_attr.get("quant_fuse"):
            if deconv_res.op.tag == "quant":
                fusion_type = 6
            else:
                fusion_type = 5
        # deconv + relu is 3
        elif "elewise" in deconv_res.op.tag:
            fusion_type = 3
        # when in write mode ++20
        if tensor_attr.get("write_select"):
            fusion_type += 20
        return fusion_type

    def get_deconv_out():
        if c_ddr.op.tag == "write_select":
            deconv_res = c_ddr.op.input_tensors[0]
        else:
            deconv_res = c_ddr
        return deconv_res

    # check tiling and set default tiling
    def check_and_set_default_tiling(  # pylint: disable=R0913
        tiling, atype, btype, stride_h, stride_w, filter_shape
    ):
        """
        check and set default tiling
        :param tiling:
        :param atype:
        :param btype:
        :param stride_h:
        :param stride_w:
        :param filter_shape:
        :return: default tiling
        """
        # check flag
        def _check_tiling(tiling):
            if tiling["AL0_matrix"][2] == 32:
                return False
            return True

        if not _check_tiling(tiling):
            tiling = {}
            _, _, k_w, k_h, _ = filter_shape
            bit_dir = {
                "float32": 16,
                "int32": 16,
                "float16": 16,
                "int8": 32
            }
            if atype in bit_dir.keys():
                k_al1 = k_w * k_h * 16
                k_al0 = bit_dir[atype]
            else:
                # defaut value 32
                k_al1 = 32
                k_al0 = 32

            if btype in bit_dir.keys():
                k_bl1 = bit_dir[atype]
                k_bl0 = bit_dir[atype]
            else:
                # defaut value 32
                k_bl1 = 32
                k_bl0 = 32

            if stride_h > 1 or stride_w > 1:
                tiling["AUB_shape"] = [k_w * k_h * 16, 1, 1, 1]
                tiling["BUB_shape"] = None
            else:
                tiling["AUB_shape"] = None
                tiling["BUB_shape"] = None

            tiling["AL1_shape"] = [k_al1, 1, 1, 1]
            tiling["BL1_shape"] = [k_bl1, 1, 1, 1]
            tiling["AL0_matrix"] = [1, 1, 16, k_al0, 1, 1]
            tiling["BL0_matrix"] = [1, 1, 16, k_bl0, 1, 1]
            tiling["CL0_matrix"] = [1, 1, 16, 16, 1, 1]
            tiling["CUB_matrix"] = [1, 1, 16, 16, 1, 1]
            tiling["block_dim"] = [1, 1, 1, 1]
            tiling["n_bef_batch_flag"] = 0
            tiling["n_bef_group_flag"] = 0
            tiling["batch_bef_group_fla"] = 0
            tiling["A_overhead_opt_flag"] = 0
            tiling["B_overhead_opt_flag"] = 0
            tiling["AUB_channel_wise_flag"] = None
            tiling["BUB_channel_wise_flag"] = None
            tiling["CUB_channel_wise_flag"] = None
            tiling["manual_pingpong_buffer"] = {
                "AUB_pbuffer": 1,
                "BUB_pbuffer": 1,
                "AL1_pbuffer": 1,
                "BL1_pbuffer": 1,
                "AL0_pbuffer": 1,
                "BL0_pbuffer": 1,
                "CL0_pbuffer": 1,
                "CUB_pbuffer": 1,
                "UBG_pbuffer": 1
            }
        return tiling

    deconv_res = get_deconv_out()

    dynamic_mode = _config_dynamic_mode(var_range)
    tensor_map, tensor_attr = _fetch_tensor_info(dynamic_mode)
    vadd_res = tensor_map.get("vadd_res")
    c_ub_cut = tensor_map.get("c_ub_cut")
    c_ub = tensor_map.get("c_ub")
    c_col = tensor_map.get("c_col")
    a_col = tensor_map.get("a_col")
    b_col = tensor_map.get("b_col")
    b_ddr = tensor_map.get("b_ddr")
    a_col_before = tensor_map.get("a_col_before")
    a_l1 = tensor_map.get("a_l1")
    a_filling = tensor_map.get("a_filling")
    vn_tensor = tensor_map.get("vn_tensor")
    a_one = tensor_map.get("a_one")
    a_zero = tensor_map.get("a_zero")
    a_ddr = tensor_map.get("a_ddr")
    b_l1 = tensor_map.get("b_l1")
    a_ub = tensor_map.get("a_ub")
    a_l1_full = tensor_map.get("a_l1_full")
    vadd_tensor_ub = tensor_map.get("vadd_tensor_ub")
    mask_ub = tensor_map.get("mask_ub")
    c_ub_drelu = tensor_map.get("c_ub_drelu")
    c_ub_vadd = tensor_map.get("c_ub_vadd")
    bias_add_vector = tensor_map.get("bias_add_vector")
    bias_ub = tensor_map.get("bias_ub")
    c_add_bias = tensor_map.get("c_add_bias")
    bias_l0c = tensor_map.get("bias_l0c")
    bias_ub_brc = tensor_map.get("bias_ub_brc")

    output_shape = tensor_attr.get("output_shape")
    padding = tensor_attr.get("padding")
    dilations = tensor_attr.get("dilations")
    dilation_h, dilation_w = dilations
    stride_h = tensor_attr.get("stride_h")
    stride_w = tensor_attr.get("stride_w")
    fusion_param = tensor_attr.get("fusion_param")
    n0_32_flag = tensor_attr.get("n0_32_flag")
    group_dict = tensor_attr.get("group_dict")

    g_after = group_dict[GroupDictKeys.g_extend].value
    cin1_g = group_dict[GroupDictKeys.dx_c1_extend].value
    cou1_g = group_dict[GroupDictKeys.dy_c1_extend].value

    # fetch tiling
    padu, padd, padl, padr = padding
    # n, howo, c1, k_h, k_w, c0
    if not dynamic_mode:
        _, howo_mad, _, kernel_h, kernel_w, _ = shape_to_list(a_col_before.shape)
    else:
        kernel_h = tensor_attr.get("kernel_h")
        kernel_w = tensor_attr.get("kernel_w")
    _, _, dx_h, dx_w, _ = output_shape
    output_shape_g = [g_after, output_shape[0], cin1_g] + output_shape[2:]

    def get_img_shape():
        img_shape = shape_to_list(a_ddr.shape)
        if l1_fusion_type != -1 and valid_shape:
            img_shape = list(valid_shape)
        return img_shape

    img_shape = get_img_shape()
    _, dy_cout1, dy_h, dy_w, _ = img_shape  # pylint: disable=W0632
    img_shape_g = [g_after, img_shape[0], cou1_g] + img_shape[2:]
    # conv1d_situation
    def _check_conv1d_situation():
        if (
            not dynamic_mode
            and dx_h == 1
            and dy_h == 1
            and dilation_h == 1
            and stride_h == 1
        ):
            return True
        return False

    is_conv1d_situation = _check_conv1d_situation()

    w_trans_flag = False
    mad_type = "float32"
    if b_ddr.dtype == "int8":
        w_trans_flag = True
        mad_type = "int32"

    def _set_filter_shape():
        if w_trans_flag:
            # GCout1HkWk, Cin1, Cin0, Cout0
            b_ddr_k1, b_ddr_n1, b_ddr_n0, b_ddr_k0 \
                = list(i.value for i in b_ddr.shape)
            # G, Cout, Cin1, Hk, Wk, Cin0
            filter_shape_g = [cou1_g * b_ddr_k0,
                              cin1_g, kernel_h, kernel_w, b_ddr_n0]
        else:
            # GCin1HkWk, Cout1, Cout0, Cin0
            b_ddr_n1, b_ddr_k1, b_ddr_k0, b_ddr_n0 \
                = shape_to_list(b_ddr.shape)
            # Cout, Cin1, Hk, Wk, Cin0
            filter_shape_g = (cou1_g * b_ddr_k0,
                              cin1_g,
                              kernel_h, kernel_w, b_ddr_n0)
        if deconv_res.dtype == "int8":
            filter_shape_g[1] = (filter_shape_g[1] + 1) // 2 * 2

        return filter_shape_g
    filter_shape_g = _set_filter_shape()

    def _set_bias_flag(c_add_bias, bias_add_vector):
        if c_add_bias is not None or bias_add_vector is not None:
            bias_flag = 1
        else:
            bias_flag = 0
        return bias_flag

    bias_flag = _set_bias_flag(c_add_bias, bias_add_vector)

    def _get_kernel_name():
        if (
            fusion_param > 0
            or tensor_attr.get("quant_fuse")
            or tensor_attr.get("elewise_fuse")
        ):
            _kernel_name = c_ub_cut.op.attrs["kernel_name"]
        else:
            _kernel_name = deconv_res.op.attrs["kernel_name"]
        return _kernel_name

    _kernel_name = _get_kernel_name()

    if not dynamic_mode:
        fusion_type = _get_fusion_type()
        info_dict = {
            "op_type": "conv2d_backprop_input",
            "A_shape": list(img_shape_g[1:]),
            "B_shape": list(filter_shape_g),
            "C_shape": list(output_shape_g[1:]),
            "A_dtype": str(a_ddr.dtype),
            "B_dtype": str(b_ddr.dtype),
            "C_dtype": str(c_ddr.dtype),
            "mad_dtype": str(mad_type),
            "padl": padl,
            "padr": padr,
            "padu": padu,
            "padd": padd,
            "strideH": 1,
            "strideW": 1,
            "strideH_expand": stride_h,
            "strideW_expand": stride_w,
            "dilationH": dilation_h,
            "dilationW": dilation_w,
            "group": g_after,
            "bias_flag": bias_flag,
            "fused_double_operand_num": fusion_param,
            "kernel_name": _kernel_name.value,
            "in_fm_memory_type": input_mem,
            "out_fm_memory_type": out_mem,
            "l1_fusion_type": l1_fusion_type,
            "fusion_type": fusion_type,
            "general_flag": True,
        }
        tiling = get_tiling(info_dict)
        if tiling.get("compile_para") is not None:
            out_of_order = (tiling.get("compile_para") >> OUT_OF_ORDER_SHIFT_BIT) & 1
    else:
        tiling = tiling_case

    tiling = check_and_set_default_tiling(tiling, a_ddr.dtype,
                                          b_ddr.dtype, stride_h,
                                          stride_w, filter_shape_g)
    if DEBUG_MODE:
        print('general input shape: ', 'filter_g: ', filter_shape_g, 'dy: ',
              output_shape, 'dx: ', img_shape)
        print('reshape according group: ', 'output_shape_g: ', output_shape_g,
              'dx_g: ', img_shape_g, 'filter_g: ', filter_shape_g)
        print("general input tiling", tiling)
        print("general dx fusion tag:", deconv_res.op.tag)
        print("general dx kernel_name:", _kernel_name)

    def _tiling_check_none():
        if (
            (tiling.get("AL1_shape") is None)
            or (tiling.get("BL1_shape") is None)
            or (tiling.get("CUB_matrix") is None)
        ):
            _raise_dx_general_err("AL1_shape/BL1_shape/CUB_matrix " "can't be None.")
        if (
            (tiling.get("AL0_matrix") is None)
            or (tiling.get("BL0_matrix") is None)
            or (tiling.get("CL0_matrix") is None)
        ):
            _raise_dx_general_err("AL0_matrix/BL0_matrix/CL0_matrix " "can't be None.")

    def _tiling_l0_process():
        if al0_tiling_ma == a_col_ma and al0_tiling_ka == a_col_ka and a_col_batch == 1:
            tiling["AL0_matrix"] = []

        if tiling.get("BL0_matrix") != []:
            (
                bl0_tiling_kb,
                bl0_tiling_nb,
                bl0_tiling_n0,
                bl0_tiling_k0,
                _,
                _
            ) = tiling.get("BL0_matrix")
        else:
            (
                _,
                bl0_tiling_kb,
                bl0_tiling_nb,
                bl0_tiling_n0,
                bl0_tiling_k0
            ) = list(i.value for i in b_col.shape)
            bl0_tiling_kb = bl0_tiling_kb
        return bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0

    def _tiling_l1_process():
        if tiling.get("AL1_shape") != []:
            al1_tilling_k, al1_tilling_m, _, _ = tiling.get("AL1_shape")
            if (al1_tilling_k == kernel_h * kernel_w * cou1_g * al1_co0 and \
               al1_tilling_m == _ceil(c_l0c_hw,
                                      cce_params.CUBE_MKN[c_col.dtype]["mac"][0] * cl0_tiling_mc)
            ):
                tiling["AL1_shape"] = []
        else:
            # batch and group is 1, other axes full load
            al1_tilling_k = kernel_h * kernel_w * cou1_g * al1_co0
            al1_tilling_m = _ceil(c_l0c_hw,
                                  cce_params.CUBE_MKN[c_col.dtype]["mac"][0] * cl0_tiling_mc)

        if tiling.get("BL1_shape") != []:
            bl1_tilling_k, bl1_tilling_n, _, _ = tiling.get("BL1_shape")
        else:
            if w_trans_flag:
                # [G*Cout1*Hk*Wk, cin1, cin0, cout0]: bl1_co1, bl1_k1, _, bl1_co0
                bl1_tilling_k = bl1_co0 * bl1_co1 // g_after
                bl1_tilling_n = bl1_k1 // cl0_tiling_nc
            else:
                # [G*Cin1*Hk*Wk, cou1, cou0, cin0]: bl1_k1, bl1_co1,bl1_co0,_
                bl1_tilling_k = kernel_h * kernel_w * bl1_co0 * bl1_co1
                bl1_tilling_n = bl1_k1 // (kernel_h * kernel_w * cl0_tiling_nc * g_after)
        return al1_tilling_k, al1_tilling_m, bl1_tilling_k, bl1_tilling_n

    # check tiling
    def _tiling_check_equal():
        if tiling.get("BL0_matrix") != []:
            if al0_tiling_ka != bl0_tiling_kb:
                _raise_dx_general_err("ka != kb.")
            if bl0_tiling_nb != cl0_tiling_nc:
                _raise_dx_general_err("nb != nc.")

        if al0_tiling_ma != cl0_tiling_mc:
            _raise_dx_general_err("ma != mc.")

    def _tiling_check_factor():
        if (kernel_w * kernel_h * cou1_g) % al0_tiling_ka != 0:
            _raise_dx_general_err("Co1*Hk*Wk % ka != 0")
        if al1_tilling_k % al0_tiling_ka != 0:
            _raise_dx_general_err("k_AL1 % ka != 0.")

        if tiling.get("BL1_shape") != [] and tiling.get("BL0_matrix") != []:
            if bl1_tilling_k % bl0_tiling_kb != 0:
                _raise_dx_general_err("k_BL1 % kb != 0.")

        if (cl0_tiling_nc % cub_tiling_nc_factor != 0) and (not cube_vector_split):
            _raise_dx_general_err("nc % nc_factor != 0.")

        if tiling.get("BL1_shape") != []:
            if al1_tilling_k > bl1_tilling_k and al1_tilling_k % bl1_tilling_k != 0:
                _raise_dx_general_err("k_AL1 > k_BL1 but k_AL1 % k_BL1 != 0.")
            if bl1_tilling_k > al1_tilling_k and bl1_tilling_k % al1_tilling_k != 0:
                _raise_dx_general_err("k_BL1 > k_AL1 but k_BL1 % k_AL1 != 0.")

    def _tiling_check_load():
        if stride_h == 1 and stride_w == 1:
            if tiling.get("AUB_shape") is not None:
                _raise_dx_general_err("stride = 1 but AUB_shape is not None.")
        elif tiling.get("AUB_shape") is None:
            _raise_dx_general_err("stride > 1 but AUB_shape is None.")

        if tiling.get("BL0_matrix") == [] and tiling.get("BL1_shape") != []:
            _raise_dx_general_err("BL0 full load but BL1 not!")

    def _tiling_check_pbuffer():
        if stride_h > 1 or stride_w > 1:
            if aub_pbuffer not in (1, 2):
                _raise_dx_general_err("value of AUB_pbuffer can only be 1 or 2")

        if al1_pbuffer not in (1, 2):
            _raise_dx_general_err("value of AL1_pbuffer can only be 1 or 2")

        if bl1_pbuffer not in (1, 2):
            _raise_dx_general_err("value of BL1_pbuffer can only be 1 or 2")

        if al0_pbuffer not in (1, 2):
            _raise_dx_general_err("value of AL0_pbuffer can only be 1 or 2")

        if bl0_pbuffer not in (1, 2):
            _raise_dx_general_err("value of BL0_pbuffer can only be 1 or 2")

        if l0c_pbuffer not in (1, 2):
            _raise_dx_general_err("value of L0C_pbuffer can only be 1 or 2")

        if cub_pbuffer not in (1, 2):
            _raise_dx_general_err("value of CUB_pbuffer can only be 1 or 2")

    def _full_load_flag():
        """
        check whether al1's m bl1's k and n is fully loaded

        Returns
        -------
        true for full loaded
        """
        # Check whether al1_tilling_m is fully loaded
        al1_m_full_load = False
        if al1_tilling_m == c_l0c_hw // (
            cce_params.CUBE_MKN[c_col.dtype]["mac"][0] * cl0_tiling_mc
        ):
            al1_m_full_load = True

        # Check whether bl1_tilling_k is fully loaded
        bl1_k_full_load = False
        if w_trans_flag and (bl1_tilling_k == bl1_co0 * bl1_co1):
            bl1_k_full_load = True
        elif bl1_tilling_k == kernel_h * kernel_w * bl1_co0 * bl1_co1:
            bl1_k_full_load = True

        bl1_n_full_load = False
        if w_trans_flag and (bl1_tilling_n == bl1_k1 // cl0_tiling_nc):
            bl1_n_full_load = True
        elif bl1_tilling_n == bl1_k1 // (kernel_h * kernel_w * cl0_tiling_nc):
            bl1_n_full_load = True

        return al1_m_full_load, bl1_k_full_load, bl1_n_full_load

    def _check_overload_dy():
        """
        check whether dy is overload
        Use the following conditions to judge:
        1. if multi core in n axis, dy will overload
        2. if split al1's and bl1's k, and al1_k < bl1_k
        3. if stride < kernel and spilt al1's m
        4. if spilt n axis
        Returns
        -------
        true for overload, false for not overload
        """

        al1_m_full_load, bl1_k_full_load, bl1_n_full_load = _full_load_flag()
        _, block_dim_n, _, _ = tiling.get("block_dim")

        if block_dim_n > 1:
            return True

        if (stride_h < kernel_h or stride_w < kernel_w) and (not al1_m_full_load):
            return True

        if ((not bl1_k_full_load) and (al1_tilling_k < bl1_tilling_k)) or (
            not bl1_n_full_load
        ):
            return True

        return False

    def _set_overload_flag(param, overload_flag, overload_axis):
        """
        set flag on the first axis
        """
        cache_read_mode = 0 if overload_flag else 1
        param.pragma(overload_axis, "json_info_cache_read_mode", cache_read_mode)

    _tiling_check_none()
    (
        cub_tiling_nc_factor,
        cub_tiling_mc_factor,
        cub_tiling_m0,
        cub_tiling_n0,
        _,
        _
    ) = tiling.get("CUB_matrix")
    cl0_tiling_nc, cl0_tiling_mc, cl0_tiling_m0, cl0_tiling_n0, _, _ = tiling.get(
        "CL0_matrix"
    )
    al0_tiling_ma, al0_tiling_ka, al0_tiling_m0, al0_tiling_k0, _, _ = tiling.get(
        "AL0_matrix"
    )

    batch_dim, n_dim, m_dim, group_dim = tiling.get("block_dim")
    aub_pbuffer = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
    al1_pbuffer = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
    bl1_pbuffer = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
    al0_pbuffer = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
    bl0_pbuffer = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
    l0c_pbuffer = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
    cub_pbuffer = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")

    _, al1_co1, _, _, al1_co0 = shape_to_list(a_l1.shape)
    _, _, _, c_l0c_hw, _ = shape_to_list(c_col.shape)
    if w_trans_flag:
        # G*Cout1*Hk*Wk, Cin1, Cin0, Cout0
        bl1_co1, bl1_k1, _, bl1_co0 = list(i.value for i in b_l1.shape)
    else:
        # G*Cin1*Hk*Wk, Cout1, Cout0, Cin0
        bl1_k1, bl1_co1, bl1_co0, _ = shape_to_list(b_l1.shape)
    c_col_k1, c_col_k0 = list(ax.dom.extent.value for ax in c_col.op.reduce_axis)
    a_col_shape = shape_to_list(a_col.shape)
    a_col_g, a_col_batch, a_col_ma, a_col_ka, _, _ = a_col_shape

    bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0 = _tiling_l0_process()
    al1_tilling_k, al1_tilling_m, bl1_tilling_k, bl1_tilling_n = _tiling_l1_process()
    _tiling_check_equal()
    _tiling_check_factor()
    _tiling_check_load()
    _tiling_check_pbuffer()

    def _cub_process():  # pylint: disable=R0912,R0915
        if cube_vector_split:
            return
        def _attach_cub():
            # c_ub will attach on deconv_res in dynamic shape by default
            if not dynamic_mode:
                status = Compare.compare(affine_cub[1:], op_shape)
            else:
                status = Compare.LESS_EQ

            if status == Compare.EQUAL:
                pass
            elif status == Compare.LESS_EQ:
                sch_agent.attach_at(c_ub, c_ddr, affine_shape=affine_cub)
            else:
                _raise_dx_general_err("c_ub attach error.")

        def fusion_cub_process():
            if deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool":
                if "elewise_binary_add" in deconv_res.op.input_tensors[1].op.tag:
                    sch_agent[c_ub].reused_by(c_ub_cut, vadd_res, c_ub_drelu)
                    sch_agent.same_attach(vadd_res, c_ub)
                    sch_agent.same_attach(vadd_tensor_ub, c_ub)
                    sch_agent.same_attach(mask_ub, c_ub)
                else:
                    sch_agent[c_ub].reused_by(c_ub_cut, c_ub_drelu)
                    if out_of_order is None:
                        sch_agent.same_attach(mask_ub, c_ub)

                sch_agent.same_attach(c_ub_cut, c_ub)
                sch_agent.same_attach(c_ub_drelu, c_ub)
            elif deconv_res.op.tag == "requant_remove_pad":
                sch_agent.same_attach(tensor_map["deq"], c_ub)
            elif tensor_attr.get("quant_fuse"):
                sch_agent.same_attach(tensor_map["deq"], c_ub)
                for ub_tensor in tensor_map["ub_list"]:
                    sch_agent.same_attach(ub_tensor, c_ub)
                for input_tensor_mem in tensor_map["input_tensor"]:
                    sch_agent.same_attach(input_tensor_mem[0], c_ub)
                    sch_agent.same_attach(input_tensor_mem[1], c_ub)
            elif "elewise" in deconv_res.op.tag:
                scope, unit = sch_agent[c_ddr].get_active_scope_and_unit()
                _, _, _, ax_hw, _ = scope
                _, _, _, len_axis, _ = unit
                len_align = tvm.min(len_axis, c_ub.shape[2] - ax_hw * len_axis) * 16
                for ub_tensor in tensor_map["ub_list"]:
                    sch_agent.same_attach(ub_tensor, c_ub)
                    sch[ub_tensor].bind_buffer(ub_tensor.op.axis[1], len_align, 0)
                    sch[c_ub].reused_by(ub_tensor)
                for input_tensor_mem in tensor_map["input_tensor_list"]:
                    sch_agent.same_attach(input_tensor_mem[0], c_ub)
                    sch_agent.same_attach(input_tensor_mem[1], c_ub)
                    sch[input_tensor_mem[0]].bind_buffer(
                        input_tensor_mem[0], len_align, 0
                    )
                    sch[c_ub].reused_by(input_tensor_mem[0])
            elif deconv_res.op.tag == "elewise_binary_add":
                sch_agent[c_ub].reused_by(c_ub_cut, c_ub_vadd)
                sch_agent.same_attach(vadd_tensor_ub, c_ub)
                sch_agent.same_attach(c_ub_cut, c_ub)
                sch_agent.same_attach(c_ub_vadd, c_ub)

        c_ub_nc_factor = cub_tiling_nc_factor

        if deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool":
            _, _, dx_h, dx_w, _ = output_shape
            dx_hw = dx_h * dx_w
            tiling_m_axis = cl0_tiling_mc * cl0_tiling_m0
            if (dx_hw - dx_hw // tiling_m_axis * tiling_m_axis) % 16 != 0:
                c_ub_nc_factor = 1

        # dx_batch, dx_cin1, dx_m, dx_cin0
        op_shape = shape_to_list(c_ub.shape)
        if n0_32_flag is not None:
            affine_cub = (
                1,
                1,
                int(c_ub_nc_factor / 2),
                cub_tiling_mc_factor * cub_tiling_m0,
                cub_tiling_n0 * 2
            )
            if deconv_res.op.tag == "quant":
                op_shape[1] = (op_shape[1] + 1) // 2
                op_shape[3] = op_shape[3] * 2
        else:
            affine_cub = (
                1,
                1,
                c_ub_nc_factor,
                cub_tiling_mc_factor * cub_tiling_m0,
                cub_tiling_n0
            )

        _attach_cub()

        sch[c_ub].buffer_align(
            (1, 1),
            (1, 1),
            (1, cce_params.CUBE_MKN[c_ub.dtype]["mac"][0]),
            (1, cce_params.CUBE_MKN[c_ub.dtype]["mac"][2])
        )
        if bias_add_vector is not None:
            sch_agent[c_ub].reused_by(bias_add_vector)
            sch[bias_add_vector].buffer_align(
                (1, 1),
                (1, 1),
                (1, cce_params.CUBE_MKN[c_ub.dtype]["mac"][0]),
                (1, cce_params.CUBE_MKN[c_ub.dtype]["mac"][2])
            )

        fusion_cub_process()

        return affine_cub

    def _cl0_process(affine_cub):
        if n0_32_flag is not None:
            affine_l0c = (1,
                          1,
                          int(cl0_tiling_nc / 2),
                          cl0_tiling_mc * cl0_tiling_m0,
                          cl0_tiling_n0 * 2)
        else:
            affine_l0c = 1, 1, cl0_tiling_nc, cl0_tiling_mc * cl0_tiling_m0, cl0_tiling_n0

        if cube_vector_split:
            sch_agent.attach_at(c_col, c_ddr, affine_shape=affine_l0c)
        else:
            c_col_shape = shape_to_list(c_col.shape)

            # c_col will attach on c_ub or c_ddr in dynamic shape by default
            if not dynamic_mode:
                status_ori = Compare.compare(affine_l0c, c_col_shape)
            else:
                status_ori = Compare.LESS_EQ
            status = Compare.compare(affine_l0c, affine_cub)

            if status_ori == Compare.EQUAL:
                pass
            elif status == Compare.EQUAL:
                sch_agent.same_attach(c_col, c_ub)
            elif status == Compare.LESS_EQ:
                sch_agent.attach_at(c_col, c_ub, affine_shape=affine_l0c)
            elif status == Compare.GREATE_EQ:
                sch_agent.attach_at(c_col, c_ddr, affine_shape=affine_l0c)
            else:
                _raise_dx_general_err("c_col attach error.")
        if deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool" \
                and "conv2d_backprop_input" in deconv_res.op.input_tensors[1].op.tag \
                and out_of_order is not None:
            align_buffer = 0
            if (dx_h * dx_w) % cce_params.CUBE_MKN[c_col.dtype]["mac"][0] != 0:
                align_buffer = reduce(lambda x, y: x * y, tiling["CUB_matrix"][1:4])
                sch[mask_ub].bind_buffer(mask_ub.op.axis[1], align_buffer, 0)
            if DEBUG_MODE:
                print("mask_ub same_attach c_col, align_buffer:", align_buffer)
            sch_agent.same_attach(mask_ub, c_col)
        sch[c_col].buffer_align(
            (1, 1),
            (1, 1),
            (1, 1),
            (1, cce_params.CUBE_MKN[c_col.dtype]["mac"][0]),
            (1, cce_params.CUBE_MKN[c_col.dtype]["mac"][2]),
            (1, 1),
            (1, cce_params.CUBE_MKN[c_col.dtype]["mac"][1])
        )

    def _l0a_process():
        l0a2l0c_affine_shape = (
            1,
            1,
            None,
            al0_tiling_ma * al0_tiling_m0,
            cl0_tiling_n0,
            al0_tiling_ka,
            al0_tiling_k0
        )
        tiling_ori_l0a = (
            1,
            1,
            al0_tiling_ma,
            al0_tiling_ka,
            al0_tiling_m0,
            al0_tiling_k0
        )

        # a_col will attach on c_col, c_ub or c_ddr in dynamic shape
        if not dynamic_mode:
            status_ori = Compare.compare(tiling_ori_l0a, a_col_shape)
        else:
            status_ori = Compare.LESS_EQ
        status = Compare.compare(
            [al0_tiling_ma, al0_tiling_m0, al0_tiling_ka, al0_tiling_k0],
            [cl0_tiling_mc, cl0_tiling_m0, c_col_k1, c_col_k0]
        )

        if status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(a_col, c_col)
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(a_col, c_col, affine_shape=l0a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l0a2out_affine_shape = [
                1,
                1,
                None,
                al0_tiling_ma * al0_tiling_m0,
                cl0_tiling_n0
            ]
            sch_agent.attach_at(a_col, c_ddr, affine_shape=l0a2out_affine_shape)
        else:
            _raise_dx_general_err("l0a attach error.")

    def _l0b_process():
        neg_src_stride = True
        if tiling.get("BL0_matrix") != []:
            l0b2l0c_affine_shape = (
                1,
                None,
                bl0_tiling_nb,
                cl0_tiling_mc * cl0_tiling_m0,
                bl0_tiling_n0,
                bl0_tiling_kb,
                bl0_tiling_k0
            )
            tiling_ori_l0b = (
                1,
                bl0_tiling_kb,
                bl0_tiling_nb,
                bl0_tiling_n0,
                bl0_tiling_k0
            )
            b_col_shape = shape_to_list(b_col.shape)
            status_ori = Compare.compare(tiling_ori_l0b, b_col_shape)
            status = Compare.compare(
                [bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_kb, bl0_tiling_k0],
                [cl0_tiling_nc, cl0_tiling_n0, c_col_k1, c_col_k0]
            )
            neg_src_stride = False
            if status_ori == Compare.EQUAL:
                neg_src_stride = True
            elif status == Compare.EQUAL:
                sch_agent.same_attach(b_col, c_col)
            elif status == Compare.LESS_EQ:
                sch_agent.attach_at(b_col, c_col, affine_shape=l0b2l0c_affine_shape)
            elif status == Compare.GREATE_EQ:
                l0b2out_affine_shape = [1, 1, bl0_tiling_nb, cl0_tiling_m0, bl0_tiling_n0]
                sch_agent.attach_at(b_col, c_ddr, affine_shape=l0b2out_affine_shape)
            else:
                _raise_dx_general_err("l0b attach error.")
        return neg_src_stride

    def _al1_process():
        l1_ma = al1_tilling_m * al0_tiling_ma
        l1_ka = al1_tilling_k // al0_tiling_k0
        l1a2l0c_affine_shape = (
            1,
            1,
            None,
            l1_ma * al0_tiling_m0,
            bl0_tiling_n0,
            l1_ka,
            al0_tiling_k0
        )

        if dynamic_mode == "dynamic_hw" and tiling.get("AL1_shape") == []:
            status = Compare.GREATE_EQ
        else:
            status = Compare.compare(
                [l1_ma, al0_tiling_m0, l1_ka, al0_tiling_k0],
                [cl0_tiling_mc, cl0_tiling_m0, c_col_k1, c_col_k0]
            )

        if tiling.get("AL1_shape") == [] and tiling.get("AL0_matrix") == []:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(a_l1, c_col)
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(a_l1, c_col, affine_shape=l1a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l1a2out_affine_shape = [1, 1, None, l1_ma * al0_tiling_m0, cl0_tiling_n0]
            sch_agent.attach_at(a_l1, c_ddr, affine_shape=l1a2out_affine_shape)
        else:
            _raise_dx_general_err("a_l1 atach error.")

        if is_conv1d_situation:
            w_align = 1
        else:
            w_align = dx_w

        if not dynamic_mode:
            sch_agent.same_attach(a_col_before, a_l1)
            sch[a_col_before].buffer_align(
                (1, 1),
                (w_align, w_align),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, cce_params.CUBE_MKN[a_col_before.dtype]["mac"][1])
            )

    def _bl1_process():
        if tiling.get("BL1_shape") != []:
            l1_nb = bl1_tilling_n * bl0_tiling_nb
            _, _k0, _n0 = cce_params.CUBE_MKN[b_l1.dtype]["mac"]
            bl1_shape = shape_to_list(b_l1.shape)
            bl0_tiling_n0_temp = bl0_tiling_n0
            cl0_tiling_nc_temp = cl0_tiling_nc
            cl0_tiling_n0_temp = cl0_tiling_n0
            if n0_32_flag is not None:
                l1_nb //= 2
                _n0 *= 2
                bl1_shape[1] = _ceil(bl1_shape[1], 2)
                bl1_shape[2] *= 2
                cl0_tiling_n0_temp *= 2
                bl0_tiling_n0_temp *= 2
                cl0_tiling_nc_temp //= 2
            l1_kb = bl1_tilling_k // _k0
            l1b2l0c_affine_shape = (
                1,
                None,
                l1_nb,
                cl0_tiling_m0,
                bl0_tiling_n0_temp,
                l1_kb,
                bl0_tiling_k0
            )
            if w_trans_flag:
                tiling_ori_bl1 = bl1_tilling_k // _k0, l1_nb, _n0, _k0
            else:
                tiling_ori_bl1 = (
                    l1_nb * kernel_h * kernel_w,
                    bl1_tilling_k // (kernel_h * kernel_w * 16),
                    16,
                    16
                )

            status_ori = Compare.compare(tiling_ori_bl1, bl1_shape)
            status = Compare.compare(
                [l1_nb, bl0_tiling_n0_temp, l1_kb, bl0_tiling_k0],
                [cl0_tiling_nc_temp, cl0_tiling_n0_temp, c_col_k1, c_col_k0]
            )
            if status_ori == Compare.EQUAL:
                # bl1 full load but tiling.get("BL1_shape") is not []
                pass
            elif status == Compare.EQUAL:
                sch_agent.same_attach(b_l1, c_col)
            elif status == Compare.LESS_EQ:
                sch_agent.attach_at(b_l1, c_col, affine_shape=l1b2l0c_affine_shape)
            elif status == Compare.GREATE_EQ:
                l1b2out_affine_shape = [1, None, l1_nb, cl0_tiling_m0, bl0_tiling_n0]
                sch_agent.attach_at(b_l1, c_ddr, affine_shape=l1b2out_affine_shape)
            else:
                _raise_dx_general_err("b_l1 attach error.")

    def _aub_process():
        if stride_h > 1 or stride_w > 1:
            aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
            _, _, _, filling_w, _ = shape_to_list(a_filling.shape)
            if aub_tiling_m == 0:
                sch_agent.same_attach(a_filling, a_l1)
            else:
                if is_conv1d_situation:
                    aub_h = 1
                    aub_w = aub_tiling_m
                else:
                    aub_h = aub_tiling_m
                    aub_w = filling_w
                ub_shape = [
                    1,
                    aub_tiling_k // (kernel_h * kernel_w * 16),
                    aub_h,
                    aub_w + kernel_w - 1,
                    al1_co0
                ]
                sch_agent.attach_at(a_filling, a_l1, ub_shape)
            if not dynamic_mode:
                sch_agent.same_attach(a_zero, a_filling)
            sch_agent.same_attach(a_ub, a_filling)

    def _attach_bias():
        if c_add_bias is not None:
            sch_agent.same_attach(c_add_bias, c_col)
            sch_agent.same_attach(bias_l0c, c_col)
            sch_agent.same_attach(bias_ub_brc, c_col)

        split_bias_flag = tiling.get("CUB_channel_wise_flag")
        if bias_ub is not None and split_bias_flag:
            sch_agent.same_attach(bias_ub, c_ub)
        if bias_add_vector is not None:
            sch_agent.same_attach(bias_add_vector, c_ub)

    def _do_l1andub_process():
        if al1_tilling_k < bl1_tilling_k:
            _al1_process()
            _bl1_process()
        else:
            _bl1_process()
            _al1_process()

        if stride_h > 1 or stride_w > 1:
            _aub_process()

    def _double_buffer():
        def _fusion_double_buffer():
            if deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool":
                sch[c_ub_cut].double_buffer()
                sch[c_ub_drelu].double_buffer()
                if "elewise_binary_add" in deconv_res.op.input_tensors[1].op.tag:
                    sch[vadd_res].double_buffer()
            elif "elewise" in deconv_res.op.tag:
                for ub_tensor in tensor_map["ub_list"]:
                    sch[ub_tensor].double_buffer()
                for input_tensor_mem in tensor_map["input_tensor_list"]:
                    sch[input_tensor_mem[1]].double_buffer()

        if stride_h > 1 or stride_w > 1:
            if tiling.get("manual_pingpong_buffer").get("AUB_pbuffer") == 2:
                sch[a_ub].double_buffer()
                sch[a_filling].double_buffer()
                sch[a_zero].double_buffer()
                if dynamic_mode is not None:
                    sch[a_one].double_buffer()
                    sch[vn_tensor].double_buffer()

        if dynamic_mode and not tiling.get("AL1_shape"):
            a_l1_db_flag = 1
        else:
            a_l1_db_flag = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")

        if a_l1_db_flag == 2:
            sch[a_l1].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") == 2:
            sch[b_l1].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == 2:
            sch[a_col].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") == 2:
            sch[b_col].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2:
            sch[c_col].double_buffer()
            if c_add_bias is not None:
                sch[c_add_bias].double_buffer()
                sch[bias_l0c].double_buffer()
                sch[bias_l0c].preload()
                sch[bias_ub_brc].double_buffer()
                sch[bias_ub_brc].preload()

        if tiling.get("manual_pingpong_buffer").get("CUB_pbuffer") == 2:
            sch[c_ub].double_buffer()
            if bias_add_vector is not None:
                sch[bias_add_vector].double_buffer()
            _fusion_double_buffer()

    def _emit_fusion_insn():
        deq_axis_mode = {False: (0, "scalar"), True: (2, "vector")}

        def _emit_requant_fusion_insn():
            sch_agent[tensor_map["deq"]].emit_insn(
                tensor_map["deq"].op.axis[0], "dma_copy"
            )
            c_ub_reform = tensor_map["c_ub"]
            reform_outer, reform_inner = sch[c_ub_reform].split(
                c_ub_reform.op.axis[3], nparts=2
            )
            sch[c_ub_reform].reorder(
                reform_outer,
                c_ub_reform.op.axis[0],
                c_ub_reform.op.axis[1],
                c_ub_reform.op.axis[2],
                reform_inner
            )
            sch[c_ub_reform].emit_insn(c_ub_reform.op.axis[2], "dma_copy")

        def _emit_quant_fusion_insn():
            sch_agent[tensor_map["deq"]].emit_insn(
                tensor_map["deq"].op.axis[0], "dma_copy"
            )
            deq_axis = deq_axis_mode[tensor_attr.get("deq_vector")][0]
            if cce_conf.is_v200_version_new():
                sch[tensor_map["c_ub"]].emit_insn(
                    tensor_map["c_ub"].op.axis[deq_axis], "dma_copy"
                )
            else:
                deq_mode = deq_axis_mode[tensor_attr.get("deq_vector")][1]
                sch_agent[c_ub].pragma(
                    sch_agent[c_ub].op.axis[deq_axis], "deq_scale", deq_mode
                )
            for ub_tensor in tensor_map["ub_list"]:
                if ub_tensor.op.name == "input_ub":
                    sch_agent[ub_tensor].emit_insn(
                        sch_agent[ub_tensor].op.axis[0], "dma_padding"
                    )
                elif "reform" in ub_tensor.op.name:
                    ndim = len(sch[ub_tensor].op.axis)
                    coo, _ = sch[ub_tensor].split(
                        sch[ub_tensor].op.axis[ndim - 1],
                        cce_params.CUBE_MKN["float16"]["mac"][1]
                    )
                    axis_list = sch[ub_tensor].op.axis[0 : ndim - 1]
                    sch[ub_tensor].reorder(coo, *axis_list)
                    sch[ub_tensor].emit_insn(sch[ub_tensor].op.axis[2], "vector_auto")
                elif ub_tensor.op.name == "cast_i8_ub":
                    if (
                        cce_conf.is_v200_version_new()
                        or cce_conf.CceProductParams().is_lhisi_version()
                    ):
                        conv_mode = "vector_conv_{}".format(
                            tensor_attr.get("q_mode").lower()
                        )
                    else:
                        conv_mode = "vector_conv"
                    sch_agent[ub_tensor].emit_insn(
                        sch_agent[ub_tensor].op.axis[0], conv_mode
                    )
                else:
                    sch_agent[ub_tensor].emit_insn(
                        sch_agent[ub_tensor].op.axis[0], "vector_auto"
                    )
            for input_tensor_mem in tensor_map["input_tensor"]:
                sch_agent[input_tensor_mem[0]].emit_insn(
                    sch_agent[input_tensor_mem[0]].op.axis[0], "dma_copy"
                )
                sch_agent[input_tensor_mem[1]].emit_insn(
                    sch_agent[input_tensor_mem[1]].op.axis[0], "vector_auto"
                )

        if deconv_res.op.tag == "requant_remove_pad":
            _emit_requant_fusion_insn()
        elif tensor_attr.get("quant_fuse"):
            _emit_quant_fusion_insn()
        else:
            if not cube_vector_split:
                sch_agent[c_ub].emit_insn(sch_agent[c_ub].op.axis[0], "dma_copy")

        if deconv_res.op.tag == "emit_insn_elewise_multiple_sel|bool":
            sch[mask_ub].emit_insn(mask_ub.op.axis[0], "dma_copy")

            if "elewise_binary_add" in deconv_res.op.input_tensors[1].op.tag:
                sch[vadd_tensor_ub].emit_insn(vadd_tensor_ub.op.axis[0], "dma_copy")
                sch[vadd_res].emit_insn(vadd_res.op.axis[0], "vector_add")
            sch[c_ub_drelu].emit_insn(c_ub_drelu.op.axis[0], "vector_selects_bool")

            sch[c_ub_cut].emit_insn(c_ub_cut.op.axis[0], "phony_insn")
        elif "elewise" in deconv_res.op.tag:
            for ub_tensor in tensor_map["ub_list"]:
                sch_agent[ub_tensor].emit_insn(
                    sch_agent[ub_tensor].op.axis[0], "vector_auto"
                )
            for input_tensor_mem in tensor_map["input_tensor_list"]:
                sch_agent[input_tensor_mem[0]].emit_insn(
                    sch_agent[input_tensor_mem[0]].op.axis[0], "dma_copy"
                )
                sch_agent[input_tensor_mem[1]].emit_insn(
                    sch_agent[input_tensor_mem[1]].op.axis[0], "vector_auto"
                )

    def _emit_l1fusion_insn(setfmatrix_dict):
        if a_l1_full is not None:
            sch[a_l1_full].emit_insn(sch[a_l1_full].op.axis[0], "dma_copy")
        if stride_h == 1 and stride_w == 1:
            if l1_fusion_type != -1 and input_mem[0] == 1:
                sch_agent[a_l1].emit_insn(sch_agent[a_l1].op.axis[0], "phony_insn")
            else:
                sch_agent[a_l1].emit_insn(sch_agent[a_l1].op.axis[0], "dma_copy")
                if l1_fusion_type != -1:
                    sch_agent[a_l1].pragma(a_l1.op.axis[0], "jump_data", 1)
        else:
            sch_agent[a_ub].emit_insn(sch_agent[a_ub].op.axis[0], "dma_copy")
            afill_n, afill_c, afill_h, afill_w, _ = sch_agent[
                a_filling
            ].get_active_scopes()
            afill_w_out, afill_w_inner = sch_agent[a_filling].split(
                afill_w, factor=stride_w
            )
            afill_h_out, afill_h_inner = sch_agent[a_filling].split(
                afill_h, factor=stride_h
            )
            sch_agent[a_filling].reorder(
                afill_h_inner, afill_w_inner, afill_n, afill_c, afill_h_out, afill_w_out
            )
            sch_agent[a_filling].unroll(afill_h_inner)
            sch_agent[a_filling].unroll(afill_w_inner)
            sch_agent[a_filling].reused_by(a_zero)
            sch_agent[a_zero].emit_insn(sch_agent[a_zero].op.axis[0], "vector_dup")
            if w_trans_flag:
                sch_agent[a_filling].emit_insn(afill_n, "dma_copy")
            else:
                sch_agent[a_filling].emit_insn(afill_n, "vector_muls")
            al1_insn, _, _ = sch_agent[a_l1].nlast_scopes(3)
            sch_agent[a_l1].emit_insn(al1_insn, "dma_copy")
        setfmatrix_dict["conv_fm_c"] = a_l1.shape[1] * a_l1.shape[4]
        setfmatrix_dict["conv_fm_h"] = valid_shape[2] if valid_shape else a_l1.shape[2]
        setfmatrix_dict["conv_fm_w"] = a_l1.shape[3]
        offset_flag = (
            (input_mem[0] == 1)
            and (l1_fusion_type != -1)
            and (stride_h == 1)
            and (stride_w == 1)
            and valid_shape
        )
        if offset_flag:
            setfmatrix_dict["conv_fm_offset_h"] = slice_offset[2]
        sch_agent[a_col_before].emit_insn(
            a_col_before.op.axis[0], 'set_fmatrix', setfmatrix_dict)
        sch_agent[a_col].emit_insn(a_col.op.axis[1], 'im2col')

    def _dynamic_emit_insn():
        setfmatrix_dict = {
            "set_fmatrix": 1,
            "conv_kernel_h": kernel_h,
            "conv_kernel_w": kernel_w,
            "conv_padding_top": padu,
            "conv_padding_bottom": padd,
            "conv_padding_left": padl,
            "conv_padding_right": padr,
            "conv_stride_h": 1,
            "conv_stride_w": 1,
            "conv_fm_c": a_l1.shape[1] * a_l1.shape[4],
            "conv_fm_c1": a_l1.shape[1],
            "conv_fm_h": a_l1.shape[2],
            "conv_fm_w": a_l1.shape[3],
            "conv_fm_c0": a_l1.shape[4]
        }
        setfmatrix_dict_0 = {
            "set_fmatrix": 0,
            "conv_kernel_h": kernel_h,
            "conv_kernel_w": kernel_w,
            "conv_padding_top": padu,
            "conv_padding_bottom": padd,
            "conv_padding_left": padl,
            "conv_padding_right": padr,
            "conv_stride_h": 1,
            "conv_stride_w": 1,
            "conv_fm_c": a_l1.shape[1] * a_l1.shape[4],
            "conv_fm_c1": a_l1.shape[1],
            "conv_fm_h": a_l1.shape[2],
            "conv_fm_w": a_l1.shape[3],
            "conv_fm_c0": a_l1.shape[4]
        }
        if stride_h == 1 and stride_w == 1:
            sch[a_l1].emit_insn(a_l1.op.axis[0], "dma_copy", setfmatrix_dict)
        else:
            afill_n, afill_c, afill_h, afill_w, _ = sch_agent[
                a_filling
            ].get_active_scopes()
            afill_w_outer, afill_w_inner = sch_agent[a_filling].split(
                afill_w, factor=stride_w
            )
            afill_h_outer, afill_h_inner = sch_agent[a_filling].split(
                afill_h, factor=stride_h
            )

            sch_agent[a_filling].reorder(
                afill_w_inner,
                afill_h_inner,
                afill_n,
                afill_c,
                afill_h_outer,
                afill_w_outer
            )
            sch_agent[a_filling].unroll(afill_h_inner)
            sch_agent[a_filling].unroll(afill_w_inner)

            sch_agent[a_zero].reused_by(a_one)
            sch_agent[a_one].reused_by(vn_tensor)

            sch_agent[a_zero].emit_insn(sch_agent[a_zero].op.axis[0], "vector_dup")
            sch_agent[a_one].emit_insn(sch_agent[a_one].op.axis[0], "vector_dup")
            sch[vn_tensor].emit_insn(vn_tensor.op.axis[0], "phony_insn")
            sch_agent[a_filling].emit_insn(afill_n, "vector_mul")
            sch_agent[a_ub].emit_insn(sch_agent[a_ub].op.axis[0], "dma_copy")
            c1_inner, _, _, _ = sch_agent[a_l1].nlast_scopes(4)
            sch_agent[a_l1].emit_insn(c1_inner, "dma_copy", setfmatrix_dict)
        sch[a_col].emit_insn(a_col.op.axis[1], 'im2col_v2', setfmatrix_dict_0)

    def _emit_insn():  # pylint: disable=R0914,R0915
        sch_agent[b_l1].emit_insn(sch_agent[b_l1].op.axis[0], "dma_copy")

        if bias_add_vector is not None:
            sch[bias_ub].emit_insn(sch[bias_ub].op.axis[0], "dma_copy")
            sch[bias_add_vector].emit_insn(
                sch[bias_add_vector].op.axis[0],
                "vector_auto"
            )

        if neg_src_stride and (
            cce_conf.CceProductParams().is_cloud_version()
            or cce_conf.is_v200_version_new()
            or cce_conf.CceProductParams().is_lhisi_version()
        ):
            _, b_col_inner = sch_agent[b_col].split(
                sch_agent[b_col].op.axis[1], factor=kernel_h * kernel_w
            )
            sch_agent[b_col].emit_insn(b_col_inner, "dma_copy")
        else:
            sch_agent[b_col].emit_insn(sch_agent[b_col].op.axis[2], "dma_copy")

        setfmatrix_dict = {
            "conv_kernel_h": kernel_h,
            "conv_kernel_w": kernel_w,
            "conv_padding_top": padu,
            "conv_padding_bottom": padd,
            "conv_padding_left": padl,
            "conv_padding_right": padr,
            "conv_stride_h": 1,
            "conv_stride_w": 1,
            "conv_dilation_h": dilation_h,
            "conv_dilation_w": dilation_w
        }

        if dynamic_mode:
            _dynamic_emit_insn()
        else:
            _emit_l1fusion_insn(setfmatrix_dict)

        scopes_intrins = sch_agent[c_col].intrin_scopes(7)
        scope_insn = scopes_intrins[1]
        inner_k_axis = sch_agent[c_col].get_relate_scope(
            c_col.op.reduce_axis[0], scope_insn
        )
        if inner_k_axis:
            mad_dict = {
                "mad_pattern": 2,
                "k_outer": sch_agent[c_col].get_relate_scope(
                    c_col.op.reduce_axis[0], scope_insn
                )
            }
        else:
            inner_g, inner_n, inner_co, inner_m, inner_co0, inner_k1, inner_k0 = scopes_intrins
            inner_ko, inner_ki = sch_agent[c_col].split(inner_k1, nparts=1)
            sch_agent[c_col].reorder(
                inner_ko, inner_g, inner_n, inner_co, inner_m, inner_co0, inner_ki, inner_k0)
            mad_dict = {"mad_pattern": 2, "k_outer": [inner_ko]}
        if bias_ub_brc is not None:
            sch[bias_l0c].reused_by(c_add_bias, c_col)
            sch[c_add_bias].emit_insn(c_add_bias.op.axis[0], 'phony_insn')
            sch[bias_l0c].split(bias_l0c.op.axis[3], BRC_STANDARD_BLOCK_SIZE)
            sch[bias_l0c].emit_insn(bias_l0c.op.axis[2], 'dma_copy')
            sch[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')
            sch[bias_ub_brc].emit_insn(bias_ub_brc.op.axis[0], 'vector_auto')
            mad_dict["init_bias"] = 1
        sch_agent[c_col].emit_insn(scope_insn, "mad", mad_dict)
        if tensor_attr.get("write_select"):
            sch[c_ddr.op.input_tensors[0]].compute_inline()
            align_length = int(c_ddr.op.attrs["HWC0"])
            sch[c_ddr].bind_buffer(c_ddr.op.axis[1], align_length, 0)
        sch[c_ddr].emit_insn(sch_agent[c_ddr].nlast_scopes(2)[0], "dma_copy")
        overload_flag = _check_overload_dy()
        _set_overload_flag(
            sch[c_ddr], overload_flag, sch_agent[c_ddr].nlast_scopes(3)[0]
        )
        _emit_fusion_insn()

    def _handle_dynamic_workspace(stride_w):
        def _get_al1_m_extent(al1_m):
            al1_h = tvm.select(
                        (tvm.floormod(al1_m, output_shape[3]) == 0).asnode(),
                        kernel_h + (al1_m // output_shape[3]) - 1,
                        kernel_h + (al1_m // output_shape[3]) + 1)
            al1_w = a_ddr.shape[3] * stride_w
            return al1_h * al1_w

        def _get_al1_bound():
            if len(tiling["AL1_shape"]) != 0:
                k_al1, multi_m_al1 = tiling["AL1_shape"][:2]
                al1_m = multi_m_al1 * cl0_tiling_mc * cl0_tiling_m0
                al1_c = k_al1 // kernel_h // kernel_w
                al1_bound = al1_c * _get_al1_m_extent(al1_m)
            elif dynamic_mode == "dynamic_batch":
                al1_m = _ceil(a_l1.shape[2] * a_l1.shape[3], cl0_tiling_m0) * cl0_tiling_m0
                al1_c = al1_co1 * al1_co0
                al1_bound = al1_c * al1_m
            else:
                al1_m = (
                    _ceil(
                        c_l0c_hw,
                        (cce_params.CUBE_MKN[c_col.dtype]["mac"][0] * cl0_tiling_mc)
                    )
                    * al0_tiling_ma
                    * al0_tiling_m0
                )
                al1_c = al1_co1 * al1_co0
                al1_bound = al1_c * _get_al1_m_extent(al1_m)

            return al1_bound

        sch[a_l1].set_storage_bound(_get_al1_bound())
        if dynamic_mode == "dynamic_hw":
            sch.set_var_range(a_ddr.shape[2], *var_range.get("dedy_h"))
            sch.set_var_range(a_ddr.shape[3], *var_range.get("dedy_w"))
            sch.set_var_range(output_shape[2], *var_range.get("dx_h"))
            sch.set_var_range(output_shape[3], *var_range.get("dx_w"))
        elif dynamic_mode == "dynamic_batch":
            sch.set_var_range(a_ddr.shape[0], *var_range.get("batch_n"))
            sch.set_var_range(output_shape[0], *var_range.get("batch_n"))
        sch.disable_allocate(cce_params.scope_cbuf)
        sch.disable_allocate(cce_params.scope_ca)
        sch.disable_allocate(cce_params.scope_cb)
        sch.disable_allocate(cce_params.scope_cc)
        sch.disable_allocate(cce_params.scope_ubuf)
        sch[a_l1].mem_unique()
        sch[a_col].mem_unique()
        sch[b_l1].mem_unique()
        sch[b_col].mem_unique()
        sch[c_col].mem_unique()
        sch[c_ub].mem_unique()
        if stride_h > 1 or stride_w > 1:
            sch[a_filling].mem_unique()
            sch[a_ub].mem_unique()

    def _handle_workspace():
        l1_tensor_map = {}
        if not fmap_l1_addr_flag:
            l1_tensor_map = None
        else:
            fmap = DeConvPattern.dedy
            if a_l1_full is not None:
                sch[a_l1_full].set_storage_bound(fmap_l1_valid_size)
                l1_tensor_map[fmap] = a_l1_full
            elif (
                l1_fusion_type != -1
                and input_mem[0] == 0
                and stride_w == 1
                and stride_h == 1
            ):
                sch[a_l1].set_storage_bound(fmap_l1_valid_size)
                l1_tensor_map[fmap] = a_l1
            else:
                l1_tensor_map = None
        L1CommonParam.l1_fusion_tensors_map = l1_tensor_map

    sch_agent = ScheduleAgent(sch)
    sch_agent[c_ddr].split_group(c_ddr.op.axis[1], nparts=g_after)

    affine_cub = _cub_process()
    _cl0_process(affine_cub)
    _l0a_process()
    neg_src_stride = _l0b_process()
    _do_l1andub_process()
    _attach_bias()

    def _bind_core():
        axs = sch_agent[c_ddr].get_active_scopes()
        ax_g, ax_ni, ax_ci, ax_hw, _ = axs
        ax_core = sch_agent[c_ddr].bind_core(
            [ax_g, ax_ni, ax_ci, ax_hw], [group_dim, batch_dim, n_dim, m_dim])
        ax_core_in = sch_agent[c_ddr].get_superkernel_axis_pragma()
        sch_agent.root_stage_at(c_ddr, ax_core)
        blocks = batch_dim * group_dim * n_dim * m_dim
        if blocks == batch_dim:
            sch[c_ddr].pragma(ax_core_in, "json_info_batchBindOnly")
        return ax_core

    def _conv1d_split_tile():
        """
        conv1d situation
        use buffer tile to split width
        """
        kernel_w_dilation = (kernel_w - 1) * dilation_w + 1
        m_axis_length_l1 = al1_tilling_m * al0_tiling_ma * al0_tiling_m0
        m_axis_origin = sch_agent[c_ddr].get_bindcore_m_axis()

        # compute the axis's block offset and part offset after split
        howo_mad_align = _align(howo_mad, al0_tiling_m0)
        howo_offset_on_block = _align(_ceil(howo_mad_align, m_dim), m_axis_length_l1)
        block_offset = ax_core % m_dim * howo_offset_on_block
        # M axis full load in Al1 or Not
        if c_l0c_hw == m_axis_length_l1:
            m_parts_offset = 0
        else:
            m_parts_offset = m_axis_origin * m_axis_length_l1
        # compute the final offset
        w_offset = block_offset + m_parts_offset - padl

        # compute the m_length of A tensor used in Al1 by the Al0 length
        # using the convolution rule
        # expand in UB , so load3d stride is 1
        load3d_stride = 1
        w_extend = (m_axis_length_l1 - 1) * load3d_stride + kernel_w_dilation
        sch[a_l1].buffer_tile(
            (None, None), (None, None), (None, None), (w_offset, w_extend), (None, None)
        )

    def _c_col_buffer_tile():
        axis_split_list, axis_unit, axis_offset = sch_agent[c_ddr].get_axis_split_list_and_extend(2)
        l0c_attach = sch_agent.apply_var(sch[c_col])
        if l0c_attach is not None:
            ddr_idx = list(sch[c_ddr].leaf_iter_vars).index(l0c_attach)
            ddr_var_list = sch[c_ddr].leaf_iter_vars[0:ddr_idx]
            for var in ddr_var_list[::-1]:
                if var in axis_split_list:
                    c1_idx = axis_split_list.index(var)
                    axis = axis_split_list[0:c1_idx + 1]
                    unit = axis_unit[0:c1_idx + 1]
                    offset = axis_offset[0:c1_idx + 1]
                    c_offset = 0
                    for idx in range(c1_idx + 1):
                        offset_idx = (offset[idx] * 2 if c_ddr.dtype == "int8" else offset[idx])
                        factor_len = (0 if unit[idx] == 1 else offset_idx)
                        c_offset = c_offset + axis[idx] * factor_len
                    sch[c_col].buffer_tile(
                    (None, None),
                    (None, None),
                    (c_offset, tiling["CL0_matrix"][0]),
                    (None, None),
                    (None, None),
                    (None, None),
                    (None, None),
                    )
                    if c_add_bias is not None:
                        sch[c_add_bias].buffer_tile(
                        (None, None),
                        (None, None),
                        (c_offset, tiling["CL0_matrix"][0]),
                        (None, None),
                        (None, None),)
                    break

    ax_core = _bind_core()
    if is_conv1d_situation:
        _conv1d_split_tile()
    _double_buffer()
    _emit_insn()

    def _full_load_bl1_bl0():
        """
        g dimension only loads 1 each time
        """
        if not tiling.get("BL1_shape") and g_after != 1:
            axs = sch_agent[c_ddr].get_active_scopes()
            ax_g, _, _, _, _ = axs
            _, bl1_at_inner = sch_agent[c_ddr].split(ax_g, factor=1)
            sch[b_l1].compute_at(sch[c_ddr], bl1_at_inner)
            if not tiling.get("BL0_matrix"):
                sch[b_col].compute_at(sch[c_ddr], bl1_at_inner)
    _full_load_bl1_bl0()
    
    sch_agent.apply()
    _c_col_buffer_tile()
    if dynamic_mode is not None:
        _handle_dynamic_workspace(stride_w)
    else:
        _handle_workspace()

    tiling.clear()
    return True
