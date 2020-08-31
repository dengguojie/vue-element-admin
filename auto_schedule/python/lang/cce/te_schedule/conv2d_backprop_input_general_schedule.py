"""
Copyright (C) 2016. Huawei Technologies Co., Ltd.

conv2d backprop input general schedule.
"""

#!/usr/bin/python
#-*- coding: UTF-8 -*-

from te import tvm
from te.platform import scope_ubuf
from te.platform import scope_ca
from te.platform import scope_cb
from te.platform import scope_cc
from te.platform import scope_cbuf
from te.lang.cce.boost_schedule_kit import ScheduleAgent
from te.lang.cce.boost_schedule_kit import Compare
from te.domain.tiling.tiling_query import tiling_query
from te.platform import CUBE_MKN
from te.platform import intrinsic_check_support
from te.utils.error_manager import error_manager_util as err_man
from topi.cce.util import is_cloud_version
from topi.cce.util import is_v200_version
from topi.cce.util import is_lhisi_version

# default false
DEBUG_MODE = False  # pylint: disable=C0302
# Don't modify, used in log_util
DX_SUPPORT_TAG_LOG_PREFIX = '#Conv2DBackpropInput only support#'
# broadcast should be 16
BRC_STANDARD_BLOCK_SIZE = 16


def raise_dx_general_err(msg):
    """
    In op Conv2DBackpropInput_general, [%s] % (msg)
    msg for discribe the error info
    the error info only for Conv2DBackpropInput_general's developers
    """
    args_dict = {
        "errCode": "E60108",
        "reason": msg
    }
    msg = err_man.get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)

def print_ir_conv(process, sch):
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
        bounds = tvm.schedule.InferBound(sch)
        stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
        print(stmt)
        print(end)

def general_schedule(tensor, sch_list):  # pylint:disable=R0914,R0915
    """
    auto_schedule for cce AI-CORE.
    For now, only one convolution operation is supported.

    Parameters
    ----------
    res : tvm.tensor

    sch_list: use sch_list[0] to return conv schedule

    Returns
    -------
    True for sucess, False for no schedule
    """
    c_ddr = tensor
    sch = sch_list[0]
    tiling = None


    def _fetch_tensor_info():  # pylint:disable=R0914,R0912,R0915

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
            temp_tensor = c_ddr
            if c_ddr.op.tag == "quant":
                tensor_attr["n0_32_flag"] = True
                tensor_attr["q_mode"] = c_ddr.op.attrs["round_mode"].value
            while temp_tensor.op.input_tensors:
                if temp_tensor.op.tag in ("dequant_vector",
                                          "dequant1_vector"):
                    tensor_map["c_ub"] = temp_tensor
                    tensor_attr["deq_vector"] = True
                    tensor_attr["quant_fuse"] = True
                    tensor_map["deq"] = temp_tensor.op.input_tensors[1]
                elif temp_tensor.op.tag in ("dequant_scale",
                                            "dequant1_scale"):
                    tensor_map["c_ub"] = temp_tensor
                    tensor_attr["quant_fuse"] = True
                    tensor_map["deq"] = temp_tensor.op.input_tensors[1]
                    tensor_attr["deq_vector"] = False
                elif len(temp_tensor.op.input_tensors) == 2 \
                         and "elewise" in temp_tensor.op.tag:
                    input_tensor = temp_tensor.op.input_tensors[0]
                    if not input_tensor.op.input_tensors:
                        temp_tensor = temp_tensor.op.input_tensors[1]
                        continue
                elif "dequant_remove_pad" in temp_tensor.op.tag \
                        and c_ddr.op.tag != temp_tensor.op.tag:
                    sch[temp_tensor].compute_inline()
                temp_tensor = temp_tensor.op.input_tensors[0]

        def fetch_quant_info():
            ub_list = []
            input_cache_buffer = []
            temp_tensor = c_ddr
            while temp_tensor.op.input_tensors:
                if len(temp_tensor.op.input_tensors) == 2 \
                        and "elewise" in temp_tensor.op.tag:
                    input_tensor = temp_tensor.op.input_tensors[0]
                    c_ub_res = temp_tensor
                    if c_ddr.op.tag == temp_tensor.op.tag:
                        c_ub_res = sch.cache_write(c_ddr, scope_ubuf)
                    if not input_tensor.op.input_tensors:
                        input_cache_buffer.append([input_tensor, c_ub_res])
                        temp_tensor = temp_tensor.op.input_tensors[1]
                        continue
                    else:
                        input_tensor = temp_tensor.op.input_tensors[1]
                        input_cache_buffer.append([input_tensor, c_ub_res])
                elif temp_tensor.op.name in ("cast_i8_ub", "scale_sqrt_ub",
                                             "offset_ub", "reform_by_vmuls",
                                             "reform_by_vadds"):
                    ub_list.append(temp_tensor)
                elif temp_tensor.op.name in ("dequant2", "dequant_relu") \
                        or "elewise" in temp_tensor.op.tag:
                    if c_ddr.op.tag == temp_tensor.op.tag:
                        c_ub_res = sch.cache_write(c_ddr, scope_ubuf)
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

        tensor_map = {}
        tensor_attr = {}
        if c_ddr.op.tag == "requant_remove_pad":
            tensor_attr["n0_32_flag"] = True
            tensor_attr["quant_fuse"] = True
            tensor_map["data_transfer"] = c_ddr.op.input_tensors[0]
            tensor_map["c_ub"] = \
                tensor_map["data_transfer"].op.input_tensors[0]
            tensor_map["deq"] = tensor_map["c_ub"].op.input_tensors[1]
            c_ub_ddr = tensor_map["c_ub"].op.input_tensors[0]
            c_ub = c_ub_ddr.op.input_tensors[0]
            output_shape = list(i.value for
                                i in c_ub_ddr.op.attrs["output_shape"])
            tensor_attr["output_shape"] = output_shape
            tensor_map["c_ub_cut"] = c_ub_ddr
            sch[c_ub_ddr].compute_inline()
            sch[c_ub].compute_inline()
            sch[c_ub_ddr].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
        elif "quant" in c_ddr.op.tag or "elewise" in c_ddr.op.tag:
            _checkout_quant_fusion()
            if tensor_attr.get("quant_fuse"):
                fetch_quant_info()
                c_ub_ddr = tensor_map["c_ub"].op.input_tensors[0]
                c_ub = c_ub_ddr.op.input_tensors[0]
                output_shape = list(i.value for
                                    i in c_ub_ddr.op.attrs["output_shape"])
                tensor_attr["output_shape"] = output_shape
                tensor_map["c_ub_cut"] = c_ub_ddr
                sch[c_ub_ddr].compute_inline()
                sch[c_ub].compute_inline()
                sch[c_ub_ddr].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
        if tensor_attr.get("quant_fuse"):
            pass
        elif c_ddr.op.tag == "emit_insn_elewise_multiple_sel|bool":
            mask = c_ddr.op.input_tensors[0]
            if "elewise_binary_add" in c_ddr.op.input_tensors[1].op.tag:
                vadd_res = c_ddr.op.input_tensors[1]
                c_ub_cut, vadd_tensor = _get_vadd_tensors(vadd_res)
                tensor_map["vadd_res"] = vadd_res
            else:
                c_ub_cut = c_ddr.op.input_tensors[1]
            c_ub = c_ub_cut.op.input_tensors[0]
            output_shape = list(i.value \
                for i in c_ub_cut.op.attrs["output_shape"])

            tensor_map["c_ub_cut"] = c_ub_cut
            tensor_map["c_ub"] = c_ub
            tensor_attr["output_shape"] = output_shape
        elif c_ddr.op.tag == "elewise_binary_add":
            c_ub_cut, vadd_tensor = _get_vadd_tensors(c_ddr)
            c_ub = c_ub_cut.op.input_tensors[0]
            output_shape = list(i.value \
                for i in c_ub_cut.op.attrs["output_shape"])

            tensor_map["c_ub_cut"] = c_ub_cut
            tensor_map["c_ub"] = c_ub
            tensor_attr["output_shape"] = output_shape
        elif c_ddr.op.tag == "conv2d_backprop_input":
            c_ub = c_ddr.op.input_tensors[0]
            output_shape = list(i.value \
                for i in c_ddr.op.attrs["output_shape"])
            if c_ub.op.name == "bias_add_vector":
                tensor_map["bias_add_vector"] = c_ub
                c_ub = c_ub.op.input_tensors[0]
            tensor_map["c_ub"] = c_ub
            tensor_attr["output_shape"] = output_shape
        else:
            raise_dx_general_err(DX_SUPPORT_TAG_LOG_PREFIX \
                + ' dx or dx+drelu or dx+vadd or dx+vadd+drelu')

        if c_ub.op.input_tensors[0].name == "c_add_bias":
            c_add_bias = c_ub.op.input_tensors[0]
            bias_l0c = c_add_bias.op.input_tensors[0]
            c_col = c_add_bias.op.input_tensors[1]
            bias_ub_brc = bias_l0c.op.input_tensors[0]
            tensor_bias = bias_ub_brc.op.input_tensors[0]
            bias_ub = sch.cache_read(tensor_bias, scope_ubuf, [bias_ub_brc])
            tensor_map["c_add_bias"] = c_add_bias
            tensor_map["bias_l0c"] = bias_l0c
            tensor_map["bias_ub_brc"] = bias_ub_brc
            tensor_map["tensor_bias"] = tensor_bias
            tensor_map["bias_ub"] = bias_ub
        else:
            c_col = c_ub.op.input_tensors[0]
        a_col = c_col.op.input_tensors[0] # im2col_fractal in L0A
        b_col = c_col.op.input_tensors[1] # weight_transform in L0B
        b_ddr = b_col.op.input_tensors[0] # weight in ddr
        a_col_before = a_col.op.input_tensors[0] # im2col_row_major in L1
        padding = list(i.value for i in a_col_before.op.attrs["padding"])
        dilations = list(i.value for i in a_col_before.op.attrs["dilation"])

        tensor_map["c_col"] = c_col
        tensor_map["a_col"] = a_col
        tensor_map["b_col"] = b_col
        tensor_map["b_ddr"] = b_ddr
        tensor_map["a_col_before"] = a_col_before
        tensor_attr["padding"] = padding
        tensor_attr["dilations"] = dilations

        # stride > 1
        if a_col_before.op.input_tensors[0].op.tag == "dy_l1" or \
            a_col_before.op.input_tensors[0].op.tag == "dy_l1_cut":
            a_l1 = a_col_before.op.input_tensors[0]
            a_filling = a_l1.op.input_tensors[0]
            stride_h, stride_w = list(i.value for i in \
                a_filling.op.attrs["stride_expand"])
            a_ddr = a_filling.op.input_tensors[0] # dEdY in ddr
            a_zero = a_filling.op.input_tensors[1] # dEdY_zero in ub

            tensor_map["a_l1"] = a_l1
            tensor_map["a_filling"] = a_filling
            tensor_map["a_zero"] = a_zero

            a_ub = sch.cache_read(a_ddr, scope_ubuf, [a_filling])
            # generate a_zero in ub
            sch[a_zero].set_scope(scope_ubuf)
            sch[a_filling].set_scope(scope_ubuf)
            # dma : a_filling ub------>L1
            sch[a_l1].set_scope(scope_cbuf)
            tensor_map["a_ub"] = a_ub
        else:
            a_ddr = a_col_before.op.input_tensors[0] # dEdY in ddr
            stride_h = 1
            stride_w = 1
            a_l1 = sch.cache_read(a_ddr, scope_cbuf, [a_col_before])
            tensor_map["a_l1"] = a_l1

        # when add bias in ub
        bias_add_vector = tensor_map.get("bias_add_vector")
        if bias_add_vector is not None:
            sch[bias_add_vector].set_scope(scope_ubuf)
            bias_tensor = bias_add_vector.op.input_tensors[1]
            bias_ub = sch.cache_read(
                bias_tensor, scope_ubuf, [bias_add_vector]
            )
            tensor_map["bias_ub"] = bias_ub

        # when add bias in l0c
        if tensor_map.get("c_add_bias") is not None:
            sch[c_add_bias].set_scope(scope_cc)
            sch[bias_l0c].set_scope(scope_cc)
            sch[bias_ub_brc].set_scope(scope_ubuf)

        tensor_map["a_ddr"] = a_ddr
        tensor_attr["stride_h"] = stride_h
        tensor_attr["stride_w"] = stride_w

        # dataflow management
        b_l1 = sch.cache_read(b_ddr, scope_cbuf, [b_col])
        tensor_map["b_l1"] = b_l1
        sch[b_col].set_scope(scope_cb)

        sch[a_col_before].set_scope(scope_cbuf)
        sch[a_col].set_scope(scope_ca)
        sch[c_col].set_scope(scope_cc)
        sch[c_ub].set_scope(scope_ubuf)

        fusion_param = 0
        if c_ddr.op.tag == "emit_insn_elewise_multiple_sel|bool":
            fusion_param = 1/16
            if "elewise_binary_add" in c_ddr.op.input_tensors[1].op.tag:
                fusion_param += 1
                sch[vadd_res].set_scope(scope_ubuf)
                vadd_tensor_ub = sch.cache_read(vadd_tensor,
                                                scope_ubuf, [vadd_res])
                tensor_map["vadd_tensor_ub"] = vadd_tensor_ub
            sch[c_ub_cut].set_scope(scope_ubuf)
            mask_ub = sch.cache_read(mask, scope_ubuf, [c_ddr])
            c_ub_drelu = sch.cache_write(c_ddr, scope_ubuf)
            tensor_map["mask_ub"] = mask_ub
            tensor_map["c_ub_drelu"] = c_ub_drelu
        elif "requant_remove_pad" in c_ddr.op.tag:
            deq = tensor_map.get("deq")
            sch[tensor_map["c_ub"]].set_scope(scope_ubuf)
            deq_ub = sch.cache_read(deq, scope_ubuf, tensor_map["c_ub"])
            tensor_map["deq"] = deq_ub
            sch[tensor_map["data_transfer"]].set_scope(scope_ubuf)
            sch[tensor_map["c_ub"]].compute_inline()
            sch[tensor_map["c_ub"]].buffer_align((1, 1),
                                                 (1, 1),
                                                 (1, 16),
                                                 (1, 16))
            tensor_map["c_ub"] = tensor_map["data_transfer"]
        elif tensor_attr.get("quant_fuse"):
            deq_list = [tensor_map["c_ub"], ]
            sch[tensor_map["c_ub"]].set_scope(scope_ubuf)
            ub_list = tensor_map["ub_list"]
            if c_ddr.op.tag == "quant":
                fusion_param = 4
            else:
                fusion_param = 0
            for ub_tensor in ub_list:
                sch[ub_tensor].set_scope(scope_ubuf)
                if  "dequant2" in ub_tensor.op.name:
                    deq_list.append(ub_tensor)
                if "dequant" in ub_tensor.op.name \
                        and c_ddr.op.tag != "quant":
                    fusion_param += 1
            deq = tensor_map.get("deq")
            deq_ub = sch.cache_read(deq, scope_ubuf, deq_list)
            tensor_map["deq"] = deq_ub
            input_tensor = tensor_map["input_tensor"]
            if input_tensor and c_ddr.op.tag != "quant":
                fusion_param += 1
            for input_tensor_mem in input_tensor:
                sch[input_tensor_mem[1]].set_scope(scope_ubuf)
                input_ub = sch.cache_read(
                    input_tensor_mem[0], scope_ubuf, input_tensor_mem[1])
                input_tensor_mem[0] = input_ub
        elif c_ddr.op.tag == "elewise_binary_add":
            fusion_param = 1
            sch[c_ub_cut].set_scope(scope_ubuf)
            vadd_tensor_ub = sch.cache_read(vadd_tensor, scope_ubuf, [c_ddr])
            c_ub_vadd = sch.cache_write(c_ddr, scope_ubuf)
            tensor_map["vadd_tensor_ub"] = vadd_tensor_ub
            tensor_map["c_ub_vadd"] = c_ub_vadd
        tensor_attr["fusion_param"] = fusion_param

        return tensor_map, tensor_attr

    tensor_map, tensor_attr = _fetch_tensor_info()
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
    a_zero = tensor_map.get("a_zero")
    a_ddr = tensor_map.get("a_ddr")
    b_l1 = tensor_map.get("b_l1")
    a_ub = tensor_map.get("a_ub")
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

    # fetch tiling
    padu, padd, padl, padr = padding
    _, _, _, kernel_h, kernel_w, _ \
        = list(i.value for i in a_col_before.shape)
    _, _, _, dx_w, _ = output_shape

    img_shape = list(i.value for i in a_ddr.shape)
    _, dy_cout1, _, _, _ = img_shape

    w_trans_flag = False
    if b_ddr.dtype == "int8":
        w_trans_flag = True

    if w_trans_flag:
        # Cout1HkWk, Cin1, Cin0, Cout0
        b_ddr_k1, b_ddr_n1, b_ddr_n0, b_ddr_k0 \
            = list(i.value for i in b_ddr.shape)
        # Cout, Cin1, Hk, Wk, Cin0
        filter_shape = [b_ddr_k1//(kernel_h*kernel_w)*b_ddr_k0, b_ddr_n1,
                        kernel_h, kernel_w, b_ddr_n0]
    else:
        # HkWkCin1, Cout1, Cout0, Cin0
        b_ddr_n1, b_ddr_k1, b_ddr_k0, b_ddr_n0 \
            = list(i.value for i in b_ddr.shape)
        # Cout, Cin1, Hk, Wk, Cin0
        filter_shape = (b_ddr_k1*b_ddr_k0, b_ddr_n1//(kernel_h*kernel_w),
                        kernel_h, kernel_w, b_ddr_n0)
    mad_type = "float32"
    if c_ddr.dtype == "int8":
        filter_shape[1] = (filter_shape[1] + 1) // 2 * 2
    if b_ddr.dtype == "int8":
        mad_type = "int32"

    if c_add_bias is not None or bias_add_vector is not None:
        bias_flag = 1
    else:
        bias_flag = 0

    def _get_kernel_name():
        if fusion_param > 0 or tensor_attr.get("quant_fuse"):
            _kernel_name = c_ub_cut.op.attrs["kernel_name"]
        else:
            _kernel_name = tensor.op.attrs["kernel_name"]
        return _kernel_name

    _kernel_name = _get_kernel_name()

    tiling = tiling_query(a_shape=img_shape,
                          b_shape=filter_shape,
                          c_shape=output_shape,
                          a_dtype=a_ddr.dtype,
                          b_dtype=b_ddr.dtype,
                          c_dtype=c_ddr.dtype,
                          mad_dtype=mad_type,
                          padl=padl, padr=padr, padu=padu, padd=padd,
                          strideh=1, stridew=1,
                          strideh_expand=stride_h,
                          stridew_expand=stride_w,
                          dilationh=dilation_h,
                          dilationw=dilation_w,
                          group=1,
                          fused_double_operand_num=fusion_param,
                          bias_flag=bias_flag,
                          op_tag='conv2d_backprop_input',
                          kernel_name=_kernel_name)

    if DEBUG_MODE:
        print('general input shape:', 'filter:', filter_shape, 'dy:',
              output_shape, 'dx:', img_shape)
        print("general input tiling", tiling)
        print("general dx fusion tag:", c_ddr.op.tag)

    def _tiling_check_none():
        if (tiling.get("AL1_shape") is None) or \
           (tiling.get("BL1_shape") is None) or \
           (tiling.get("CUB_matrix") is None):
            raise_dx_general_err("AL1_shape/BL1_shape/CUB_matrix "
                               "can't be None.")
        if (tiling.get("AL0_matrix") is None) or \
           (tiling.get("BL0_matrix") is None) or \
           (tiling.get("CL0_matrix") is None):
            raise_dx_general_err("AL0_matrix/BL0_matrix/CL0_matrix "
                               "can't be None.")

    def _tiling_l0_process():
        if al0_tiling_ma == a_col_ma \
                and al0_tiling_ka == a_col_ka \
                and a_col_batch == 1:
            tiling["AL0_matrix"] = []

        if tiling.get("BL0_matrix") != []:
            bl0_tiling_kb, bl0_tiling_nb,\
            bl0_tiling_n0, bl0_tiling_k0, _, _ = tiling.get("BL0_matrix")
        else:
            bl0_tiling_kb, bl0_tiling_nb,\
            bl0_tiling_n0, bl0_tiling_k0, = list(i.value for i in b_col.shape)
        return bl0_tiling_kb, bl0_tiling_nb, bl0_tiling_n0, bl0_tiling_k0

    def _tiling_l1_process():
        if tiling.get("AL1_shape") != []:
            al1_tilling_k, al1_tilling_m, _, _ = tiling.get("AL1_shape")
            if al1_tilling_k == kernel_h*kernel_w*al1_co1*al1_co0 and \
               al1_tilling_m == c_l0c_hw \
               // (CUBE_MKN[c_col.dtype]["mac"][0]*cl0_tiling_mc):
                tiling["AL1_shape"] = []
        else:
            # batch = 1 other axes full load
            al1_tilling_k = kernel_h*kernel_w*al1_co1*al1_co0
            al1_tilling_m = c_l0c_hw \
                            // (CUBE_MKN[c_col.dtype]["mac"][0]*cl0_tiling_mc)

        if tiling.get("BL1_shape") != []:
            bl1_tilling_k, bl1_tilling_n, _, _ = tiling.get("BL1_shape")
        else:
            if w_trans_flag:
                bl1_tilling_k = bl1_co0*bl1_co1
                bl1_tilling_n = bl1_k1 // cl0_tiling_nc
            else:
                bl1_tilling_k = kernel_h*kernel_w*bl1_co0*bl1_co1
                bl1_tilling_n = bl1_k1//(kernel_h*kernel_w*cl0_tiling_nc)
        return al1_tilling_k, al1_tilling_m, bl1_tilling_k, bl1_tilling_n

    # check tiling
    def _tiling_check_equal():
        if tiling.get("BL0_matrix") != []:
            if al0_tiling_ka != bl0_tiling_kb:
                raise_dx_general_err("ka != kb.")
            if bl0_tiling_nb != cl0_tiling_nc:
                raise_dx_general_err("nb != nc.")

        if al0_tiling_ma != cl0_tiling_mc:
            raise_dx_general_err("ma != mc.")

    def _tiling_check_factor():
        if (kernel_w*kernel_h*dy_cout1) % al0_tiling_ka != 0:
            raise_dx_general_err("Co1*Hk*Wk % ka != 0")

        if al1_tilling_k % al0_tiling_ka != 0:
            raise_dx_general_err("k_AL1 % ka != 0.")

        if tiling.get("BL1_shape") != [] and tiling.get("BL0_matrix") != []:
            if bl1_tilling_k % bl0_tiling_kb != 0:
                raise_dx_general_err("k_BL1 % kb != 0.")

        if cl0_tiling_nc % cub_tiling_nc_factor != 0:
            raise_dx_general_err("nc % nc_factor != 0.")

        if tiling.get("BL1_shape") != []:
            if al1_tilling_k > bl1_tilling_k \
                    and al1_tilling_k % bl1_tilling_k != 0:
                raise_dx_general_err("k_AL1 > k_BL1 but k_AL1 % k_BL1 != 0.")
            if bl1_tilling_k > al1_tilling_k \
                    and bl1_tilling_k % al1_tilling_k != 0:
                raise_dx_general_err("k_BL1 > k_AL1 but k_BL1 % k_AL1 != 0.")

    def _tiling_check_load():
        if stride_h == 1 and stride_w == 1:
            if tiling.get("AUB_shape") is not None:
                raise_dx_general_err("stride = 1 but AUB_shape is not None.")

        if  tiling.get("BL0_matrix") == [] and tiling.get("BL1_shape") != []:
            raise_dx_general_err("BL0 full load but BL1 not!")

    def _tiling_check_pbuffer():
        if stride_h > 1 or stride_w > 1:
            if aub_pbuffer not in (1, 2):
                raise_dx_general_err("value of AUB_pbuffer can only be 1 or 2")

        if al1_pbuffer not in (1, 2):
            raise_dx_general_err("value of AL1_pbuffer can only be 1 or 2")

        if bl1_pbuffer not in (1, 2):
            raise_dx_general_err("value of BL1_pbuffer can only be 1 or 2")

        if al0_pbuffer not in (1, 2):
            raise_dx_general_err("value of AL0_pbuffer can only be 1 or 2")

        if bl0_pbuffer not in (1, 2):
            raise_dx_general_err("value of BL0_pbuffer can only be 1 or 2")

        if l0c_pbuffer not in (1, 2):
            raise_dx_general_err("value of L0C_pbuffer can only be 1 or 2")

        if cub_pbuffer not in (1, 2):
            raise_dx_general_err("value of CUB_pbuffer can only be 1 or 2")

    _tiling_check_none()
    cub_tiling_nc_factor, cub_tiling_mc_factor, \
    cub_tiling_m0, cub_tiling_n0, _, _ \
        = tiling.get("CUB_matrix")
    cl0_tiling_nc, cl0_tiling_mc, \
    cl0_tiling_m0, cl0_tiling_n0, _, _ \
        = tiling.get("CL0_matrix")
    al0_tiling_ma, al0_tiling_ka, \
    al0_tiling_m0, al0_tiling_k0, _, _ \
        = tiling.get("AL0_matrix")

    batch_dim, n_dim, m_dim, _ = tiling.get("block_dim")
    aub_pbuffer = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
    al1_pbuffer = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
    bl1_pbuffer = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
    al0_pbuffer = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
    bl0_pbuffer = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
    l0c_pbuffer = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
    cub_pbuffer = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")

    _, al1_co1, _, _, al1_co0 = list(i.value for i in a_l1.shape)
    _, _, c_l0c_hw, _ = list(i.value for i in c_col.shape)
    if w_trans_flag:
        # Cout1*Hk*Wk, Cin1, Cin0, Cout0
        bl1_co1, bl1_k1, _, bl1_co0 = list(i.value for i in b_l1.shape)
    else:
        # Cin1*Hk*Wk, Cout1, Cout0, Cin0
        bl1_k1, bl1_co1, bl1_co0, _ = list(i.value for i in b_l1.shape)
    c_col_k1, c_col_k0 = list(ax.dom.extent.value for ax in
                              c_col.op.reduce_axis)
    a_col_shape = list(i.value for i in a_col.shape)
    a_col_batch, a_col_ma, a_col_ka, _, _ = a_col_shape

    bl0_tiling_kb, bl0_tiling_nb, \
    bl0_tiling_n0, bl0_tiling_k0 = _tiling_l0_process()
    al1_tilling_k, al1_tilling_m, \
    bl1_tilling_k, bl1_tilling_n = _tiling_l1_process()
    _tiling_check_equal()
    _tiling_check_factor()
    _tiling_check_load()
    _tiling_check_pbuffer()

    def _cub_process():  # pylint: disable=R0912

        def fusion_cub_process():
            if c_ddr.op.tag == "emit_insn_elewise_multiple_sel|bool":
                if "elewise_binary_add" in c_ddr.op.input_tensors[1].op.tag:
                    sch_agent[c_ub].reused_by(c_ub_cut, vadd_res, c_ub_drelu)
                    sch_agent.same_attach(vadd_res, c_ub)
                    sch_agent.same_attach(vadd_tensor_ub, c_ub)
                else:
                    sch_agent[c_ub].reused_by(c_ub_cut, c_ub_drelu)

                sch_agent.same_attach(c_ub_cut, c_ub)
                sch_agent.same_attach(c_ub_drelu, c_ub)
                sch_agent.same_attach(mask_ub, c_ub)
            elif c_ddr.op.tag == "requant_remove_pad":
                sch_agent.same_attach(tensor_map["deq"], c_ub)
            elif tensor_attr.get("quant_fuse"):
                sch_agent.same_attach(tensor_map["deq"], c_ub)
                for ub_tensor in tensor_map["ub_list"]:
                    sch_agent.same_attach(ub_tensor, c_ub)
                for input_tensor_mem in tensor_map["input_tensor"]:
                    sch_agent.same_attach(input_tensor_mem[0], c_ub)
                    sch_agent.same_attach(input_tensor_mem[1], c_ub)
            elif c_ddr.op.tag == "elewise_binary_add":
                sch_agent[c_ub].reused_by(c_ub_cut, c_ub_vadd)
                sch_agent.same_attach(vadd_tensor_ub, c_ub)
                sch_agent.same_attach(c_ub_cut, c_ub)
                sch_agent.same_attach(c_ub_vadd, c_ub)

        c_ub_nc_factor = cub_tiling_nc_factor

        if c_ddr.op.tag == "emit_insn_elewise_multiple_sel|bool":
            _, _, dx_h, dx_w, _ = output_shape
            dx_hw = dx_h*dx_w
            tiling_m_axis = cl0_tiling_mc*cl0_tiling_m0
            if (dx_hw - dx_hw // tiling_m_axis*tiling_m_axis) % 16 != 0:
                c_ub_nc_factor = 1

        # dx_batch, dx_cin1, dx_m, dx_cin0
        op_shape = list(i.value for i in c_ub.shape)
        if n0_32_flag is not None:
            affine_cub = 1, \
                        int(c_ub_nc_factor / 2), \
                        cub_tiling_mc_factor*cub_tiling_m0, \
                        cub_tiling_n0 * 2
            if c_ddr.op.tag == "quant":
                op_shape[1] = (op_shape[1] + 1) // 2
                op_shape[3] = op_shape[3] * 2
        else:
            affine_cub = 1, \
                        c_ub_nc_factor, \
                        cub_tiling_mc_factor*cub_tiling_m0, \
                        cub_tiling_n0

        status = Compare.compare(affine_cub, op_shape)
        if status == Compare.EQUAL:
            pass
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(c_ub, c_ddr, affine_shape=affine_cub)
        else:
            raise_dx_general_err("c_ub attach error.")

        sch[c_ub].buffer_align(
            (1, 1),
            (1, 1),
            (1, CUBE_MKN[c_ub.dtype]["mac"][0]),
            (1, CUBE_MKN[c_ub.dtype]["mac"][2]))
        if bias_add_vector is not None:
            sch_agent[c_ub].reused_by(bias_add_vector)
            sch[bias_add_vector].buffer_align(
                (1, 1), (1, 1),
                (1, CUBE_MKN[c_ub.dtype]["mac"][0]),
                (1, CUBE_MKN[c_ub.dtype]["mac"][2]))

        fusion_cub_process()

        return affine_cub

    def _cl0_process(affine_cub):
        if n0_32_flag is not None:
            affine_l0c = 1, int(cl0_tiling_nc / 2), \
                     cl0_tiling_mc*cl0_tiling_m0, cl0_tiling_n0*2
        else:
            affine_l0c = 1, cl0_tiling_nc, \
                cl0_tiling_mc*cl0_tiling_m0, cl0_tiling_n0
        c_col_shape = list(i.value for i in c_col.shape)
        status_ori = Compare.compare(affine_l0c, c_col_shape)
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
            raise_dx_general_err("c_col attach error.")

        sch[c_col].buffer_align(
            (1, 1),
            (1, 1),
            (1, CUBE_MKN[c_col.dtype]["mac"][0]),
            (1, CUBE_MKN[c_col.dtype]["mac"][2]),
            (1, 1),
            (1, CUBE_MKN[c_col.dtype]["mac"][1]))

    def _l0a_process():
        l0a2l0c_affine_shape = 1, \
                            None, \
                            al0_tiling_ma*al0_tiling_m0, \
                            cl0_tiling_n0, \
                            al0_tiling_ka, \
                            al0_tiling_k0
        tiling_ori_l0a = 1, \
                        al0_tiling_ma, \
                        al0_tiling_ka, \
                        al0_tiling_m0, \
                        al0_tiling_k0

        status_ori = Compare.compare(tiling_ori_l0a, a_col_shape)
        status = Compare.compare([al0_tiling_ma,
                                  al0_tiling_m0,
                                  al0_tiling_ka,
                                  al0_tiling_k0],
                                 [cl0_tiling_mc,
                                  cl0_tiling_m0,
                                  c_col_k1,
                                  c_col_k0])
        if status_ori == Compare.EQUAL:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(a_col, c_col)
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(
                a_col, c_col, affine_shape=l0a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l0a2out_affine_shape = [1,
                                    None,
                                    al0_tiling_ma*al0_tiling_m0,
                                    cl0_tiling_n0]
            sch_agent.attach_at(
                a_col, c_ddr, affine_shape=l0a2out_affine_shape)
        else:
            raise_dx_general_err("l0a attach error.")

    def _l0b_process():
        neg_src_stride = True
        if tiling.get("BL0_matrix") != []:
            l0b2l0c_affine_shape = None, \
                                   bl0_tiling_nb, \
                                   cl0_tiling_mc*cl0_tiling_m0,\
                                   bl0_tiling_n0,\
                                   bl0_tiling_kb, \
                                   bl0_tiling_k0
            tiling_ori_l0b = bl0_tiling_kb,\
                             bl0_tiling_nb, \
                             bl0_tiling_n0, \
                             bl0_tiling_k0
            b_col_shape = list(i.value for i in b_col.shape)
            status_ori = Compare.compare(tiling_ori_l0b, b_col_shape)
            status = Compare.compare([bl0_tiling_nb,
                                      bl0_tiling_n0,
                                      bl0_tiling_kb,
                                      bl0_tiling_k0],
                                     [cl0_tiling_nc,
                                      cl0_tiling_n0,
                                      c_col_k1,
                                      c_col_k0])
            neg_src_stride = False
            if status_ori == Compare.EQUAL:
                neg_src_stride = True
            elif status == Compare.EQUAL:
                sch_agent.same_attach(b_col, c_col)
            elif status == Compare.LESS_EQ:
                sch_agent.attach_at(
                    b_col, c_col, affine_shape=l0b2l0c_affine_shape)
            elif status == Compare.GREATE_EQ:
                l0b2out_affine_shape = [1,
                                        bl0_tiling_nb,
                                        cl0_tiling_m0,
                                        bl0_tiling_n0]
                sch_agent.attach_at(
                    b_col, c_ddr, affine_shape=l0b2out_affine_shape)
            else:
                raise_dx_general_err("l0b attach error.")
        return neg_src_stride

    def _al1_process():
        l1_ma = al1_tilling_m*al0_tiling_ma
        l1_ka = al1_tilling_k // al0_tiling_k0
        l1a2l0c_affine_shape = 1,\
                               None, \
                               l1_ma*al0_tiling_m0,\
                               bl0_tiling_n0,\
                               l1_ka,\
                               al0_tiling_k0
        status = Compare.compare([l1_ma,
                                  al0_tiling_m0,
                                  l1_ka,
                                  al0_tiling_k0],
                                 [cl0_tiling_mc,
                                  cl0_tiling_m0,
                                  c_col_k1,
                                  c_col_k0])
        if tiling.get("AL1_shape") == [] and tiling.get("AL0_matrix") == []:
            pass
        elif status == Compare.EQUAL:
            sch_agent.same_attach(a_l1, c_col)
        elif status == Compare.LESS_EQ:
            sch_agent.attach_at(a_l1, c_col,
                                affine_shape=l1a2l0c_affine_shape)
        elif status == Compare.GREATE_EQ:
            l1a2out_affine_shape = [1,
                                    None,
                                    l1_ma*al0_tiling_m0,
                                    cl0_tiling_n0]
            sch_agent.attach_at(a_l1, c_ddr,
                                affine_shape=l1a2out_affine_shape)
        else:
            raise_dx_general_err("a_l1 atach error.")
        sch_agent.same_attach(a_col_before, a_l1)
        sch[a_col_before].buffer_align(
            (1, 1),
            (dx_w, dx_w),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, CUBE_MKN[a_col_before.dtype]["mac"][1]))

    def _bl1_process():
        if tiling.get("BL1_shape") != []:
            l1_nb = bl1_tilling_n*bl0_tiling_nb
            _, _k0, _n0 = CUBE_MKN[b_l1.dtype]["mac"]
            l1_kb = bl1_tilling_k // _k0
            l1b2l0c_affine_shape = None,\
                                   l1_nb,\
                                   cl0_tiling_m0,\
                                   bl0_tiling_n0,\
                                   l1_kb,\
                                   bl0_tiling_k0
            if w_trans_flag:
                tiling_ori_bl1 = bl1_tilling_k,\
                                 l1_nb,\
                                 _n0, _k0
            else:
                tiling_ori_bl1 = l1_nb*kernel_h*kernel_w, \
                                bl1_tilling_k//(kernel_h*kernel_w*16), 16, 16

            bl1_shape = list(i.value for i in b_l1.shape)
            status_ori = Compare.compare(tiling_ori_bl1, bl1_shape)
            status = Compare.compare([l1_nb,
                                      bl0_tiling_n0,
                                      l1_kb,
                                      bl0_tiling_k0],
                                     [cl0_tiling_nc,
                                      cl0_tiling_n0,
                                      c_col_k1,
                                      c_col_k0])
            if status_ori == Compare.EQUAL:
                # bl1 full load but tiling.get("BL1_shape") is not []
                pass
            elif status == Compare.EQUAL:
                sch_agent.same_attach(b_l1, c_col)
            elif status == Compare.LESS_EQ:
                sch_agent.attach_at(
                    b_l1, c_col, affine_shape=l1b2l0c_affine_shape)
            elif status == Compare.GREATE_EQ:
                l1b2out_affine_shape = [None, l1_nb,
                                        cl0_tiling_m0, bl0_tiling_n0]
                sch_agent.attach_at(
                    b_l1, c_ddr, affine_shape=l1b2out_affine_shape)
            else:
                raise_dx_general_err("b_l1 attach error.")

    def _aub_process():
        if stride_h > 1 or stride_w > 1:
            aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
            _, _, _, aub_w, _ = list(i.value for i in a_filling.shape)
            if aub_tiling_m == 0:
                sch_agent.same_attach(a_filling, a_l1)
            else:
                ub_shape = [1,
                            aub_tiling_k//(kernel_h*kernel_w*16),
                            aub_tiling_m,
                            aub_w,
                            al1_co0]
                sch_agent.attach_at(a_filling, a_l1, ub_shape)
            sch_agent.same_attach(a_zero, a_filling)
            sch_agent.same_attach(a_ub, a_filling)

    def _attach_bias():
        if c_add_bias is not None:
            sch_agent.same_attach(c_add_bias, c_col)
            sch_agent.same_attach(bias_l0c, c_col)
            sch_agent.same_attach(bias_ub_brc, c_col)

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
            if c_ddr.op.tag == "emit_insn_elewise_multiple_sel|bool":
                sch[c_ub_cut].double_buffer()
                sch[c_ub_drelu].double_buffer()
                if "elewise_binary_add" in c_ddr.op.input_tensors[1].op.tag:
                    sch[vadd_res].double_buffer()
            elif c_ddr.op.tag == "elewise_binary_add" \
                    and not tensor_attr.get("quant_fuse"):
                sch[c_ub_cut].double_buffer()
                sch[c_ub_vadd].double_buffer()

        if stride_h > 1 or stride_w > 1:
            if tiling.get("manual_pingpong_buffer").get("AUB_pbuffer") == 2:
                sch[a_ub].double_buffer()
                sch[a_filling].double_buffer()
                sch[a_zero].double_buffer()

        if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == 2:
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
        deq_axis_mode = {False: (0, "scalar"),
                         True: (2, "vector")}

        def _emit_requant_fusion_insn():
            sch_agent[tensor_map["deq"]].emit_insn(
                tensor_map["deq"].op.axis[0], 'dma_copy')
            c_ub_reform = tensor_map["c_ub"]
            reform_outer, reform_inner = sch[c_ub_reform].split(
                c_ub_reform.op.axis[3], nparts=2
            )
            sch[c_ub_reform].reorder(reform_outer,
                                     c_ub_reform.op.axis[0],
                                     c_ub_reform.op.axis[1],
                                     c_ub_reform.op.axis[2],
                                     reform_inner)
            sch[c_ub_reform].emit_insn(
                c_ub_reform.op.axis[2], "dma_copy")

        def _emit_quant_fusion_insn():
            sch_agent[tensor_map["deq"]].emit_insn(
                tensor_map["deq"].op.axis[0], 'dma_copy')
            deq_axis = deq_axis_mode[tensor_attr.get("deq_vector")][0]
            if is_v200_version():
                sch[tensor_map["c_ub"]].emit_insn(
                    tensor_map["c_ub"].op.axis[deq_axis], 'dma_copy')
            else:
                deq_mode = deq_axis_mode[tensor_attr.get("deq_vector")][1]
                sch_agent[c_ub].pragma(
                    sch_agent[c_ub].op.axis[deq_axis],
                    "deq_scale", deq_mode)
            for ub_tensor in tensor_map["ub_list"]:
                if ub_tensor.op.name == "input_ub":
                    sch_agent[ub_tensor].emit_insn(
                        sch_agent[ub_tensor].op.axis[0], "dma_padding")
                elif "reform" in ub_tensor.op.name:
                    ndim = len(sch[ub_tensor].op.axis)
                    coo, _ = sch[ub_tensor].split(
                        sch[ub_tensor].op.axis[ndim - 1],
                        CUBE_MKN["float16"]["mac"][1])
                    axis_list = sch[ub_tensor].op.axis[0:ndim - 1]
                    sch[ub_tensor].reorder(coo, *axis_list)
                    sch[ub_tensor].emit_insn(
                        sch[ub_tensor].op.axis[2], "vector_auto")
                elif ub_tensor.op.name == "cast_i8_ub":
                    if is_v200_version() or is_lhisi_version():
                        conv_mode = "vector_conv_{}".format(
                            tensor_attr.get("q_mode").lower())
                    else:
                        conv_mode = "vector_conv"
                    sch_agent[ub_tensor].emit_insn(
                        sch_agent[ub_tensor].op.axis[0], conv_mode)
                else:
                    sch_agent[ub_tensor].emit_insn(
                        sch_agent[ub_tensor].op.axis[0], "vector_auto")
            for input_tensor_mem in tensor_map["input_tensor"]:
                sch_agent[input_tensor_mem[0]].emit_insn(
                    sch_agent[input_tensor_mem[0]].op.axis[0], "dma_copy")
                sch_agent[input_tensor_mem[1]].emit_insn(
                    sch_agent[input_tensor_mem[1]].op.axis[0], "vector_auto")

        if c_ddr.op.tag == "requant_remove_pad":
            _emit_requant_fusion_insn()
        elif tensor_attr.get("quant_fuse"):
            _emit_quant_fusion_insn()
        else:
            sch_agent[c_ub].emit_insn(sch_agent[c_ub].op.axis[0], "dma_copy")

        if c_ddr.op.tag == "emit_insn_elewise_multiple_sel|bool":
            sch[mask_ub].emit_insn(mask_ub.op.axis[0], 'dma_copy')

            if "elewise_binary_add" in c_ddr.op.input_tensors[1].op.tag:
                sch[vadd_tensor_ub].emit_insn(vadd_tensor_ub.op.axis[0],
                                              'dma_copy')
                sch[vadd_res].emit_insn(vadd_res.op.axis[0], "vector_add")
            sch[c_ub_drelu].emit_insn(c_ub_drelu.op.axis[0],
                                      "vector_selects_bool")

            sch[c_ub_cut].emit_insn(c_ub_cut.op.axis[0], 'phony_insn')
        elif c_ddr.op.tag == "elewise_binary_add" \
                and not tensor_attr.get("quant_fuse"):
            sch[vadd_tensor_ub].emit_insn(vadd_tensor_ub.op.axis[0],
                                          'dma_copy')
            sch[c_ub_vadd].emit_insn(c_ub_vadd.op.axis[0], "vector_add")
            sch[c_ub_cut].emit_insn(c_ub_cut.op.axis[0], 'dma_copy')

    def _emit_insn():  # pylint: disable=R0914,R0915
        sch_agent[b_l1].emit_insn(sch_agent[b_l1].op.axis[0], "dma_copy")

        if bias_add_vector is not None:
            sch[bias_ub].emit_insn(sch[bias_ub].op.axis[0], "dma_copy")
            sch[bias_add_vector].emit_insn(
                sch[bias_add_vector].op.axis[0], 'vector_auto',
            )

        if neg_src_stride \
            and (is_cloud_version()
                 or is_v200_version()
                 or is_lhisi_version()):
            _, b_col_inner \
                        = sch_agent[b_col].split(sch_agent[b_col].op.axis[0],
                                                 factor=kernel_h*kernel_w)
            sch_agent[b_col].emit_insn(b_col_inner, "dma_copy")
        else:
            sch_agent[b_col].emit_insn(sch_agent[b_col].op.axis[1], "dma_copy")

        if stride_h == 1 and stride_w == 1:
            sch_agent[a_l1].emit_insn(sch_agent[a_l1].op.axis[0], "dma_copy")
        else:
            sch_agent[a_ub].emit_insn(sch_agent[a_ub].op.axis[0], "dma_copy")
            afill_n, afill_c, afill_h, afill_w, _ = \
                sch_agent[a_filling].get_active_scopes()
            afill_w_out, afill_w_inner = sch_agent[a_filling].split(
                afill_w, factor=stride_w)
            sch_agent[a_filling].reorder(
                afill_w_inner,
                afill_n,
                afill_c,
                afill_h,
                afill_w_out)
            sch_agent[a_filling].unroll(afill_w_inner)
            sch_agent[a_filling].reused_by(a_zero)
            sch_agent[a_zero].emit_insn(sch_agent[a_zero].op.axis[0],
                                        "vector_dup")
            if w_trans_flag:
                sch_agent[a_filling].emit_insn(afill_n, "dma_copy")
            else:
                sch_agent[a_filling].emit_insn(afill_n, "vector_muls")
            al1_insn, _ = sch_agent[a_l1].nlast_scopes(2)
            sch_agent[a_l1].emit_insn(al1_insn, "dma_copy")

        setfmatrix_dict = {"conv_kernel_h": kernel_h,
                           "conv_kernel_w": kernel_w,
                           "conv_padding_top": padu,
                           "conv_padding_bottom": padd,
                           "conv_padding_left": padl,
                           "conv_padding_right": padr,
                           "conv_stride_h": 1,
                           "conv_stride_w": 1,
                           "conv_dilation_h": dilation_h,
                           "conv_dilation_w": dilation_w,
                          }

        setfmatrix_dict["conv_fm_c"] = a_l1.shape[1]*a_l1.shape[4]
        setfmatrix_dict["conv_fm_h"] = a_l1.shape[2]
        setfmatrix_dict["conv_fm_w"] = a_l1.shape[3]

        sch_agent[a_col_before].emit_insn(a_col_before.op.axis[0],
                                          'set_fmatrix',
                                          setfmatrix_dict)
        sch_agent[a_col].emit_insn(a_col.op.axis[0], 'im2col')

        scopes_intrins = sch_agent[c_col].intrin_scopes(6)
        scope_insn = scopes_intrins[0]
        inner_k_axis = sch_agent[c_col].get_relate_scope(
            c_col.op.reduce_axis[0], scope_insn)
        if inner_k_axis:
            mad_dict = {"mad_pattern": 2,
                        "k_outer": sch_agent[c_col].get_relate_scope(
                            c_col.op.reduce_axis[0], scope_insn)}
        else:
            inner_n, inner_co, inner_m, \
            inner_co0, inner_k1, inner_k0 = scopes_intrins
            inner_ko, inner_ki = sch_agent[c_col].split(inner_k1, nparts=1)
            sch_agent[c_col].reorder(inner_ko, inner_n, inner_co,
                                     inner_m, inner_co0,
                                     inner_ki, inner_k0)
            mad_dict = {"mad_pattern": 2, "k_outer": [inner_ko]}
        if bias_ub_brc is not None:
            sch[bias_l0c].reused_by(c_add_bias, c_col)
            sch[c_add_bias].emit_insn(c_add_bias.op.axis[0], 'phony_insn')
            cc_outer, _ = sch[bias_l0c].split(
                bias_l0c.op.axis[2], BRC_STANDARD_BLOCK_SIZE)
            sch[bias_l0c].emit_insn(cc_outer, 'dma_copy')
            sch[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')
            sch[bias_ub_brc].emit_insn(bias_ub_brc.op.axis[0], 'vector_auto')
            mad_dict["init_bias"] = 1
        sch_agent[c_col].emit_insn(scope_insn, "mad", mad_dict)
        sch[c_ddr].emit_insn(sch_agent[c_ddr].nlast_scopes(2)[0], "dma_copy")
        _emit_fusion_insn()

    sch_agent = ScheduleAgent(sch)
    affine_cub = _cub_process()
    _cl0_process(affine_cub)
    _l0a_process()
    neg_src_stride = _l0b_process()
    _do_l1andub_process()
    _attach_bias()

    def _bind_core():
        axs = sch_agent[c_ddr].get_active_scopes()
        ax_ni, ax_ci, ax_hw, _ = axs
        ax_core = sch_agent[c_ddr].bind_core([ax_ni, ax_ci, ax_hw],
                                             [batch_dim, n_dim, m_dim])
        ax_core_in = sch_agent[c_ddr].get_superkernel_axis_pragma()
        sch_agent.root_stage_at(c_ddr, ax_core)
        blocks = batch_dim * n_dim * m_dim
        if blocks == batch_dim:
            sch[c_ddr].pragma(ax_core_in, 'json_info_batchBindOnly')
    _bind_core()
    _double_buffer()
    _emit_insn()
    sch_agent.apply()
    tiling.clear()
    return True
