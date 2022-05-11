#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
Schedule of conv2d + bn1 fusion.
"""
from tbe import tvm
from tbe.dsl.compute.conv_compute import ConvParam
from tbe.dsl.compute.conv_compute import is_support_fixpipe
from tbe.dsl.static_schedule.conv_schedule_util import get_src_tensor


def check_conv_bn1(outs):
    """
    check conv + bn1

    Parameters
    ----------
    outs : the outputs of op

    Returns
    -------

    """
    if isinstance(outs, tvm.tensor.Tensor) or not isinstance(outs, list) or len(outs) != 3:
        return False

    conv_out, reduce_0, reduce_1 = outs
    if "convolution_" not in conv_out.op.tag and "conv_vector_bias_add" not in conv_out.op.tag:
        return False
    if ("reduce_sum" not in reduce_0.op.tag) or \
            ("reduce_sum" not in reduce_1.op.tag):
        return False

    # check conv_type is fp16
    if conv_out.dtype != "float16" and not is_support_fixpipe():
        return False
    # check cast->reduce_0
    reduce0_src = get_src_tensor(reduce_0)
    reduce0_src_pre = get_src_tensor(reduce0_src)

    if conv_out.dtype == "float16":
        if ("elewise_single_cast" not in reduce0_src.op.tag) or (reduce0_src_pre != conv_out):
            return False
    if conv_out.dtype == "float32" and reduce0_src != conv_out:
        return False

    # check reduce_axis
    if (conv_out.op.axis[1].dom.extent.value != reduce_0.op.axis[1].dom.extent.value) or \
            (conv_out.op.axis[1].dom.extent.value != reduce_1.op.axis[1].dom.extent.value):
        return False

    return True


def convbn1_recompute(outs):
    """
    reform conv + bn1 compute
    """
    conv_out, _, _ = outs
    conv_res_shape = tuple(conv_out.shape)
    # add for group pattern
    cout1_opt = ConvParam.para_dict["cout1_opt"]
    # end for group pattern
    reduce_shape = (conv_res_shape[1], conv_res_shape[3])
    k_0 = tvm.reduce_axis((0, conv_res_shape[0]), name='k_0')
    k_1 = tvm.reduce_axis((0, conv_res_shape[2]), name='k_1')
    op_tag = "convolution_"
    group = ConvParam.para_dict["group"]

    from tbe.dsl.api import vmul

    def _fcombine(arg_0, arg_1):
        """
        return index tupe

        Parameters
        ----------
        arg_0 : the operand of reduction, a tuple of index and value
        arg_1 : the operand of reduction, a tuple of index and value

        Returns
        -------
        index tupe

        """
        return arg_0[0] + arg_1[0], arg_0[1] + arg_1[1]

    def _fidentity(t_0, t_1):
        """
        return tvm const tupe

        """
        return tvm.const(0, t_0), tvm.const(0, t_1)

    if is_support_fixpipe():
        fmap = ConvParam.tensor_map["fmap"]
        enable_fp32_flag = (fmap.dtype == "float32" and conv_out.dtype == "float32")
        # gm -> UB
        c_ub = tvm.compute(conv_res_shape,
                           lambda *indice: conv_out(*indice),
                           name='c_ub',
                           tag=op_tag + "c_ub")
        tuple_reduce = tvm.comm_reducer(_fcombine,
                                        _fidentity,
                                        name='tuple_reduce')
        if enable_fp32_flag:
            mul_1 = vmul(c_ub, c_ub)
            mean_out, _ = tvm.compute(reduce_shape,
                                      lambda c1, c0:
                                          tuple_reduce((c_ub[k_0, c1, k_1, c0],
                                                        mul_1[k_0, c1, k_1, c0]),
                                                       axis=[k_0, k_1]),
                                      name="mean_out")
        else:
            cast1 = tvm.compute(conv_res_shape,
                                lambda *indice:
                                    c_ub(*indice).astype("float32"),
                                name="cast1",
                                tag="elewise_single_cast")
            mul_1 = vmul(cast1, cast1)
            mean_out, _ = tvm.compute(reduce_shape,
                                      lambda c1, c0:
                                          tuple_reduce((cast1[k_0, c1, k_1, c0],
                                                        mul_1[k_0, c1, k_1, c0]),
                                                       axis=[k_0, k_1]),
                                      name="mean_out")
        outputs = [conv_out, mean_out]
    else:
        if ConvParam.tiling_query_param["bias_flag"]:
            cub = ConvParam.tensor_map["c_ub"]
            bias_ub = ConvParam.tensor_map["bias_ub"]
        else:
            cub = ConvParam.tensor_map["c_ub"]

        c_col = cub.op.input_tensors[0]

        if ConvParam.tiling_query_param["bias_flag"]:
            mad_shape = c_col.shape
            bias_l0c = tvm.compute(mad_shape,
                                   lambda group, n, c1_opt, m, c0:
                                       bias_ub(group*mad_shape[2]*mad_shape[4] +
                                               c1_opt*mad_shape[4] + c0).astype(c_col.dtype),
                                   name=op_tag + "bias_l0c",
                                   tag=op_tag + "bias_l0c")
            c_col = tvm.compute(mad_shape, lambda *indice:
                                bias_l0c(*indice) + c_col(*indice),
                                name=op_tag + "c_col_bias",
                                tag=op_tag + "c_col_bias")
            ConvParam.tensor_map["c_col_bias"] = c_col
            ConvParam.tensor_map["bias_l0c"] = bias_l0c

        c_ub = tvm.compute(cub.shape,
                           lambda batch, cout1, howo, cout0:
                               c_col(0 if group == 1 else cout1 // cout1_opt,
                                     batch,
                                     cout1 if group == 1 else cout1 % cout1_opt,
                                     howo,
                                     cout0),
                           name='c_ub',
                           tag=op_tag + "c_ub",
                           attrs=cub.op.attrs)
        ConvParam.tensor_map["c_ub"] = c_ub

        if ConvParam.v200_width_out_1_flag:
            removepad_shape = ConvParam.tensor_map["remove_padded_column"].shape
            res_tensor = tvm.compute(removepad_shape,
                                     lambda batch, cout1, howo, cout0:
                                         c_ub(batch, cout1, howo*2, cout0),
                                     name="remove_padded_column",
                                     tag=op_tag + "remove_padded_column",
                                     attrs={"width_out": ConvParam.w_out})
            ConvParam.tensor_map["remove_padded_column"] = res_tensor
            c_ub = res_tensor

        res_c = tvm.compute(conv_res_shape,
                            lambda batch, cout1, howo, cout0:
                                c_ub(batch, cout1, howo, cout0),
                            name='C',
                            tag=op_tag + "C",
                            attrs={"width_out": ConvParam.w_out})
        ConvParam.tensor_map["C"] = res_c
        cast_0_ub = tvm.compute(conv_res_shape,
                                lambda *indice:
                                    res_c(*indice).astype("float16"),
                                name="cast_0_ub")
        cast0 = tvm.compute(conv_res_shape,
                            lambda *indice: cast_0_ub(*indice),
                            name="cast0")
        cast1 = tvm.compute(conv_res_shape,
                            lambda *indice: cast0(*indice).astype("float32"),
                            name="cast1")
        ConvParam.tensor_map["cast1"] = cast1
        mul_0 = vmul(cast1, cast1)

        tuple_reduce = tvm.comm_reducer(_fcombine,
                                        _fidentity,
                                        name='tuple_reduce')
        mean_out, _ = tvm.compute(reduce_shape,
                                  lambda c1, c0:
                                      tuple_reduce((cast1[k_0, c1, k_1, c0],
                                                    mul_0[k_0, c1, k_1, c0]),
                                                   axis=[k_0, k_1]),
                                  name="mean_out")
        outputs = [cast0, mean_out]
    ConvParam.convbn1_flag = True  # used in cce_schedule
    return outputs


class Conv2dBN1Fusion:
    """
    Class of Conv2d + BN1 Fusion
    """
    def __init__(self, conv_param, fmap_dtype, op_graph, cub):
        """
        class Conv2dBN1Fusion init func
        """
        self.flag = conv_param.convbn1_flag
        self.fp32_bn1_flag = False
        if self.flag and fmap_dtype == "float32":
            self.fp32_bn1_flag = True
        self.fusion_type = 20
        self.cub = None
        self.conv_out = None
        self.sum_x_global = None
        self.sum_x_global_pragma_axis = None
        self.res_ub_rf_at_sum_x_axis = None
        if self.flag:
            self.cub = cub
            self.conv_out = self.get_bn1_conv_out_tensor(op_graph)

    def get_bn1_conv_out_tensor(self, op_graph):
        """
        get bn1 conv_out tensor (gm)
        """
        for op in op_graph.body_ops:
            if op['dst_buffer'] == self.cub:
                return op['prev_op'][0]['dst_buffer']
        return None

    def get_info_dict_cdtype(self):
        """
        get info dict cdtype for conv2d+bn1
        """
        if self.fp32_bn1_flag:
            return "float32"
        return "float16"

    def bn1fusion_special_process_pre(self, sch):
        """
        conv2d+bn1 fusion special process pre
        compute_inline res_conv2d in fp32 scenario
        """
        if self.fp32_bn1_flag:
            conv_out_tmp = self.conv_out.op.input_tensors[0]
            sch[conv_out_tmp].compute_inline()

    def set_sum_x_global(self, sum_x_global, sum_x_global_pragma_axis, res_ub_rf_at_sum_x_axis):
        """
        set sum_x_global, sum_x_global_pragma_axis and res_ub_rf_at_sum_x_axis
        """
        self.sum_x_global = sum_x_global
        self.sum_x_global_pragma_axis = sum_x_global_pragma_axis
        self.res_ub_rf_at_sum_x_axis = res_ub_rf_at_sum_x_axis

    def bn1fusion_compute_at(self, sch, res, cub_at_res_axis):
        """
        conv2d+bn1 fusion special compute_at
        """
        if self.flag:
            sch[res].compute_at(sch[self.sum_x_global], self.res_ub_rf_at_sum_x_axis)
            sch[self.conv_out].compute_at(sch[res], cub_at_res_axis)

    def bn1fusion_cub_emit_insn(self, sch):
        """
        conv2d+bn1 fusion cub emit_insn
        """
        sch[self.cub].emit_insn(self.cub.op.axis[0], "dma_copy")

    def bn1fusion_res_emit_insn(self, sch, res, res_pragma_axis, dynamic_flag):
        """
        conv2d+bn1 fusion res emit_insn
        """
        sch[self.sum_x_global].emit_insn(self.sum_x_global_pragma_axis, "dma_copy")
        sch[self.conv_out].emit_insn(self.conv_out.op.axis[0], "dma_copy")
        if dynamic_flag:
            sch[res].emit_insn(res_pragma_axis, "vector_reduce_sum")
        else:
            sch[res].emit_insn(res_pragma_axis, "vector_dichotomy_add_for_bn_reduce")
