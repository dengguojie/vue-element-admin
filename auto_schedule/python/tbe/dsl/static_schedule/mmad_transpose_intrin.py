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
matmul confusion_transpose_d intrinsic functions
"""
from te import tvm

from tbe.dsl.instrinsic.cce_emitinsn_params import cceEmitParamsIns
from tbe.dsl.instrinsic.cce_intrin_md import reset_mask_insn
from te.platform import cce_util
from .util import ceil

@tvm.register_func("tvm.intrin.cce.dma_copy_matmul_transpose")
def dma_copy_matmul_transpose(stmt):
    """dma_copy for Matmul + ConfusionTranspose"""
    return dma_copy_mmad_transpose(stmt, "matmul_transpose")


def dma_copy_mmad_transpose(stmt, intrin_cmd):
    """dma_copy for mmad + ConfusionTranspose"""
    ir_builder = tvm.ir_builder.create()
    vars_list = []

    def recursive_get_var(stmt):
        if isinstance(stmt, (tvm.expr.Add, tvm.expr.Mul,
                             tvm.expr.FloorDiv, tvm.expr.FloorMod)):
            recursive_get_var(stmt.a)
            recursive_get_var(stmt.b)
        elif isinstance(stmt, tvm.expr.Var):
            vars_list.append(stmt)
        else:
            pass

    def interpret_statement(stmt):
        recursive_get_var(stmt.index)

    tvm.ir_pass.IRTransform(stmt, None, interpret_statement, ["Store"])
    if intrin_cmd == "matmul_transpose":
        batch = cceEmitParamsIns.get_param("batch")
        if batch > 1:
            ub_2_gm_batch_matmul_transopose(ir_builder, stmt, vars_list, batch)
        else:
            ub_2_gm_matmul_transopose(ir_builder, stmt, vars_list)
    else:
        raise RuntimeError("Wrong Intrin_cmd.")

    return ir_builder.get()


def _get_matmul_axis(vars_list, blk_m, blk_n, split_m, split_n):
    """get matmul outer axis of m, n"""
    if blk_n == 1:
        block_idx = 0
        var_n = vars_list[0] if split_n != 1 else 0
    else:
        block_idx = vars_list[0]
        var_n = vars_list[1] if split_n != 1 else 0

    if split_m == 1:
        var_m = 0
        if blk_m != 1:
            block_idx = vars_list[-4]
    else:
        var_m = vars_list[-4]
        if blk_m != 1:
            block_idx = vars_list[-5]

    return block_idx, var_m, var_n


def _get_batch_matmul_axis(vars_list, batch, split_m, split_n):
    """get batch matmul outer axis of m, n"""
    block_idx = vars_list[0]
    var_b = vars_list[1]
    var_n = 0 if split_n == 1 else vars_list[2]
    var_m = 0 if split_m == 1 else vars_list[-4]

    return block_idx, var_b, var_m, var_n
    

def ub_2_gm_matmul_transopose(*args):
    """
    ub_2_gm for matmul_confusion_transpose_d
    """
    ir_builder, stmt, vars_list = args
    blk_m = cceEmitParamsIns.get_param("matmul_m_blk")
    blk_n = cceEmitParamsIns.get_param("matmul_n_blk")
    # the shape of result after transpose and reshape
    transpose_shape = cceEmitParamsIns.get_param("transpose_shape")
    # the shape of result after reshape
    shape_trans = list(transpose_shape[1:])
    shape_trans.insert(2, transpose_shape[0])

    # the m,n block in cube
    matmul_m_split = cceEmitParamsIns.get_param("matmul_m_split")
    matmul_n_split = cceEmitParamsIns.get_param("matmul_n_split")
    # the shape of m and n
    shape_m = cceEmitParamsIns.get_param("matmul_m")
    shape_n = cceEmitParamsIns.get_param("matmul_n")
    # the out_axis of m and n
    split_m = shape_m // blk_m // matmul_m_split
    split_n = shape_n // blk_n // matmul_n_split

    block_idx, var_m, var_n = _get_matmul_axis(vars_list, blk_m, blk_n, split_m, split_n)

    # the shape of m and n which in block
    shape_m_block = shape_trans[-3] * shape_trans[-4]
    shape_n_block = shape_trans[-5] * shape_trans[-6]

    # get the input and output of dmacopy, where input is ub, output is ddr
    ins, outs, = cce_util.get_buffer(stmt, need_origin_adress=True)
    dtype = outs[0].dtype

    # min burst in m and n
    n_burst = shape_n_block // split_n // blk_n
    m_burst = shape_m_block // split_m // blk_m
    # means times of dmacopy in min burst
    loop = shape_trans[2] // blk_m // split_m

    min_data_move = shape_trans[-1] * shape_trans[-2] * shape_trans[-3]
    mini_block = shape_trans[-1] * shape_trans[-2]
    if loop == 0:
        m_parts = (blk_m * split_m // shape_trans[2])
        m_offset = ((block_idx % blk_m)*split_m + var_m) // m_parts * min_data_move * shape_n_block + \
                    ((block_idx % blk_m)*split_m + var_m) % m_parts * m_burst * mini_block
        n_offset = ((block_idx // blk_m)*split_n + var_n) * n_burst * min_data_move
        dst_offset = m_offset + n_offset

        dst_addr = outs[0].access_ptr('w', offset=dst_offset)
        dma_offset = 0
        sid = 0
        len_burst = matmul_m_split * matmul_n_split // n_burst // 16
        src_stride = 0
        dst_stride = (min_data_move - m_burst * mini_block) // 16
        if dst_stride == 0:
            len_burst *= n_burst
            n_burst = 1

        ir_builder.emit(
            tvm.call_extern(dtype, "copy_ubuf_to_gm", dst_addr,
                            ins[0].access_ptr("rw", offset=dma_offset),
                            sid, n_burst, len_burst, src_stride, dst_stride))
    else:
        with ir_builder.for_range(0, loop) as num_i:
            m_offset = (((block_idx % blk_m)*split_m + var_m)*loop + num_i) * min_data_move * shape_n_block
            n_offset = ((block_idx // blk_m)*split_n + var_n) * n_burst * min_data_move
            dst_offset = m_offset + n_offset
            dst_addr = outs[0].access_ptr('w', offset=dst_offset)
            dma_offset = num_i * min_data_move
            sid = 0
            len_burst = matmul_m_split * matmul_n_split // n_burst // loop // 16
            src_stride = (loop - 1) * min_data_move // 16
            dst_stride = 0
            if src_stride == 0:
                len_burst *= n_burst
                n_burst = 1
            ir_builder.emit(
                tvm.call_extern(dtype, "copy_ubuf_to_gm", dst_addr,
                                ins[0].access_ptr("rw", offset=dma_offset),
                                sid, n_burst, len_burst,
                                src_stride, dst_stride))

    reset_mask_insn(ir_builder, dtype)


def ub_2_gm_batch_matmul_transopose(*args):
    """
    ub_2_gm for batch_matmul_confusion_transpose_d
    """
    ir_builder, stmt, vars_list, batch = args
    # the shape of result after transpose and reshape
    transpose_shape = cceEmitParamsIns.get_param("transpose_shape")
    blk_batch = cceEmitParamsIns.get_param("matmul_batch_blk")

    batch = batch // (transpose_shape[0] // 4)
    ins, outs, = cce_util.get_buffer(stmt, need_origin_adress=True)
    # the shape of m and n
    shape_m = cceEmitParamsIns.get_param("matmul_m")
    shape_n = cceEmitParamsIns.get_param("matmul_n")

    shape_trans = [batch, transpose_shape[0]//4, shape_n//16, shape_m//16, 16, 16]
    # the m,n block in cube
    matmul_m_split = cceEmitParamsIns.get_param("matmul_m_split")
    matmul_n_split = cceEmitParamsIns.get_param("matmul_n_split")
    # the out_axis of m and n
    split_m = shape_m // matmul_m_split
    split_n = shape_n // matmul_n_split
    # min burst in m and n
    m_burst = matmul_m_split // 16
    n_burst = matmul_n_split // 16
    
    block_idx, var_b, var_m, var_n = _get_batch_matmul_axis(vars_list, batch, split_m, split_n)

    min_data_move = shape_trans[-1] * shape_trans[-2] * shape_trans[-3]
    mini_block = shape_trans[-1] * shape_trans[-2]
    split_blk = ceil(shape_trans[0] * shape_trans[1], blk_batch)

    if (shape_m == 128 and shape_n == 64) or \
        (shape_m == 512 and shape_n == 64) or \
        (shape_m == 128 and shape_n == 64):
        batch_offset = ((block_idx*split_blk + var_b) % shape_trans[1]) * \
                        shape_trans[0] * shape_trans[-4] * min_data_move + \
                        ((block_idx*split_blk + var_b) // shape_trans[1]) * min_data_move
        m_offset = var_m * m_burst * mini_block
        n_offset = var_n * n_burst * shape_trans[0] * min_data_move
        dst_offset = batch_offset + m_offset + n_offset
        len_burst = m_burst * mini_block // 16
    else:
        raise RuntimeError("This case does not support fusion now.")
    dtype = outs[0].dtype
    dma_offset = 0
    sid = 0
    src_stride = 0
    dst_stride = shape_trans[0]*min_data_move//16 - len_burst

    ir_builder.emit(
        tvm.call_extern(dtype, "copy_ubuf_to_gm",
                        outs[0].access_ptr('w', offset=dst_offset),
                        ins[0].access_ptr("rw", offset=dma_offset),
                        sid, n_burst, len_burst, src_stride, dst_stride))
