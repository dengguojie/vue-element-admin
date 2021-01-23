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

from te.platform.cce_emitinsn_params import cceEmitParamsIns
from tbe.dsl.instrinsic.cce_intrin_md import reset_mask_insn
from te.platform import cce_util


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
    block_idx = vars_list[0]
    if intrin_cmd == "matmul_transpose":
        batch = cceEmitParamsIns.get_param("batch")
        blk_m = cceEmitParamsIns.get_param("matmul_m_blk")
        blk_n = cceEmitParamsIns.get_param("matmul_n_blk")
        if batch > 1:
            var_b = vars_list[1]
            ub_2_gm_batch_matmul_transopose(ir_builder, stmt, block_idx,
                                            batch // 16, var_b)
        else:
            if blk_n == 1:
                block_idx = vars_list[2]
                var_n = vars_list[0]
                var_m = vars_list[3]
            else:
                var_n = vars_list[1]
                var_m = vars_list[4]
            ub_2_gm_matmul_transopose(ir_builder, stmt, block_idx, blk_m,
                                      blk_n, var_n, var_m)
    else:
        raise RuntimeError("Wrong Intrin_cmd.")

    return ir_builder.get()


def ub_2_gm_matmul_transopose(*args):
    """
    ub_2_gm for matmul_confusion_transpose_d
    """
    ir_builder, stmt, block_idx, blk_m, blk_n, var_n, var_m = args
    matmul_m_split = cceEmitParamsIns.get_param("matmul_m_split")
    matmul_n_split = cceEmitParamsIns.get_param("matmul_n_split")
    shape_m = cceEmitParamsIns.get_param("matmul_m")
    shape_n = cceEmitParamsIns.get_param("matmul_n")
    split_m = shape_m // blk_m // matmul_m_split
    split_n = shape_n // blk_n // matmul_n_split
    if shape_m < 16384:
        shape_trans = [16, 4, shape_m // 16 // 32, 32, 16, 16]
    else:
        shape_trans = [16, 4, shape_m // 16 // 8, 8, 16, 16]
    ins, outs, = cce_util.get_buffer(stmt, need_origin_adress=True)
    dtype = outs[0].dtype
    n_burst = shape_trans[0] * shape_trans[1] // split_n // blk_n
    loop = shape_trans[2] // blk_m // split_m
    min_data_move = shape_trans[-1] * shape_trans[-2] * shape_trans[-3]
    if loop == 0:
        dst_offset = ((block_idx % blk_m) * split_m + var_m) // \
                     (blk_m * split_m // shape_trans[2]) * min_data_move * \
            shape_trans[-5] * shape_trans[-6] + \
                     ((block_idx // blk_m) * split_n + var_n) * n_burst * \
            min_data_move + \
                     (((block_idx % blk_m) * split_m + var_m) %
                      (blk_m * split_m // shape_trans[2])) * \
                     (shape_trans[-3] * shape_trans[-4] // blk_m // split_m) * \
            shape_trans[-1] * shape_trans[-2]
        dst_addr = outs[0].access_ptr('w', offset=dst_offset)
        dma_offset = 0
        sid = 0
        len_burst = matmul_m_split * matmul_n_split // n_burst // 16
        src_stride = 0
        dst_stride = (min_data_move -
                      (shape_trans[-3] * shape_trans[-4] // blk_m // split_m) *
                      shape_trans[-1] * shape_trans[-2]) // 16
        if dst_stride == 0:
            len_burst *= n_burst
            n_burst = 1
        ir_builder.emit(
            tvm.call_extern(dtype, "copy_ubuf_to_gm", dst_addr,
                            ins[0].access_ptr("rw", offset=dma_offset),
                            sid, n_burst, len_burst, src_stride, dst_stride))
    else:
        with ir_builder.for_range(0, loop) as num_i:
            dst_offset = (((block_idx % blk_m) * split_m + var_m) * loop +
                          num_i) * min_data_move * shape_trans[-5] * \
                shape_trans[-6] + \
                         ((block_idx // blk_m) * split_n + var_n) * n_burst * \
                min_data_move
            dst_addr = outs[0].access_ptr('w', offset=dst_offset)
            dma_offset = num_i * min_data_move
            sid = 0
            len_burst = matmul_m_split * matmul_n_split // \
                n_burst // loop // 16
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
    ir_builder, stmt, block_idx, batch, var_b = args
    shape_m = cceEmitParamsIns.get_param("matmul_m")
    shape_n = cceEmitParamsIns.get_param("matmul_n")
    if batch == 160 and shape_m == 128 and shape_n == 64:
        ins, outs, = cce_util.get_buffer(stmt, need_origin_adress=False)
        shape_trans = [batch, 16, 4, 8, 16, 16]
        split_blk = shape_trans[0] * shape_trans[1] // 32
        min_data_move = shape_trans[-1] * shape_trans[-2] * shape_trans[-3]
        dst_offset = ((block_idx * split_blk + var_b) % shape_trans[1]) * \
            shape_trans[0] * shape_trans[-4] * min_data_move + \
                     (block_idx * split_blk + var_b) // shape_trans[1] * \
                     min_data_move - block_idx * split_blk * shape_trans[2] * \
            min_data_move - var_b * shape_trans[-4] * min_data_move
        len_burst = min_data_move // 2 // 16
    elif ((batch == 24 and shape_m == 512 and shape_n == 64) or
     (batch == 32 and shape_m == 512 and shape_n == 64)):
        ins, outs, = cce_util.get_buffer(stmt, need_origin_adress=True)
        shape_trans = [batch, 16, 4, 32, 16, 16]
        split_blk = shape_trans[0] * shape_trans[1] // 2 // 32
        min_data_move = shape_trans[-1] * shape_trans[-2] * shape_trans[-3]
        dst_offset = ((block_idx * split_blk + var_b // 2) %
                      8) * 2 * shape_trans[0] * \
            shape_trans[-4] * min_data_move + \
                     (block_idx * split_blk + var_b // 2) // 8 * \
                     min_data_move + (var_b % 2) * min_data_move *\
            shape_trans[0] * shape_trans[-4]
        len_burst = min_data_move // 16
    else:
        raise RuntimeError("This case does not support fusion now.")
    dtype = outs[0].dtype
    dma_offset = 0
    sid = 0
    n_burst = shape_trans[-4]
    src_stride = 0
    dst_stride = shape_trans[0] * min_data_move // 16 - len_burst
    ir_builder.emit(
        tvm.call_extern(dtype, "copy_ubuf_to_gm",
                        outs[0].access_ptr('w', offset=dst_offset),
                        ins[0].access_ptr("rw", offset=dma_offset),
                        sid, n_burst, len_burst, src_stride, dst_stride))
