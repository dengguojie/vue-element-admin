"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

transpose
"""

import te.lang.dynamic
from te import tvm
from te.tvm import make as _make
from te.tvm import expr as _expr
from te.tvm import stmt as _stmt
from te.platform.cce_runtime import PIPELINES
from te.platform import cce_params
from te import tik
from te import platform as tbe_platform

UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
TILING_PARAM_BURST_LEN = 84
TILING_PARAM_NUM = TILING_PARAM_BURST_LEN*4
TILING_PARAM_SIZEOF = 8
MAX_INT64_VALUE = 2 ** 64 - 1
BLOCK_SIZE = 32
TRANSPOSE_MAX_AXIS_NUM = 8
RESERVED_UB = 4 # 4KB
ELE_NUM_PER_BLOCK_FP16 = 16
ELE_NUM_PER_BLOCK_FP32 = 8
ELE_NUM_PER_BLOCK_INT64 = 4
BARRIER_INT_LEN = 4
LIST_NUMBER = 16
TILING_OFFSET_FOR_DIRTY_DATA = CORE_NUM * ELE_NUM_PER_BLOCK_INT64


# pylint: disable=unused-argument,invalid-name, too-many-arguments, unused-variable, too-many-locals
# pylint: disable=too-many-statements, invalid-name, no-self-use, protected-access
# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines
class Barrier:
    """this class should be part of tik."""
    def emit(self, tik_inst, stmt):
        """Emit a statement to the end of current scope.

        Parameters
        ----------
        stmt : Stmt or callable.
        The statement to be emitted or callable that build stmt given
        body.
        """
        if isinstance(stmt, _expr.Call):
            stmt = _make.Evaluate(stmt)
        assert isinstance(stmt, _stmt.Stmt) or callable(stmt)
        tik_inst._seq_stack[-1].append(stmt)

    def __init__(self, tik_instance, workspace, block_num, block_id):
        """soft synchronization initialize"""
        self.block_id = block_id
        self.block_num = block_num
        self.int32_byte_size = 4
        self.tik_instance = tik_instance
        self.gm_workspace = workspace
        self.seq = self.tik_instance.Scalar('int64', init_value=1)
        self.sync_ub = tik_instance.Tensor(
            'int64', (self.int32_byte_size * self.block_num,), tik.scope_ubuf, 'barrier_ub')
        zero = self.tik_instance.Scalar('int64')
        zero.set_as(0)
        self.sync_ub[self.block_id * self.int32_byte_size].set_as(zero)
        self.sync_ub[self.block_id * self.int32_byte_size + 1].set_as(zero)
        self.int64_max = 0x7FFFFFFFFFFFFFFF
        self.loop_bound = self.tik_instance.Scalar('int64')
        self.sync_ub[self.block_id * self.int32_byte_size].set_as(zero)
        self.sync_ub[self.block_id * self.int32_byte_size + 1].set_as(zero)
        self.tik_instance.tensor_mov(
            self.gm_workspace[self.block_id * self.int32_byte_size],
            self.sync_ub[self.block_id * self.int32_byte_size], '', 1, 1, 0, 0)
        self.emit(
            self.tik_instance, tvm.call_extern(
                'int32', 'pipe_barrier', tvm.call_pure_intrin('int32', 'tvm_cce_string_print', 'PIPE_MTE3')))

    def sync(self):
        """ barrier sync func"""
        # add pipe_barrier MTE3 here manually
        self.emit(
            self.tik_instance, tvm.call_extern(
                'int32', 'pipe_barrier', tvm.call_pure_intrin('int32', 'tvm_cce_string_print', 'PIPE_MTE3')))
        self.sync_ub[self.block_id * self.int32_byte_size + self.seq % 2].set_as(self.seq)
        self.tik_instance.tensor_mov(
            self.gm_workspace[self.block_id * self.int32_byte_size],
            self.sync_ub[self.block_id * self.int32_byte_size], '', 1, 1, 0, 0)
        self.loop_bound.set_as(self.int64_max)
        pipe_line_dict = dict(zip(PIPELINES.values(), PIPELINES.keys()))
        with self.tik_instance.new_scope():
            self.tik_instance.scope_attr(cce_params.CCE_AXIS, 'group_coproc_scope', pipe_line_dict['PIPE_ALL'])
            with self.tik_instance.for_range(0, self.loop_bound, dtype='int64'):
                self.tik_instance.tensor_mov(self.sync_ub, self.gm_workspace, '', 1, self.block_num, 0, 0)
                # insert set_flag wait by manual
                self.emit(self.tik_instance, tvm.call_intrin(
                    'int32', 'tvm_cce_string_print', 'set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)'))
                synced = self.tik_instance.Scalar('int64', init_value=1)
                # insert wait_flag by manual. IR above all is four.
                self.emit(self.tik_instance, tvm.call_intrin(
                    'int32', 'tvm_cce_string_print', 'wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)'))

                with self.tik_instance.for_range(0, self.block_num, dtype='int64') as core_id:
                    with self.tik_instance.if_scope(
                            self.sync_ub[core_id * self.int32_byte_size + self.seq % 2] != self.seq):
                        synced.set_as(0)

                with self.tik_instance.if_scope(synced == 1):
                    self.loop_bound.set_as(0)

        self.seq.set_as(self.seq + 1)


class Transpose:
    """
    Transpose
    """
    class TilingParam:
        """
        TilingParam
        """
        def __init__(self, tiling_reg_list, ub_tiling, tik_inst):
            """
            get tiling parameters

            0               1               2               3
            0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                      overlap_data_1                           |
            |                      ...                                      |   32 block(32 core)
            |                      overlap_data_n(n=core_num)               |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |    core_num   |  ub_size      | identical     |  nburst       |   1 block
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |  nburst_tail  |  burst_len    | burstlen_tail |  src_stride   |   1 block
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            | align_element | reorder_factor| ub_threshold  |identical_loop |   1 block
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            | fp16_offset_1 | fp16_offset_2 | fp16_offset_3 | cycle_num_wsp |   1 block
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            | loop_num_wsp  | nburst_wsp    |nburst_tail_wsp| by_workspace  |   1 block
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            | last_axis_ele |last_axis_ele_a|str_stride_wsp | dst_stride_wsp|   1 block
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     level_loop_num (8)                        |   2 block
            |                                                               |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     level_gap(8)                              |   2 block
            |                                                               |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     has_tail(8)                               |   2 block
            |                                                               |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     dst_base_addr_1                           |
            |                     ...                                       |   8 block(32 core)
            |                     dst_base_addr_n                           |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     src_base_addr_1                           |
            |                     ...                                       |   8 block(32 core)
            |                     src_base_addr_n                           |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     src_base_addr_wsp_1                       |
            |                     ...                                       |   8 block(32 core)
            |                     src_base_addr_wsp_n                       |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     dst_base_addr_wsp_1                       |
            |                     ...                                       |   8 block(32 core)
            |                     dst_base_addr_wsp_n                       |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            |                     dirty_data_start_addr_1                   |
            |                     ...                                       |   8 block(32 core)
            |                     dirty_data_start_addr_n                   |
            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                                                                               --------
                                                                                84 block
            """


            for i in range(TILING_PARAM_NUM - TILING_OFFSET_FOR_DIRTY_DATA):
                tiling_reg_list[i].set_as(ub_tiling[i + TILING_OFFSET_FOR_DIRTY_DATA])

            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.identical = tiling_reg_list[2]
            self.nburst = tiling_reg_list[3]
            self.nburst_tail = tiling_reg_list[4]
            self.burst_len = tiling_reg_list[5]
            self.burstlen_tail = tiling_reg_list[6]
            self.src_stride = tiling_reg_list[7]
            self.align_element = tiling_reg_list[8]
            self.reorder_factor= tiling_reg_list[9]
            self.ub_threshold = tiling_reg_list[10]
            self.identical_loop = tiling_reg_list[11]
            self.fp16_offset_1 = tiling_reg_list[12]
            self.fp16_offset_2 = tiling_reg_list[13]
            self.fp16_offset_3 = tiling_reg_list[14]
            self.cycle_num_wsp = tiling_reg_list[15]
            self.loop_num_wsp = tiling_reg_list[16]
            self.nburst_wsp = tiling_reg_list[17]
            self.nburst_tail_wsp = tiling_reg_list[18]
            self.by_workspace = tiling_reg_list[19]
            self.last_axis_ele = tiling_reg_list[20]
            self.last_axis_ele_a = tiling_reg_list[21]
            self.src_stride_wsp = tiling_reg_list[22]
            self.dst_stride_wsp = tiling_reg_list[23]
            self.count= 24

            self.level_loop_num = []
            self.level_gap = []
            self.has_tail = []

            # case_3
            self.case = tik_inst.Scalar('int64', init_value=3)
            self.src_axis_num = tik_inst.Scalar('int64', init_value=3)
            self.dst_axis_num = tik_inst.Scalar('int64', init_value=1)
            tiling_reg_list[100].set_as(8)
            tiling_reg_list[101].set_as(128)
            tiling_reg_list[102].set_as(4)
            tiling_reg_list[103].set_as(224)
            tiling_reg_list[104].set_as(1792)
            tiling_reg_list[105].set_as(229376)
            tiling_reg_list[106].set_as(0)
            tiling_reg_list[107].set_as(0)
            tiling_reg_list[108].set_as(0)
            tiling_reg_list[109].set_as(0)
            tiling_reg_list[110].set_as(0)
            tiling_reg_list[111].set_as(0)
            #[2]:0,  [1]:0,   [0]:0
            self.src_jump_counter_0 = tiling_reg_list[109]
            self.src_jump_counter_1 = tiling_reg_list[110]
            self.src_jump_counter_2 = tiling_reg_list[111]
            #[2]:4,  [1]:128, [0]:8
            self.src_jump_factor_0 = tiling_reg_list[100]
            self.src_jump_factor_1 = tiling_reg_list[101]
            self.src_jump_factor_2 = tiling_reg_list[102]
            #[2]:229376,  [1]:1792,  [0]:224
            self.src_jump_stride_0 = tiling_reg_list[103]
            self.src_jump_stride_1 = tiling_reg_list[104]
            self.src_jump_stride_2 = tiling_reg_list[105]
            #[0]:0
            self.dst_jump_factor_0 = tiling_reg_list[106]
            #[1]:0,   [0]:0
            self.dst_jump_stride_0 = tiling_reg_list[107]
            self.dst_jump_stride_1 = tiling_reg_list[108]
            self.dst_jump_stride_2 = tiling_reg_list[0]#unused unused

            #case_3_1_begin
            self.line_per_batch = tik_inst.Scalar('int64', init_value=128)
            self.line_tail_batch = tik_inst.Scalar('int64', init_value=0)
            self.line_block_per_batch = tik_inst.Scalar('int64', init_value=16)
            self.line_block_tail_batch = tik_inst.Scalar('int64', init_value=0)
            self.round_num_per_core = tik_inst.Scalar('int64', init_value=1) # 32 core
            self.batch_num_per_round = tik_inst.Scalar('int64', init_value=1)
            self.back_step_up = tik_inst.Scalar('int64', init_value=0)
            self.back_step_left = tik_inst.Scalar('int64', init_value=0)
            self.loop_per_line = tik_inst.Scalar('int64', init_value=1)
            self.col_ele_per_batch = tik_inst.Scalar('int64', init_value=224)
            self.col_block_per_batch = tik_inst.Scalar('int64', init_value=28)
            self.col_ele_tail = tik_inst.Scalar('int64', init_value=0)
            self.col_block_tail = tik_inst.Scalar('int64', init_value=0)
            self.element_per_core = tik_inst.Scalar('int64', init_value=0)
            self.used_core = tik_inst.Scalar('int64', init_value=32)
            self.pad_ele_num_major = tik_inst.Scalar('int64', init_value=0)
            self.pad_ele_num_tail_col = tik_inst.Scalar('int64', init_value=0)
            self.major_col_major_batch = 1
            self.tail_col_major_batch = 0
            self.major_col_tail_batch = 0
            self.tail_col_tail_batch = 0

            self.axis_7_src_stride = tik_inst.Scalar('int64', init_value=0)
            self.axis_6_src_stride = tik_inst.Scalar('int64', init_value=0)
            self.axis_5_src_stride = tik_inst.Scalar('int64', init_value=0)
            self.axis_4_src_stride = tik_inst.Scalar('int64', init_value=896000)
            self.axis_3_src_stride = tik_inst.Scalar('int64', init_value=128000)
            self.axis_2_src_stride = tik_inst.Scalar('int64', init_value=1280)
            self.axis_1_src_stride = tik_inst.Scalar('int64', init_value=0)
            self.axis_0_src_stride = tik_inst.Scalar('int64', init_value=0)

            self.axis_7_src_jump_factor = tik_inst.Scalar('int64', init_value=0)
            self.axis_6_src_jump_factor = tik_inst.Scalar('int64', init_value=0)
            self.axis_5_src_jump_factor = tik_inst.Scalar('int64', init_value=0)
            self.axis_4_src_jump_factor = tik_inst.Scalar('int64', init_value=3)
            self.axis_3_src_jump_factor = tik_inst.Scalar('int64', init_value=7)
            self.axis_2_src_jump_factor = tik_inst.Scalar('int64', init_value=100)
            self.axis_1_src_jump_factor = tik_inst.Scalar('int64', init_value=0)
            self.axis_0_src_jump_factor = tik_inst.Scalar('int64', init_value=0)

            self.axis_3_dst_jump_factor = tik_inst.Scalar('int64', init_value=21)
            self.axis_2_dst_jump_factor = tik_inst.Scalar('int64', init_value=5)
            self.axis_1_dst_jump_factor = tik_inst.Scalar('int64', init_value=0)
            self.axis_0_dst_jump_factor = tik_inst.Scalar('int64', init_value=0)

            self.axis_3_dst_stride = tik_inst.Scalar('int64', init_value=21)
            self.axis_2_dst_stride = tik_inst.Scalar('int64', init_value=2100)
            self.axis_1_dst_stride = tik_inst.Scalar('int64', init_value=10500)
            self.axis_0_dst_stride = tik_inst.Scalar('int64', init_value=0)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.level_loop_num.append(tiling_reg_list[self.count+ i])
            self.count += TRANSPOSE_MAX_AXIS_NUM

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.level_gap.append(tiling_reg_list[self.count + i])
            self.count += TRANSPOSE_MAX_AXIS_NUM

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.has_tail.append(tiling_reg_list[self.count+ i])
            self.count += TRANSPOSE_MAX_AXIS_NUM

            self.dst_addr_offset = self.count + TILING_OFFSET_FOR_DIRTY_DATA
            self.src_addr_offset = self.dst_addr_offset + self.core_num
            self.src_addr_wsp_offset = self.src_addr_offset + self.core_num
            self.dst_addr_wsp_offset = self.src_addr_wsp_offset + self.core_num
            self.dirty_data_offset = self.dst_addr_wsp_offset + self.core_num


    def __init__(self, tik_inst, x_dtype, tensor_list, kernel_name):
        self.tik_inst = tik_inst
        self.x_dtype = x_dtype
        self.kernel_name = kernel_name
        self.data_in, self.data_perm, self.data_out, self.data_workspace, self.data_tiling = tensor_list
        self.ub_tiling = tik_inst.Tensor("int64", (TILING_PARAM_NUM, ), tik.scope_ubuf, "ub_tiling")
        self.barrier_workspace = tik_inst.Tensor("int64", (BARRIER_INT_LEN * CORE_NUM,), tik.scope_gm,
                                                 "barrier_workspace", is_workspace=True, is_atomic_add=True)
        self.tiling_reg_list = [tik_inst.Scalar("int64") for i in range(TILING_PARAM_NUM)]
        tik_inst.data_move(self.ub_tiling, self.data_tiling,  0, 1, TILING_PARAM_BURST_LEN, 0, 0)
        self.tiling_param = self.TilingParam(self.tiling_reg_list, self.ub_tiling, tik_inst)
        self.ub_size = self._get_ub_size_by_dtype()
        self.element_per_block = self._element_per_block(self.x_dtype)
        self.align_ele_in_fp16 = self.tiling_param.align_element * self._sizeof_dtype("float16")
        self.fp16_times = 2 # fp32/int32:2  fp16/int16:1
        self.element_num_per_block = BLOCK_SIZE // self._sizeof_dtype(x_dtype)

        self.axis_7_jump_counter = self.tik_inst.Scalar('int64', init_value=0)
        self.axis_6_jump_counter = self.tik_inst.Scalar('int64', init_value=0)
        self.axis_5_jump_counter = self.tik_inst.Scalar('int64', init_value=0)
        self.axis_4_jump_counter = self.tik_inst.Scalar('int64', init_value=0)
        self.axis_3_jump_counter = self.tik_inst.Scalar('int64', init_value=0)
        self.axis_2_jump_counter = self.tik_inst.Scalar('int64', init_value=0)
        self.axis_1_jump_counter = self.tik_inst.Scalar('int64', init_value=0)
        self.axis_0_jump_counter = self.tik_inst.Scalar('int64', init_value=0)

    def _sizeof_dtype(self, dtype):
        if dtype in ("int8", "uint8"):
            return 1
        if dtype in ("float16", "int16", "uint16"):
            return 2
        if dtype in ("float", "float32", "int32", "uint32"):
            return 4
        return 8

    def _element_per_block(self, dtype):
        if dtype in ("int8", "uint8"):
            return 32
        if dtype in ("float16", "int16", "uint16"):
            return 16
        if dtype in ("float", "float32", "int32", "uint32"):
            return 8
        return 4

    def _get_ub_size_by_dtype(self):
        return (UB_SIZE - RESERVED_UB * 2048) // self._sizeof_dtype(self.x_dtype)

    def _copy_input_2_workspace2(self, ub_input):
        with self.tik_inst.for_range(0, 8) as i:
            with self.tik_inst.for_range(0, 480) as j:
                self.tik_inst.data_move(ub_input, self.data_in[j*1000*33 + i*33], 0, 1000, 5, 28, 0)
                self.tik_inst.data_move(self.data_workspace[j*1000*40 + i*40], ub_input, 0, 1000, 5, 0, 35)

    def _copy_workspace_2_out(self, ub_input):
        with self.tik_inst.for_range(0, 15002) as i:
            self.tik_inst.data_move(ub_input, self.data_workspace[i * 40], 0, 1, 5, 0, 0)
            self.tik_inst.data_move(self.data_out[i * 40], ub_input, 0, 1, 5, 0, 0)

    #       cycle_num_wsp: 4
    #                loop_0       loop_1    tail
    #             ------------|------------|----
    #    cycle_0  -   -   -   |-   -   -   |-
    #    cycle_1   -   -   -  | -   -   -  | -
    #    cycle_2    -   -   - |  -   -   - |  -
    #    cycle_3     -   -   -|   -   -   -|   -

    # pylint: disable=too-many-arguments, unused-argument, invalid-name
    def _copy_input_2_workspace(self, ub_input, src_pos_wsp, dst_pos_wsp):
        tp = self.tiling_param
        with self.tik_inst.for_range(0, tp.cycle_num_wsp) as i:
            with self.tik_inst.for_range(0, tp.loop_num_wsp) as j:
                self.tik_inst.data_move(ub_input,
                                        self.data_in[src_pos_wsp + \
                                                     j * tp.nburst_wsp * tp.last_axis_ele * self.element_per_block + \
                                                     i * tp.last_axis_ele],
                                        0,
                                        tp.nburst_wsp,
                                        tp.burst_len,
                                        tp.src_stride_wsp,
                                        0)
                self.tik_inst.data_move(self.data_workspace[dst_pos_wsp + \
                                                            j * tp.nburst_wsp * tp.last_axis_ele_a * self.element_per_block + \
                                                            i * tp.last_axis_ele_a],
                                        ub_input,
                                        0,
                                        tp.nburst_wsp,
                                        tp.burst_len,
                                        0,
                                        tp.dst_stride_wsp)
            with self.tik_inst.if_scope(tp.nburst_tail_wsp != 0):
                self.tik_inst.data_move(ub_input,
                                        self.data_in[src_pos_wsp + \
                                                     tp.loop_num_wsp * tp.nburst_wsp * tp.last_axis_ele * \
                                                     self.element_per_block + \
                                                     i * tp.last_axis_ele],
                                        0,
                                        tp.nburst_tail_wsp,
                                        tp.burst_len,
                                        tp.src_stride_wsp,
                                        0)
                self.tik_inst.data_move(self.data_workspace[dst_pos_wsp + \
                                                            tp.loop_num_wsp * tp.nburst_wsp * tp.last_axis_ele_a * \
                                                            self.element_per_block + \
                                                            i * tp.last_axis_ele_a],
                                        ub_input,
                                        0,
                                        tp.nburst_tail_wsp,
                                        tp.burst_len,
                                        0,
                                        tp.dst_stride_wsp)

    # pylint: disable=too-many-arguments, unused-argument, invalid-name
    def _reorder(self, dst_pos, ub_input, ub_offset, ub_offset_exclude_pad, dst_pos_offset_elements_fp16):
        # step1. make all elements in the first col
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")
        src_ele_num_in_fp16 = ub_offset * ELE_NUM_PER_BLOCK_FP16
        src_list = [ub_input_fp16[src_ele_num_in_fp16 * i] for i in range(ELE_NUM_PER_BLOCK_FP16)]
        dst_list_intermediate = [ub_input_fp16[self.tiling_param.fp16_offset_1 + ELE_NUM_PER_BLOCK_FP16 * i] \
                                 for i in range(ELE_NUM_PER_BLOCK_FP16)]
        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate, src_list, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate,
                                    src_list, ub_offset, ELE_NUM_PER_BLOCK_FP16, 1)

        # step2. erase unused elements aligned
        all_line_number = self.tiling_param.burst_len * ELE_NUM_PER_BLOCK_FP16
        pad_line_number = self.align_ele_in_fp16
        nburst = ub_offset // self.tiling_param.burst_len
        burst_len = all_line_number - pad_line_number
        dst_pos_offset_elements_fp16.set_as(nburst * burst_len)
        self.tik_inst.data_move(ub_input_fp16[self.tiling_param.fp16_offset_2],
                                ub_input_fp16[self.tiling_param.fp16_offset_1],
                                0, nburst, burst_len, pad_line_number, 0)

        # step3. make all elements in the first col be in memory of contiguous
        # block numbers after erasing  elements aligned
        ub_offset_exclude_pad.set_as((nburst * burst_len + ELE_NUM_PER_BLOCK_FP16 - 1) // ELE_NUM_PER_BLOCK_FP16)
        src_ele_num_in_fp16_exclude_pad = ub_offset_exclude_pad * ELE_NUM_PER_BLOCK_FP16
        src_list_intermediate = [ub_input_fp16[self.tiling_param.fp16_offset_2 + \
                                               ELE_NUM_PER_BLOCK_FP16 * i] for i in range(ELE_NUM_PER_BLOCK_FP16)]
        dst_list_finally = [ub_input_fp16[src_ele_num_in_fp16_exclude_pad * i] for i in range(ELE_NUM_PER_BLOCK_FP16)]

        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate,
                                    ub_offset_exclude_pad, 1, ELE_NUM_PER_BLOCK_FP16)

    def _save_ub_2_gm(self, dst_pos, ub_input, ub_offset):
        with self.tik_inst.if_scope(self.tiling_param.reorder_factor != 1): # 33 means need reorder
            ub_offset_exclude_pad = self.tik_inst.Scalar("int32") # unit : block
            dst_pos_offset_elements_fp16 = self.tik_inst.Scalar("int32")
            self._reorder(dst_pos, ub_input, ub_offset, ub_offset_exclude_pad, dst_pos_offset_elements_fp16)
            self.tik_inst.data_move(self.data_out[dst_pos], ub_input, 0, 1, ub_offset_exclude_pad, 0, 0)
            dst_pos.set_as(dst_pos + dst_pos_offset_elements_fp16//2)
        with self.tik_inst.if_scope(self.tiling_param.reorder_factor == 1):
            ub_offset_dup = self.tik_inst.Scalar('int64')
            ub_offset_dup.set_as(ub_offset)
            self.tik_inst.data_move(self.data_out[dst_pos], ub_input, 0, 1, ub_offset_dup, 0, 0)
            dst_pos.set_as(dst_pos + ub_offset * self._element_num_per_block())
        ub_offset.set_as(0)

    def _dump_tail(self, level_id, dst_pos, ub_input, ub_offset):
        with self.tik_inst.if_scope(self.tiling_param.has_tail[level_id] != 0):
            with self.tik_inst.if_scope(self.tiling_param.reorder_factor != 1):# 33 means need reorder
                ub_offset_exclude_pad = self.tik_inst.Scalar("int32") # unit : block
                dst_pos_offset_elements_fp16 = self.tik_inst.Scalar("int32")
                self._reorder(dst_pos, ub_input, ub_offset, ub_offset_exclude_pad, dst_pos_offset_elements_fp16)
                self.tik_inst.data_move(self.data_out[dst_pos], ub_input, 0, 1, ub_offset_exclude_pad, 0, 0)
                dst_pos.set_as(dst_pos + dst_pos_offset_elements_fp16//2)
            with self.tik_inst.if_scope(self.tiling_param.reorder_factor == 1):
                self.tik_inst.data_move(self.data_out[dst_pos], ub_input, 0, 1, ub_offset, 0, 0)
                dst_pos.set_as(dst_pos + ub_offset * self._element_num_per_block())
            ub_offset.set_as(0)

    def _correct_dirty_data(self, dirty_data_start_addr, block_idx, ub_input):
        self.tik_inst.data_move(self.data_out[dirty_data_start_addr],
                                ub_input[self.tiling_param.fp16_offset_3//2 + \
                                         block_idx * self._element_num_per_block()],
                                0, 1, 1, 0, 0)

    def _element_num_per_block(self):
        return BLOCK_SIZE // self._sizeof_dtype(self.x_dtype)

    def _do_shape_identical_copy(self, ub_input):
        tp = self.tiling_param
        with self.tik_inst.for_range(0, tp.identical_loop) as i:
            self.tik_inst.data_move(ub_input, self.data_in[i * tp.burst_len * self.element_per_block],
                                    0, 1, tp.burst_len, 0, 0)
            self.tik_inst.data_move(self.data_out[i * tp.burst_len * self.element_per_block],
                                    ub_input, 0, 1, tp.burst_len, 0, 0)
        with self.tik_inst.if_scope(tp.burstlen_tail != 0):
            self.tik_inst.data_move(ub_input, self.data_in[tp.identical_loop * tp.burst_len * self.element_per_block],
                                    0, 1, tp.burstlen_tail, 0, 0)
            self.tik_inst.data_move(self.data_out[tp.identical_loop * tp.burst_len * self.element_per_block],
                                    ub_input, 0, 1, tp.burstlen_tail, 0, 0)

    def _move_data(self):
        with self.tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
            with self.tik_inst.if_scope(block_idx < CORE_NUM):

                ub_offset = self.tik_inst.Scalar("int32") # unit : block
                ub_offset.set_as(0)

                dst_pos = self.tik_inst.Scalar("int64")
                dst_pos.set_as(self.ub_tiling[self.tiling_param.dst_addr_offset + block_idx])

                src_pos_base = self.tik_inst.Scalar("int64")
                src_pos_base.set_as(self.ub_tiling[self.tiling_param.src_addr_offset + block_idx])

                src_pos_wsp = self.tik_inst.Scalar("int64")
                src_pos_wsp.set_as(self.ub_tiling[self.tiling_param.src_addr_wsp_offset + block_idx])

                dst_pos_wsp = self.tik_inst.Scalar("int64")
                dst_pos_wsp.set_as(self.ub_tiling[self.tiling_param.dst_addr_wsp_offset + block_idx])

                dirty_data_start_addr = self.tik_inst.Scalar("int64")
                with self.tik_inst.if_scope(self.tiling_param.align_element != 0):
                    with self.tik_inst.if_scope(block_idx < CORE_NUM - 1):
                        dirty_data_start_addr.set_as(self.ub_tiling[self.tiling_param.dirty_data_offset + block_idx])

                src_pos_7 = self.tik_inst.Scalar("int64")
                src_pos_6 = self.tik_inst.Scalar("int64")
                src_pos_5 = self.tik_inst.Scalar("int64")
                src_pos_4 = self.tik_inst.Scalar("int64")
                src_pos_3 = self.tik_inst.Scalar("int64")
                src_pos_2 = self.tik_inst.Scalar("int64")
                src_pos_1 = self.tik_inst.Scalar("int64")

                # before use ub_input ,all used tiling_param should have been read.
                ub_input = self.tik_inst.Tensor(self.data_in.dtype, (self.ub_size,), tik.scope_ubuf, "ub_input")
                # store recovery data for dirty data
                ub_input_x = self.ub_tiling.reinterpret_cast_to(self.data_in.dtype)
                self.tik_inst.data_move(ub_input[self.tiling_param.fp16_offset_3//2], ub_input_x, 0, 1, CORE_NUM, 0, 0)
                tp = self.tiling_param

                with self.tik_inst.if_scope(tp.identical == 1):
                    with self.tik_inst.if_scope(block_idx == 0):
                        self._do_shape_identical_copy(ub_input)
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tp.by_workspace != 0):
                        barrier = Barrier(self.tik_inst, self.barrier_workspace, CORE_NUM, block_idx)
                        self._copy_input_2_workspace(ub_input, src_pos_wsp, dst_pos_wsp)
                        barrier.sync()

                    with self.tik_inst.for_range(0, tp.level_loop_num[0]) as level_7:
                        src_pos_7.set_as(src_pos_base +  level_7 * tp.level_gap[0])
                        with self.tik_inst.for_range(0, tp.level_loop_num[1]) as level_6:
                            src_pos_6.set_as(src_pos_7 +  level_6 * tp.level_gap[1])
                            with self.tik_inst.for_range(0, tp.level_loop_num[2]) as level_5:
                                src_pos_5.set_as(src_pos_6 +  level_5 * tp.level_gap[2])
                                with self.tik_inst.for_range(0, tp.level_loop_num[3]) as level_4:
                                    src_pos_4.set_as(src_pos_5 +  level_4 * tp.level_gap[3])
                                    with self.tik_inst.for_range(0, tp.level_loop_num[4]) as level_3:
                                        src_pos_3.set_as(src_pos_4 +  level_3 * tp.level_gap[4])
                                        with self.tik_inst.for_range(0, tp.level_loop_num[5]) as level_2:
                                            src_pos_2.set_as(src_pos_3 + level_2 * tp.level_gap[5])
                                            with self.tik_inst.for_range(0, tp.level_loop_num[6]) as level_1:
                                                src_pos_1.set_as(src_pos_2 + level_1 * tp.level_gap[6])
                                                with self.tik_inst.if_scope(tp.by_workspace != 0):
                                                    self.tik_inst.data_move(ub_input[ub_offset * 8],
                                                                            self.data_workspace[src_pos_1],
                                                                            0,
                                                                            tp.nburst,
                                                                            tp.burst_len,
                                                                            tp.src_stride,
                                                                            0)
                                                with self.tik_inst.if_scope(tp.by_workspace == 0):
                                                    self.tik_inst.data_move(ub_input[ub_offset * 8],
                                                                            self.data_in[src_pos_1],
                                                                            0,
                                                                            tp.nburst,
                                                                            tp.burst_len,
                                                                            tp.src_stride,
                                                                            0)
                                                ub_offset.set_as(ub_offset + tp.nburst * tp.burst_len)
                                                with self.tik_inst.if_scope(tp.ub_threshold == ub_offset):
                                                    self._save_ub_2_gm(dst_pos, ub_input, ub_offset)
                                            with self.tik_inst.if_scope(tp.nburst_tail != 0):
                                                src_pos_1.set_as(src_pos_2 + tp.level_loop_num[6] * tp.level_gap[6])
                                                with self.tik_inst.if_scope(tp.by_workspace != 0):
                                                    self.tik_inst.data_move(ub_input,
                                                                            self.data_workspace[src_pos_1],
                                                                            0,
                                                                            tp.nburst_tail,
                                                                            tp.burst_len,
                                                                            tp.src_stride,
                                                                            0)
                                                with self.tik_inst.if_scope(tp.by_workspace == 0):
                                                    self.tik_inst.data_move(ub_input,
                                                                            self.data_in[src_pos_1],
                                                                            0,
                                                                            tp.nburst_tail,
                                                                            tp.burst_len,
                                                                            tp.src_stride,
                                                                            0)
                                                ub_offset.set_as(tp.nburst_tail * tp.burst_len)
                                            self._dump_tail(6, dst_pos, ub_input, ub_offset)#level_1
                                        self._dump_tail(5, dst_pos, ub_input, ub_offset)#level_2
                                    self._dump_tail(4, dst_pos, ub_input, ub_offset)#level_3
                                self._dump_tail(3, dst_pos, ub_input, ub_offset)#level_4
                            self._dump_tail(2, dst_pos, ub_input, ub_offset)#level_5
                        self._dump_tail(1, dst_pos, ub_input, ub_offset)#level_6
                    self._dump_tail(0, dst_pos, ub_input, ub_offset)#level_7

                    with self.tik_inst.if_scope(self.tiling_param.align_element != 0):
                        with self.tik_inst.if_scope(block_idx < CORE_NUM - 1):
                            # Last core do not need to correct dirt data
                            self._correct_dirty_data(dirty_data_start_addr, block_idx, ub_input)

    def _init_jump_param(self, block_idx):
        with self.tik_inst.if_scope(self.tiling_param.case == 3):
            self.tiling_param.src_jump_counter_2.set_as(block_idx*128%4)
            self.tiling_param.src_jump_counter_1.set_as(((block_idx*128)//4)%128)
            self.tiling_param.src_jump_counter_0.set_as(((block_idx*128)//(4*128))%8)
        with self.tik_inst.else_scope():
            self.tiling_param.src_jump_counter_2.set_as(0)
            self.tiling_param.src_jump_counter_1.set_as(0)
            self.tiling_param.src_jump_counter_0.set_as(0)

    def _update_jump_param(self):
        tp = self.tiling_param

        # with self.tik_inst.if_scope(tp.src_axis_num == 4):
        #    with self.tik_inst.if_scope(tp.src_jump_counter_3 == tp.src_jump_factor_3 - 1):
        #        tp.src_jump_counter_3.set_as(0)
        #        with self.tik_inst.if_scope(tp.src_jump_counter_2 == tp.src_jump_factor_2 - 1):
        #            tp.src_jump_counter_2.set_as(0)
        #            with self.tik_inst.if_scope(tp.src_jump_counter_1 == tp.src_jump_factor_1 -1):
        #                tp.src_jump_counter_1.set_as(0)
        #                tp.src_jump_counter_0.set_as(tp.src_jump_counter_0 + 1)
        #            with self.tik_inst.else_scope():
        #                tp.src_jump_counter_1.set_as(tp.src_jump_counter_1 + 1)
        #        with self.tik_inst.else_scope():
        #            tp.src_jump_counter_2.set_as(tp.src_jump_counter_2 + 1)
        #    with self.tik_inst.else_scope():
        #            tp.src_jump_counter_3.set_as(tp.src_jump_counter_3 + 1)

        with self.tik_inst.if_scope(tp.src_axis_num == 3):
            with self.tik_inst.if_scope(tp.src_jump_counter_2 == tp.src_jump_factor_2 - 1):
                tp.src_jump_counter_2.set_as(0)
                with self.tik_inst.if_scope(tp.src_jump_counter_1 == tp.src_jump_factor_1 -1):
                    tp.src_jump_counter_1.set_as(0)
                    tp.src_jump_counter_0.set_as(tp.src_jump_counter_0 + 1)
                with self.tik_inst.else_scope():
                    tp.src_jump_counter_1.set_as(tp.src_jump_counter_1 + 1)
            with self.tik_inst.else_scope():
                tp.src_jump_counter_2.set_as(tp.src_jump_counter_2 + 1)

        with self.tik_inst.if_scope(tp.src_axis_num == 2):
            with self.tik_inst.if_scope(tp.src_jump_counter_1 == tp.src_jump_factor_1 - 1):
                tp.src_jump_counter_1.set_as(0)
                with self.tik_inst.if_scope(tp.src_jump_counter_0 == tp.src_jump_factor_0 -1):
                    tp.src_jump_counter_0.set_as(0)
                with self.tik_inst.else_scope():
                    tp.src_jump_counter_0.set_as(tp.src_jump_counter_0 + 1)
            with self.tik_inst.else_scope():
                tp.src_jump_counter_1.set_as(tp.src_jump_counter_1 + 1)

        with self.tik_inst.if_scope(tp.src_axis_num == 1):
            with self.tik_inst.if_scope(tp.src_jump_counter_0 == tp.src_jump_factor_0 - 1):
                tp.src_jump_counter_0.set_as(0)
            with self.tik_inst.else_scope():
                tp.src_jump_counter_0.set_as(tp.src_jump_counter_0 + 1)

    def _update_jump_param_batch_tail(self):
        with self.tik_inst.if_scope(self.tiling_param.case == 1):
            self.tiling_param.src_jump_counter_2.set_as(1)
            self.tiling_param.src_jump_counter_1.set_as(4)
            self.tiling_param.src_jump_counter_0.set_as(99)
        with self.tik_inst.if_scope(self.tiling_param.case == 2):
            self.tiling_param.src_jump_counter_1.set_as(1)
            self.tiling_param.src_jump_counter_0.set_as(4)

    def _get_src_addr(self, block_idx, loop_on_col=0, round_num=0, ELEMENT_PER_CORE=0):
        tp = self.tiling_param
        src_addr = self.tik_inst.Scalar("int64", init_value=0)
        with self.tik_inst.if_scope(tp.case == 1):
            with self.tik_inst.if_scope(tp.src_axis_num == 3):
                src_addr.set_as(block_idx * ELEMENT_PER_CORE + \
                                loop_on_col * tp.col_ele_per_batch + \
                                tp.src_jump_counter_2 * tp.src_jump_stride_2 + \
                                tp.src_jump_counter_1 * tp.src_jump_stride_1 + \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)
            with self.tik_inst.if_scope(tp.src_axis_num == 2):
                src_addr.set_as(block_idx * ELEMENT_PER_CORE + \
                                loop_on_col * tp.col_ele_per_batch + \
                                tp.src_jump_counter_1 * tp.src_jump_stride_1 + \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)
            with self.tik_inst.if_scope(tp.src_axis_num == 1):
                src_addr.set_as(block_idx * ELEMENT_PER_CORE + \
                                loop_on_col * tp.col_ele_per_batch + \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)

        with self.tik_inst.if_scope(tp.case == 2):
            with self.tik_inst.if_scope(block_idx <= 12):
                src_addr.set_as(block_idx * 3840  + round_num * 256 + \
                                loop_on_col * tp.col_ele_per_batch+ \
                                tp.src_jump_counter_1 * tp.src_jump_stride_1+ \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)
            with self.tik_inst.else_scope():
                src_addr.set_as(46080+(block_idx-12)*4096 + round_num*256 + \
                                loop_on_col * tp.col_ele_per_batch+ \
                                tp.src_jump_counter_1 * tp.src_jump_stride_1+ \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)

        with self.tik_inst.if_scope(tp.case == 3):
            with self.tik_inst.if_scope(tp.src_axis_num == 3):
                src_addr.set_as(loop_on_col * tp.col_ele_per_batch + \
                                tp.src_jump_counter_2 * tp.src_jump_stride_2 + \
                                tp.src_jump_counter_1 * tp.src_jump_stride_1 + \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)

        return src_addr

    def _get_src_addr_tail_col(self, block_idx, ELEMENT_PER_CORE):
        tp = self.tiling_param
        src_addr = self.tik_inst.Scalar("int64", init_value=0)
        with self.tik_inst.if_scope(tp.case == 1):
            with self.tik_inst.if_scope(tp.src_axis_num == 3):
                src_addr.set_as(block_idx * ELEMENT_PER_CORE + tp.loop_per_line * tp.col_ele_per_batch + \
                                tp.src_jump_counter_2 * tp.src_jump_stride_2 + \
                                tp.src_jump_counter_1 * tp.src_jump_stride_1 + \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)
            with self.tik_inst.if_scope(tp.src_axis_num == 2):
                src_addr.set_as(block_idx * ELEMENT_PER_CORE + tp.loop_per_line * tp.col_ele_per_batch + \
                                tp.src_jump_counter_1 * tp.src_jump_stride_1 + \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)
            with self.tik_inst.if_scope(tp.src_axis_num == 1):
                src_addr.set_as(block_idx * ELEMENT_PER_CORE + tp.loop_per_line * tp.col_ele_per_batch + \
                                tp.src_jump_counter_0 * tp.src_jump_stride_0)

        return src_addr

    def _get_dst_addr(self, block_idx, round_num, batch_num, loop_on_col, col_id, back_step_up):
        tp = self.tiling_param
        dst_addr = self.tik_inst.Scalar("int64", init_value=0)
        with self.tik_inst.if_scope(tp.case == 1):
            dst_addr.set_as(((block_idx * 1 + round_num) % tp.dst_jump_factor_0) * tp.dst_jump_stride_1 + \
                            (loop_on_col * tp.col_ele_per_batch + col_id) * tp.dst_jump_stride_0+ \
                            batch_num * tp.line_per_batch - back_step_up)

        with self.tik_inst.if_scope(tp.case == 2):
            with self.tik_inst.if_scope(block_idx <= 12):
                dst_addr.set_as(((block_idx*15+round_num)//tp.dst_jump_factor_0)*tp.dst_jump_stride_2 + \
                                ((block_idx*15+round_num)%tp.dst_jump_factor_0)*tp.dst_jump_stride_1 + \
                                (loop_on_col * tp.col_ele_per_batch + col_id) * tp.dst_jump_stride_0 + \
                                batch_num * tp.line_per_batch - back_step_up)
            with self.tik_inst.else_scope():
                dst_addr.set_as(((180+(block_idx-12)*16+round_num)//tp.dst_jump_factor_0)*tp.dst_jump_stride_2 + \
                                ((180+(block_idx-12)*16+round_num)%tp.dst_jump_factor_0)*tp.dst_jump_stride_1 + \
                                (loop_on_col * tp.col_ele_per_batch + col_id) * tp.dst_jump_stride_0 + \
                                batch_num * tp.line_per_batch - back_step_up)

        with self.tik_inst.if_scope(tp.case == 3):
            dst_addr.set_as(block_idx * 128 + col_id * 4096)
        return dst_addr

    # --------------------------------------------------------
    #                         |                        |
    #             A           |          A             |  B
    # --------------------------------------------------------
    #                         |                        |
    #             A           |          A             |  B
    # --------------------------------------------------------
    #             C           |          C             |  D
    # --------------------------------------------------------
    # A:   major_col_major_batch
    # B:   tail_col_major_batch
    # C:   major_col_tail_batch
    # D:   tail_col_tail_batch

    # pylint: disable=unused-argument,invalid-name, too-many-arguments
    def _save_ub2gm_major_col_major_batch(self, ub_input, ub_offset, block_idx, round_num, batch_num, loop_on_col):
        tp = self.tiling_param
        with self.tik_inst.for_range(0, tp.col_ele_per_batch) as col_id:
            self.tik_inst.data_move(self.data_out[self._get_dst_addr(block_idx, round_num, batch_num,
                                                                     loop_on_col, col_id, 0)],
                                    ub_input[col_id * tp.line_per_batch],
                                    0, 1, tp.line_block_per_batch, 0, 0)

    def _save_ub2gm_tail_col_major_batch(self, ub_input, ub_offset, block_idx, round_num, batch_num):
        tp = self.tiling_param
        with self.tik_inst.for_range(0, tp.col_ele_tail) as col_id:
            self.tik_inst.data_move(self.data_out[self._get_dst_addr(block_idx, round_num, batch_num,
                                                                     tp.loop_per_line, col_id, 0)],
                                    ub_input[col_id * tp.line_per_batch],
                                    0, 1, tp.line_block_per_batch, 0, 0)

    def _save_ub2gm_major_col_tail_batch(self, ub_input, ub_offset, block_idx, round_num, loop_on_col):
        tp = self.tiling_param
        with self.tik_inst.for_range(0, tp.col_ele_per_batch) as col_id:
            self.tik_inst.data_move(self.data_out[self._get_dst_addr(block_idx, round_num, tp.batch_num_per_round,
                                                                     loop_on_col, col_id, tp.back_step_up)],
                                    ub_input[col_id * tp.line_tail_batch],
                                    0, 1, tp.line_block_tail_batch, 0, 0)

    def _save_ub2gm_tail_col_tail_batch(self, ub_input, ub_offset, block_idx, round_num):
        tp = self.tiling_param
        with self.tik_inst.for_range(0, tp.col_ele_tail) as col_id:
            self.tik_inst.data_move(self.data_out[self._get_dst_addr(block_idx, round_num, tp.batch_num_per_round,
                                                                     tp.loop_per_line, col_id, tp.back_step_up)],
                                    ub_input[col_id * tp.line_tail_batch],
                                    0, 1, tp.line_block_tail_batch, 0, 0)

    def _reorder_last_axis(self, ub_input, ub_offset, col_ele_num, line_per_batch, pad_ele_num):
        # step1. make all elements in the first col
        self.tiling_param.fp16_offset_1.set_as(2048)
        self.tiling_param.fp16_offset_2.set_as(2048+32768)

        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")
        src_ele_num_in_fp16 = ub_offset * ELE_NUM_PER_BLOCK_FP16
        src_list = [ub_input_fp16[src_ele_num_in_fp16 * i] for i in range(ELE_NUM_PER_BLOCK_FP16)]
        dst_list_intermediate = [ub_input_fp16[self.tiling_param.fp16_offset_1 + ELE_NUM_PER_BLOCK_FP16 * i] \
                                 for i in range(ELE_NUM_PER_BLOCK_FP16)]
        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate, src_list, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate,
                                    src_list, ub_offset, ELE_NUM_PER_BLOCK_FP16, 1)
        # step2. move output elements together
        with self.tik_inst.for_range(0, col_ele_num) as i:
            self.tik_inst.data_move(ub_input_fp16[self.tiling_param.fp16_offset_2 + \
                                                  i * line_per_batch * self.fp16_times * ELE_NUM_PER_BLOCK_FP16],
                                    ub_input_fp16[self.tiling_param.fp16_offset_1 + \
                                                  i * self.fp16_times * ELE_NUM_PER_BLOCK_FP16],
                                    0, line_per_batch, self.fp16_times,
                                    (col_ele_num + pad_ele_num) * self.fp16_times - self.fp16_times, 0)
        # step3. make all elements in the first col be in memory of contiguous
        # block numbers after erasing  elements aligned
        src_list_intermediate = [ub_input_fp16[self.tiling_param.fp16_offset_2 + ELE_NUM_PER_BLOCK_FP16 * i] \
                                 for i in range(ELE_NUM_PER_BLOCK_FP16)]
        dst_list_finally = [ub_input_fp16[ub_offset * 16 * i] for i in range(ELE_NUM_PER_BLOCK_FP16)]

        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate,
                                    ub_offset, 1, ELE_NUM_PER_BLOCK_FP16)

    def _move_data_last_axis(self):
        tp = self.tiling_param
        ub_offset = self.tik_inst.Scalar("int32") # unit : block
        ub_offset.set_as(0)
        ub_input = self.tik_inst.Tensor(self.data_in.dtype, (self.ub_size,), tik.scope_ubuf, "ub_input")
        # 32 core
        ELEMENT_PER_CORE = 256
        round_num_per_core = self.tik_inst.Scalar("int32")
        round_num_per_core.set_as(tp.round_num_per_core)

        with self.tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
            with self.tik_inst.if_scope(tp.case == 2):
                with self.tik_inst.if_scope(block_idx > 11):
                    round_num_per_core.set_as(tp.round_num_per_core+1)

            with self.tik_inst.if_scope(block_idx < tp.used_core):
                with self.tik_inst.for_range(0, round_num_per_core) as round_num:
                    with self.tik_inst.for_range(0, tp.loop_per_line) as loop_on_col:
                        self._init_jump_param(block_idx)
                        with self.tik_inst.for_range(0, tp.batch_num_per_round) as batch_num:
                            ub_offset.set_as(0)
                            with self.tik_inst.for_range(0, tp.line_per_batch) as line_number:
                                self.tik_inst.data_move(ub_input[ub_offset * self.element_num_per_block],
                                                        self.data_in[self._get_src_addr(block_idx, loop_on_col,
                                                                                        round_num, ELEMENT_PER_CORE)],
                                                        0,
                                                        1,
                                                        tp.col_block_per_batch,
                                                        0,
                                                        0)
                                self._update_jump_param()
                                ub_offset.set_as(ub_offset + tp.col_block_per_batch)
                            self._reorder_last_axis(ub_input, ub_offset, tp.col_ele_per_batch,
                                                    tp.line_per_batch, tp.pad_ele_num_major)
                            self._save_ub2gm_major_col_major_batch(ub_input, ub_offset, block_idx,
                                                                   round_num, batch_num, loop_on_col)
                        # major_col_tail_batch
                        with self.tik_inst.if_scope(tp.line_tail_batch != 0):
                            ub_offset.set_as(0)
                            self._update_jump_param_batch_tail()
                            with self.tik_inst.for_range(0, tp.line_tail_batch) as line_number:
                                self.tik_inst.data_move(ub_input[ub_offset * self.element_num_per_block],
                                                        self.data_in[self._get_src_addr(block_idx, loop_on_col,
                                                                                        round_num,
                                                                                        ELEMENT_PER_CORE)],
                                                        0,
                                                        1,
                                                        tp.col_block_per_batch,
                                                        0,
                                                        0)
                                self._update_jump_param()
                                ub_offset.set_as(ub_offset + tp.col_block_per_batch)

                            self._reorder_last_axis(ub_input, ub_offset, tp.col_ele_per_batch,
                                                    tp.line_tail_batch, tp.pad_ele_num_major)
                            self._save_ub2gm_major_col_tail_batch(ub_input, ub_offset,
                                                                  block_idx, round_num, loop_on_col)
                    # tail_col
                    with self.tik_inst.if_scope(tp.col_ele_tail != 0):
                        self._init_jump_param(block_idx)
                        # tail_col_major_batch
                        with self.tik_inst.for_range(0, tp.batch_num_per_round) as batch_num:
                            ub_offset.set_as(0)
                            with self.tik_inst.for_range(0, tp.line_per_batch) as line_number:
                                self.tik_inst.data_move(ub_input[ub_offset * self.element_num_per_block],
                                                        self.data_in[self._get_src_addr_tail_col(block_idx,
                                                                                                 ELEMENT_PER_CORE)],
                                                        0,
                                                        1,
                                                        tp.col_block_tail,
                                                        0,
                                                        0)
                                self._update_jump_param()
                                ub_offset.set_as(ub_offset + tp.col_block_tail)
                            self._reorder_last_axis(ub_input, ub_offset, tp.col_ele_tail, tp.line_per_batch,
                                                    tp.pad_ele_num_tail_col)
                            self._save_ub2gm_tail_col_major_batch(ub_input, ub_offset, block_idx, round_num, batch_num)
                        # tail_col_tail_batch
                        with self.tik_inst.if_scope(tp.line_tail_batch != 0):
                            ub_offset.set_as(0)
                            self._update_jump_param_batch_tail()
                            with self.tik_inst.for_range(0, tp.line_tail_batch) as line_number:
                                self.tik_inst.data_move(ub_input[ub_offset * self.element_num_per_block],
                                                        self.data_in[self._get_src_addr_tail_col(block_idx,
                                                                                                 ELEMENT_PER_CORE)],
                                                        0,
                                                        1,
                                                        tp.col_block_tail,
                                                        0,
                                                        0)
                                self._update_jump_param()
                                ub_offset.set_as(ub_offset + tp.col_block_tail)
                            self._reorder_last_axis(ub_input, ub_offset, tp.col_ele_tail, tp.line_tail_batch,
                                                    tp.pad_ele_num_tail_col)
                            self._save_ub2gm_tail_col_tail_batch(ub_input, ub_offset, block_idx, round_num)

    def _reorder_university(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param
        self.tiling_param.fp16_offset_1.set_as(256*256)
        self.tiling_param.fp16_offset_2.set_as(256*256*2)

        # do hwc to chw transfer
        inner_hw_len_1 = 8
        fp16_inner_hwc_len = 8 * 224 * 2
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")
        ub_input_fp32 = ub_input.reinterpret_cast_to("float32")

        # first vnchwconv
        src_addr_list = [ub_input_fp16[fp16_inner_hwc_len * i] for i in range(ELE_NUM_PER_BLOCK_FP16)]
        dst_addr_list = [ub_input_fp16[tp.fp16_offset_1 + ELE_NUM_PER_BLOCK_FP16 * i] for \
                         i in range(ELE_NUM_PER_BLOCK_FP16)]
        repeat_cnt = 224
        src_stride = 0 if repeat_cnt == 1 else 1
        dst_stride = 0 if repeat_cnt == 1 else 16
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # do hwc to chw transfer
        with self.tik_inst.for_range(0, inner_hw_len_1) as i:
            self.tik_inst.data_move(ub_input_fp16[i * 2 * ELE_NUM_PER_BLOCK_FP16],
                                    ub_input_fp16[tp.fp16_offset_1 + i * tp.col_ele_per_batch * 2 * ELE_NUM_PER_BLOCK_FP16],
                                    0, tp.col_ele_per_batch, 2, 0, (inner_hw_len_1 - 1) * 2)

        # second vnchwconv
        src_addr_list = [ub_input_fp16[ELE_NUM_PER_BLOCK_FP16 * i] for i in range(ELE_NUM_PER_BLOCK_FP16)]
        dst_addr_list = [ub_input_fp16[tp.fp16_offset_1 + fp16_inner_hwc_len * i] for \
                         i in range(ELE_NUM_PER_BLOCK_FP16)]
        src_stride = 0 if repeat_cnt == 1 else 16
        dst_stride = 0 if repeat_cnt == 1 else 1
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt, dst_stride, src_stride)

        # move hw in together
        with self.tik_inst.for_range(0, tp.col_ele_per_batch) as axis_c_idx:
            with self.tik_inst.for_range(0, 2) as add_idx:
                self.tik_inst.vadds(64, ub_input_fp32[axis_c_idx * 128 + add_idx * 64],
                                    ub_input_fp32[tp.fp16_offset_1//2 + axis_c_idx * 8 + \
                                                  add_idx * tp.col_ele_per_batch * 8 * 8],
                                    0, 1, 1, tp.col_ele_per_batch, 8, 8)

    def _copy_in_major_col_major_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param
        ub_offset.set_as(0)
        with self.tik_inst.for_range(0, tp.line_per_batch) as line_number:
            self.tik_inst.data_move(ub_input[ub_offset * self.element_num_per_block],
                                    self.data_in[self._get_src_addr(block_idx, 0, 0, 0)],
                                    0,
                                    1,
                                    tp.col_block_per_batch,
                                    0,
                                    0)
            self._update_jump_param()
            ub_offset.set_as(ub_offset + tp.col_block_per_batch)

    def _copy_in_major_col_tail_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param
        ub_offset.set_as(0)

    def _copy_in_tail_col_major_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param
        ub_offset.set_as(0)

    def _copy_in_tail_col_tail_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param
        ub_offset.set_as(0)

    def _copy_out_major_col_major_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param
        with self.tik_inst.for_range(0, tp.col_ele_per_batch) as col_id:
            self.tik_inst.data_move(self.data_out[self._get_dst_addr(block_idx, 0, 0, tp.loop_per_line, col_id, 0)],
                                    ub_input[col_id * tp.line_per_batch],
                                    0, 1, tp.line_block_per_batch, 0, 0)

    def _copy_out_major_col_tail_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param

    def _copy_out_tail_col_major_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param

    def _copy_out_tail_col_tail_batch(self, block_idx, ub_input, ub_offset):
        tp = self.tiling_param

    def _move_data_last_axis_university(self):
        ub_offset = self.tik_inst.Scalar("int32") # unit : block
        ub_offset.set_as(0)
        ub_input = self.tik_inst.Tensor(self.data_in.dtype, (self.ub_size,), tik.scope_ubuf, "ub_input")
        tp = self.tiling_param

        with self.tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
            self._init_jump_param(block_idx)
            with self.tik_inst.if_scope(block_idx < tp.used_core):
                with self.tik_inst.for_range(0, tp.loop_per_line) as loop_on_col:
                    self._copy_in_major_col_major_batch(block_idx, ub_input, ub_offset)
                    self._reorder_university(block_idx, ub_input, ub_offset)
                    self._copy_out_major_col_major_batch(block_idx, ub_input, ub_offset)

                with self.tik_inst.if_scope(tp.major_col_tail_batch):
                    self._copy_in_major_col_tail_batch(block_idx, ub_input, ub_offset)
                    self._reorder_university(block_idx, ub_input, ub_offset)
                    self._copy_out_major_col_tail_batch(block_idx, ub_input, ub_offset)

                with self.tik_inst.if_scope(tp.tail_col_major_batch):
                    self._copy_in_tail_col_major_batch(block_idx, ub_input, ub_offset)
                    self._reorder_university(block_idx, ub_input, ub_offset)
                    self._copy_out_tail_col_major_batch(block_idx, ub_input, ub_offset)

                with self.tik_inst.if_scope(tp.tail_col_tail_batch):
                    self._copy_in_tail_col_tail_batch(block_idx, ub_input, ub_offset)
                    self._reorder_university(block_idx, ub_input, ub_offset)
                    self._copy_out_tail_col_tail_batch(block_idx, ub_input, ub_offset)

    def _compute(self):
        self._move_data_last_axis_university()
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.data_in, self.data_perm],
                               outputs=[self.data_out],
                               flowtable=[self.data_tiling])
        te.op.add_compile_info("vars", {"ub_size": UB_SIZE//BLOCK_SIZE, "core_num": CORE_NUM, "dtype": self.x_dtype})
        return {"compile_info": te.op.get_compile_info()}


@te.op.register_operator("Transpose")
def transpose(x, perm, y, kernel_name="transpose"):
    """
    do transpose by perm attribute

    Parameters
    ----------
    x : dict
        shape and dtype of input
    perm : list or tuple
        permutation of the dimension of tensor
    y : dict
        shape and dtype of output, the dtype should be same as input
    kernel_name : str
        kernel name, default value is "transpose"

    Returns
    -------
    compile info
    """

    x_dtype = x.get("dtype").lower()
    p_dtype = perm.get("dtype").lower()
    y_dtype = y.get("dtype").lower()

    tik_inst = tik.Tik()

    data_in = tik_inst.Tensor(x_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_in")
    data_perm = tik_inst.Tensor(p_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_perm")
    data_out = tik_inst.Tensor(y_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_out")
    data_workspace = tik_inst.Tensor(y_dtype, (2*1024*1024*1024,), tik.scope_gm, "data_workspace", is_workspace=True)
    data_tiling = tik_inst.Tensor("int64", (TILING_PARAM_NUM,), tik.scope_gm, "data_tiling")
    tensor_list = [data_in, data_perm, data_out, data_workspace, data_tiling]

    transpose_instance = Transpose(tik_inst, x_dtype, tensor_list, kernel_name)

    return transpose_instance._compute()
