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
Define the main function of generate schedule by cheque
"""
from ast import literal_eval as make_tuple
import re
from te import tvm
from te.platform import cce_emitinsn_params
from te.lang.cce.rl_bank.bank_cfg import INTRIN_MAP
from te.lang.cce.rl_bank.bank_cfg import SCOPE_DICT
from te.lang.cce.rl_bank.bank_cfg import MODE_RUNTIME
from te.lang.cce.rl_bank.bank_cfg import ScheduleTarget
from te.lang.cce.rl_bank.bank_cfg import Axis
from te.lang.cce.rl_bank.bank_cfg import PRIMITIVE_DICT


def proc_cache_read(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments,
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 0:
        sch_target = sch_targets[stage_index]
        # cache Read args  is Scope and Consumers
        scope_id = args[0]
        scope = SCOPE_DICT[scope_id]
        consumers_indicies = args[1]
        consumers = [sch_targets[i].obj for i in consumers_indicies]
        consumer_names = ', '.join([sch_targets[i].name for i in consumers_indicies])

        readed_tensor = None
        if mode == MODE_RUNTIME:
            readed_tensor = sch.cache_read(sch_target.obj, scope, consumers)
        # orignal Tensor name x，cacheRead Tensor name：x_l_n
        read_index = 0
        readed_pattern = r'^%s_l_\d+$' % (sch_target.name)
        if stage_index + 1 < len(sch_targets) and re.match(readed_pattern,
                                                           sch_targets[stage_index + 1].name):
            read_index = int(sch_targets[stage_index + 1].name.split('_')[-1]) + 1
        readed_name = '%s_l_%03d' % (sch_target.name, read_index)
        # inset after orignal tensor
        sch_targets.insert(stage_index + 1, ScheduleTarget(readed_name, readed_tensor, []))

        code_line = "%s = sch.cache_read(%s, '%s', [%s])" % (readed_name, sch_target.name, scope,
                                                             consumer_names)
        code_lines.append(code_line)


def proc_cache_write(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 1:
        if isinstance(stage_index, list):
            # cheque form is [[6, 2], 1, 1] when more than one tensors do cache_write
            write_tensor_nums = stage_index[1]
            stage_index = stage_index[0]
            sch_target = sch_targets[stage_index]
            stage_name = sch_target.name
            # cache write args is Scope
            scope_id = args[0]
            scope = SCOPE_DICT[scope_id]
            written_tensors = [None]
            write_tensor_objs = []
            write_tensor_names = []
            written_tensor_names = []
            for idx in range(write_tensor_nums):
                write_tensor_objs.append(sch_target.obj.op.output(idx))
                write_tensor_names.append(stage_name + "_v%s" % idx)
                written_tensor_names.append(stage_name + "_v%s_l" % idx)
            if mode == MODE_RUNTIME:
                written_tensors = sch.cache_write(write_tensor_objs, scope)

            written_name = '%s_l' % stage_name
            # insert before orignal  tensor
            sch_targets.insert(stage_index, ScheduleTarget(written_name, written_tensors[0], []))
            code_lines.append(
                "%s = sch.cache_write([%s], '%s')" %
                (', '.join(written_tensor_names), ', '.join(write_tensor_names), scope))
            code_lines.append('%s = %s' % (written_name, written_tensor_names[0]))

        else:
            sch_target = sch_targets[stage_index]
            # cache write args is Scope
            scope_id = args[0]
            scope = SCOPE_DICT[scope_id]
            written_tensor = None
            if mode == MODE_RUNTIME:
                written_tensor = sch.cache_write(sch_target.obj, scope)
            # Tensor name x，after x_l_n
            written_name = '%s_l' % sch_target.name
            # insert before orignal  tensor
            sch_targets.insert(stage_index, ScheduleTarget(written_name, written_tensor, []))
            code_lines.append("%s = sch.cache_write(%s, '%s')" %
                              (written_name, sch_target.name, scope))


def proc_preload(stage_index, primitive, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 20:
        sch_target = sch_targets[stage_index]
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].preload()
        code_lines.append("sch[%s].preload()" % sch_target.name)


def proc_double_buffer(stage_index, primitive, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 2:
        sch_target = sch_targets[stage_index]
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].double_buffer()
        code_lines.append("sch[%s].double_buffer()" % sch_target.name)


def proc_compute_inline(stage_index, primitive, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 3:
        sch_target = sch_targets[stage_index]
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].compute_inline()
        code_lines.append("sch[%s].compute_inline()" % sch_target.name)


def proc_get_axis(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 4:
        # get axis
        sch_target = sch_targets[stage_index]
        # axis_num
        axis_num = args[0]
        for i in range(axis_num):
            axis_obj = None
            if mode == MODE_RUNTIME:
                axis_obj = sch[sch_target.obj].op.axis[i]
            axis_name = '%s_axis_%d' % (sch_target.name, i)
            sch_target.axes.append(Axis(axis_name, axis_obj))
            code_lines.append("%s = sch[%s].op.axis[%d]" % (axis_name, sch_target.name, i))


def proc_get_reduce_axis(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 5:
        # get reduce axis
        sch_target = sch_targets[stage_index]
        axis_num = args[0]
        for i in range(axis_num):
            axis_obj = None
            if mode == MODE_RUNTIME:
                axis_obj = sch[sch_target.obj].op.reduce_axis[i]
            axis_name = '%s_reduce_axis_%d' % (sch_target.name, i)
            sch_target.axes.append(Axis(axis_name, axis_obj))
            code_lines.append("%s = sch[%s].op.reduce_axis[%d]" % (axis_name, sch_target.name, i))


def proc_split(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 6:
        # Split by Factor
        sch_target = sch_targets[stage_index]
        # SplitByFactor args is axis_index and Factor
        axis_index = args[0]
        factor = args[1]
        # delete split axis
        proc_axis = sch_target.axes.pop(axis_index)
        axis_name = proc_axis.name
        axis_obj = proc_axis.obj
        outer, inner = None, None
        if mode == MODE_RUNTIME:
            outer, inner = sch[sch_target.obj].split(axis_obj, factor=factor)
        # insert inner then outer
        sch_target.axes.insert(axis_index, Axis("%s_i" % axis_name, inner))
        sch_target.axes.insert(axis_index, Axis("%s_o" % axis_name, outer))
        code_lines.append("%s_o, %s_i = sch[%s].split(%s, factor=%d)" %
                          (axis_name, axis_name, sch_target.name, axis_name, factor))


def proc_nparts(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 7:
        # Split by Nparts
        sch_target = sch_targets[stage_index]
        # SplitByFactor args is axis index and nparts
        axis_index = args[0]
        nparts = args[1]
        # delete split axis
        proc_axis = sch_target.axes.pop(axis_index)
        axis_name = proc_axis.name
        axis_obj = proc_axis.obj
        outer, inner = None, None
        if mode == MODE_RUNTIME:
            outer, inner = sch[sch_target.obj].split(axis_obj, nparts=nparts)
        # insert inner,then outer
        sch_target.axes.insert(axis_index, Axis("%s_i" % axis_name, inner))
        sch_target.axes.insert(axis_index, Axis("%s_o" % axis_name, outer))
        code_lines.append("%s_o, %s_i = sch[%s].split(%s, nparts=%d)" %
                          (axis_name, axis_name, sch_target.name, axis_name, nparts))


def proc_reorder(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 8:
        # Reorder
        sch_target = sch_targets[stage_index]
        left_axis_idx_list = list(range(len(sch_targets[stage_index].axes)))
        order = args[0]
        for axis_idx in order:
            left_axis_idx_list.remove(axis_idx)
        axis_reorder_list = [sch_target.axes[i] for i in order]
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].reorder(*([axis.obj for axis in axis_reorder_list]))
        sch_target.axes = [sch_target.axes[i] for i in order + left_axis_idx_list]
        new_order_str = ', '.join([axis.name for axis in axis_reorder_list])
        code_lines.append("sch[%s].reorder(%s,)" % (sch_target.name, new_order_str))


def proc_allocate_at(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    """
    proc_allocate_at
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    """
    if primitive == 21:
        sch_target = sch_targets[stage_index]
        at_stage_index = args[0]
        at_axis_index = args[1]
        run_once_axes_index_list = args[2]
        at_sch_target = sch_targets[at_stage_index]
        at_axis = at_sch_target.axes[at_axis_index]
        run_once_axes_list = [at_sch_target.axes[i] for i in run_once_axes_index_list]
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].allocate_at(sch[at_sch_target.obj], at_axis.obj,
                                            [axis.obj for axis in run_once_axes_list])

        run_once_axes_str = ""
        if run_once_axes_list:
            run_once_axes_str = ", [%s]" % ", ".join([axis.name for axis in run_once_axes_list])

        code_lines.append("sch[%s].allocate_at(sch[%s], %s%s)" %
                          (sch_target.name, at_sch_target.name, at_axis.name, run_once_axes_str))


def proc_mem_unique(stage_index, primitive, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 22:
        sch_target = sch_targets[stage_index]
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].mem_unique()
        code_lines.append("sch[%s].mem_unique()" % sch_target.name)


def proc_compute_at(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 9:
        # compute at
        sch_target = sch_targets[stage_index]
        at_stage_index = args[0]
        at_axis_index = args[1]
        at_sch_target = sch_targets[at_stage_index]
        at_axis = at_sch_target.axes[at_axis_index]
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].compute_at(sch[at_sch_target.obj], at_axis.obj)
        code_lines.append("sch[%s].compute_at(sch[%s], %s)" %
                          (sch_target.name, at_sch_target.name, at_axis.name))


def proc_fuse(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 15:
        sch_target = sch_targets[stage_index]
        fuse_axis_idx_list = args[0]
        fuse_axis_obj_list = [sch_target.axes[i].obj for i in fuse_axis_idx_list]

        fuse_axis_name_list = [sch_target.axes[i].name for i in fuse_axis_idx_list]
        axis_type = "axis"
        if "reduce_axis" in fuse_axis_name_list[0]:
            axis_type = "reduce_axis"
        code_lines.append(
            "%s_%s_fused_0 = sch[%s].fuse(%s)" %
            (sch_target.name, axis_type, sch_target.name, ", ".join(fuse_axis_name_list)))

        if mode == MODE_RUNTIME:
            fused_axis_obj = sch[sch_target.obj].fuse(*(fuse_axis_obj_list))
            fuse_start_idx = min(fuse_axis_idx_list)
            for _ in fuse_axis_idx_list:
                sch_target.axes.pop(fuse_start_idx)
            sch_target.axes.insert(
                fuse_start_idx, Axis("%s_%s_fused_0" % (sch_target.name, axis_type),
                                     fused_axis_obj))


def proc_rfactor(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 17:
        sch_target = sch_targets[stage_index]
        rfactor_axis = sch_target.axes[args[0]]
        factor_axis = args[1]

        rfactor_name = sch_target.name + "_rfactor"
        code_lines.append("%s = sch.rfactor(%s, %s, factor_axis=%s)" %
                          (rfactor_name, sch_target.name, rfactor_axis.name, factor_axis))

        if mode == MODE_RUNTIME:
            tensor_rfactor = sch.rfactor(sch_target.obj, rfactor_axis.obj, factor_axis)
            if not isinstance(tensor_rfactor, tvm.tensor.Tensor):
                tensor_rfactor = tensor_rfactor[0]
            sch_targets.insert(stage_index, ScheduleTarget(rfactor_name, tensor_rfactor, []))


def proc_set_scope(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 18:
        sch_target = sch_targets[stage_index]
        scope_id = int(args[0])

        if mode == MODE_RUNTIME:
            sch[sch_target.obj].set_scope(SCOPE_DICT[scope_id])
        code_lines.append("sch[%s].set_scope('%s')" % (sch_target.name, SCOPE_DICT[scope_id]))


def proc_bind(stage_index, primitive, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 10:
        sch_target = sch_targets[stage_index]
        bind_axis = sch_target.axes[0]
        if mode == MODE_RUNTIME:
            block = tvm.thread_axis('blockIdx.x')
            sch[sch_target.obj].bind(bind_axis.obj, block)
        code_lines.append("block = tvm.thread_axis('blockIdx.x')")

        code_lines.append("sch[%s].bind(%s, block)" % (sch_target.name, bind_axis.name))


def proc_pragma(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 16:
        # Pragma
        sch_target = sch_targets[stage_index]
        axis_index = args[0]
        pragma_insn_name = INTRIN_MAP[args[1]]
        pragma_insn_offset = args[2]
        if axis_index[0] == -1:
            axis = sch_target.axes[args[0][1]]
        else:
            axis_index = axis_index[0]
            if mode == MODE_RUNTIME:
                axis = Axis('sch[%s].op.axis[%d]' % (sch_target.name, axis_index),
                            sch[sch_target.obj].op.axis[axis_index])
            else:
                axis = Axis('sch[%s].op.axis[%d]' % (sch_target.name, axis_index), None)

        if mode == MODE_RUNTIME:
            sch[sch_target.obj].pragma(axis.obj, pragma_insn_name, pragma_insn_offset)
        code_lines.append("sch[%s].pragma(%s, '%s', %s)" %
                          (sch_target.name, axis.name, pragma_insn_name, pragma_insn_offset))


def proc_emit_insn(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 11:
        # EmitInsn
        sch_target = sch_targets[stage_index]
        axis_index = args[0]
        intrinsic = INTRIN_MAP[args[1]]
        if axis_index[0] == -1:
            axis = sch_target.axes[args[0][1]]
        else:
            axis_index = axis_index[0]
            if mode == MODE_RUNTIME:
                axis = Axis('sch[%s].op.axis[%d]' % (sch_target.name, axis_index),
                            sch[sch_target.obj].op.axis[axis_index])
            else:
                axis = Axis('sch[%s].op.axis[%d]' % (sch_target.name, axis_index), None)

        if mode == MODE_RUNTIME:
            if intrinsic == "mad":
                mad_pattern_value = int(args[2][0])
                init_bias_value = int(args[2][1])
                k_outer_axis_obj_list = [sch_target.axes[axis_idx].obj for axis_idx in args[2][2:]]
                k_outer_axis_name_list = [
                    sch_target.axes[axis_idx].name for axis_idx in args[2][2:]
                ]
                mad_dict = {"mad_pattern": mad_pattern_value, "k_outer": k_outer_axis_obj_list}
                if init_bias_value:
                    mad_dict["init_bias"] = init_bias_value
                code_lines.append(
                    'mad_dict = {"mad_pattern": %s, "k_outer": [%s]%s}' %
                    (mad_pattern_value, ", ".join(k_outer_axis_name_list),
                     ', "init_bias": %s' % init_bias_value if init_bias_value else ""))
                sch[sch_target.obj].emit_insn(axis.obj, intrinsic, mad_dict)
            else:
                sch[sch_target.obj].emit_insn(axis.obj, intrinsic)

        # gen code
        code_lines.append(
            "sch[%s].emit_insn(%s, '%s'%s)" %
            (sch_target.name, axis.name, intrinsic, ", mad_dict" if intrinsic == "mad" else ""))


def proc_reused_by(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    """
    proc_reused_by
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    """
    if primitive == 19:
        sch_target = sch_targets[stage_index]
        reused_sch_target = sch_targets[args[0]]
        reuse_data = False
        if len(args) > 1:
            reuse_data = bool(args[1])
        if reused_sch_target == -1:
            # gen code
            code_lines.append("sch[%s].reused_by(reuse_data=%s)" % (sch_target.name, reuse_data))
        else:
            # gen code
            code_lines.append("sch[%s].reused_by(%s)" % (sch_target.name, reused_sch_target.name))

        if mode == MODE_RUNTIME:
            sch[sch_target.obj].reused_by(reused_sch_target.obj, reuse_data=reuse_data)


def proc_insert_param(primitive, args, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 12:
        # broadcast_axis_offset
        offset = args[0]
        if mode == MODE_RUNTIME:
            cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')
            cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', offset)
        code_lines.append(
            "cce_emitinsn_params.cceEmitParamsIns.del_param('broadcast_axis_offset')")

        code_lines.append(
            "cce_emitinsn_params.cceEmitParamsIns.insert_param('broadcast_axis_offset', %d)" %
            offset)


def proc_storage_align(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 13:
        # storage_align args : axis_index and block_num
        sch_target = sch_targets[stage_index]
        axis_index = args[0]
        block_num = args[1]
        if mode == MODE_RUNTIME:
            axis = Axis('sch[%s].op.axis[%d]' % (sch_target.name, axis_index),
                        sch[sch_target.obj].op.axis[axis_index])
        else:
            axis = Axis('sch[%s].op.axis[%d]' % (sch_target.name, axis_index), None)

        if mode == MODE_RUNTIME:
            sch[sch_target.obj].storage_align(axis.obj, block_num, 0)
        code_lines.append("sch[%s].storage_align(%s, %s, 0)" %
                          (sch_target.name, axis.name, block_num))


def proc_buffer_align(stage_index, primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_buffer_align
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 25:
        # storage_align args : axis_index and block_num
        sch_target = sch_targets[stage_index]
        align_args = args[0]
        align_args_obj = make_tuple(align_args)
        if mode == MODE_RUNTIME:
            sch[sch_target.obj].buffer_align(*align_args_obj)
        code_lines.append("sch[%s].buffer_align%s" % (sch_target.name, align_args))


def proc_cce_special(primitive, args, sch_targets, sch, mode, code_lines):  # pylint: disable=too-many-locals, too-many-arguments
    '''
    proc_cache_read
    :param stage_index:
    :param primitive:
    :param args:
    :param sch_targets:
    :param sch:
    :param mode:
    :param code_lines:
    :return:
    '''
    if primitive == 14:
        # cce_special
        tensor_list_objs = []
        tensor_list_names = []
        orign_out_tensor_list_objs = []
        orign_out_tensor_list_names = []
        real_out_tensor_list_objs = []
        real_out_tensor_list_names = []
        # general cce_special cheque form is [-1, 14, [], [8], [7]]
        # tuple_reduce cce_special cheque form is [-1, 14, [], [[8, 2]], [[7, 2]]]
        for arg_index, tmp_tensor_list_index in enumerate(args):
            tmp_tensor_list_objs = []
            tmp_tensor_list_names = []
            for stage_index in tmp_tensor_list_index:
                if isinstance(stage_index, list):
                    tensor_nums = stage_index[1]
                    stage_index = stage_index[0]
                    sch_target = sch_targets[stage_index]
                    stage_name = sch_target.name
                    for idx in range(tensor_nums):
                        tmp_tensor_list_objs.append(sch_target.obj.op.output(idx))
                        if stage_name.endswith('_l'):
                            tensor_name = "%s_v%s_l" % (stage_name.split('_l')[0], idx)
                        else:
                            tensor_name = "%s_v%s" % (stage_name.split('_l')[0], idx)
                        tmp_tensor_list_names.append(tensor_name)
                else:
                    tmp_tensor_list_objs.append(sch_targets[stage_index].obj.op.output(0))
                    tmp_tensor_list_names.append(sch_targets[stage_index].name)
            if arg_index == 0:
                tensor_list_objs = tmp_tensor_list_objs
                tensor_list_names = tmp_tensor_list_names
            elif arg_index == 1:
                orign_out_tensor_list_objs = tmp_tensor_list_objs
                orign_out_tensor_list_names = tmp_tensor_list_names
            else:
                real_out_tensor_list_objs = tmp_tensor_list_objs
                real_out_tensor_list_names = tmp_tensor_list_names

        if mode == MODE_RUNTIME:
            sch.cce_special = dict()
            sch.cce_special["tensor_list"] = tensor_list_objs
            sch.cce_special["orign_out_tensor"] = orign_out_tensor_list_objs
            sch.cce_special["real_out_tensor"] = real_out_tensor_list_objs

        code_lines.append("sch.cce_special = dict()")

        code_lines.append('sch.cce_special["tensor_list"] = [%s]' % "".join(tensor_list_names))

        code_lines.append('sch.cce_special["orign_out_tensor"] = [%s]' %
                          "".join(orign_out_tensor_list_names))

        code_lines.append('sch.cce_special["real_out_tensor"] = [%s]' %
                          "".join(real_out_tensor_list_names))


def get_leaves(res_list):
    '''

    :param res_list:
    :return:
    '''
    tensors = [tensor for tensor in res_list]
    visited_tensor = set()
    non_leaf_nodes = set()
    while tensors:
        current_tensor = tensors.pop(0)
        current_op = current_tensor.op
        current_name = current_op.name
        if current_name in visited_tensor \
                or isinstance(current_op, tvm.tensor.PlaceholderOp):
            continue
        visited_tensor.add(current_name)
        for input_tensor in current_op.input_tensors:
            input_name = input_tensor.op.name
            if input_name not in non_leaf_nodes:
                non_leaf_nodes.add(input_name)
            tensors.append(input_tensor)
    leaf_outs = []
    for tensor in res_list:
        if tensor.name not in non_leaf_nodes:
            leaf_outs.append(tensor)
    return leaf_outs


def withdraw(res_list, cheque, mode="runtime"):
    '''
    withdraw
    :param res_list:
    :param cheque:
    :param mode:
    :return:
    '''
    if not isinstance(res_list, list):
        res_list = [res_list]
    # firstly create_schedule
    code_lines = []
    # create schedule only need leaf-node. Multi leaf op should add proc
    leaves = get_leaves(res_list)
    sch = tvm.create_schedule([res.op for res in leaves])
    # element in sch_targets List is [Tensor name， Tensor obj， comm axis list， reduce axis list]
    sch_targets = []
    for stage in sch.stages:
        sch_targets.append(ScheduleTarget(stage.op.name, stage.op.output(0), []))

    for action in cheque:
        stage_index, primitive, *args = action
        if primitive not in PRIMITIVE_DICT:
            RuntimeError('Invalid primitive: [%s]' % primitive)

        proc_cache_read(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_cache_write(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_preload(stage_index, primitive, sch_targets, sch, mode, code_lines)
        proc_double_buffer(stage_index, primitive, sch_targets, sch, mode, code_lines)
        proc_compute_inline(stage_index, primitive, sch_targets, sch, mode, code_lines)
        proc_get_axis(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_get_reduce_axis(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_split(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_nparts(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_reorder(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_allocate_at(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_mem_unique(stage_index, primitive, sch_targets, sch, mode, code_lines)
        proc_compute_at(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_fuse(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_rfactor(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_set_scope(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_bind(stage_index, primitive, sch_targets, sch, mode, code_lines)
        proc_pragma(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_emit_insn(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_reused_by(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_insert_param(primitive, args, mode, code_lines)
        proc_storage_align(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_buffer_align(stage_index, primitive, args, sch_targets, sch, mode, code_lines)
        proc_cce_special(primitive, args, sch_targets, sch, mode, code_lines)

    return sch, code_lines


def gen_sch_by_cheque(out_tensors, action_list):
    '''
    gen_sch_by_cheque
    :param out_tensors:
    :param action_list:
    :return:
    '''
    try:
        sch, _ = withdraw(out_tensors, action_list, MODE_RUNTIME)
        return True, sch
    except RuntimeError:
        return False, None
