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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""
import collections  # 'pylint: disable=E0401, W0611, C0412, C0302
import copy
import json
import os
import re
import itertools
import warnings
from functools import reduce as functools_reduce

from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.common.platform import scope_cbuf_fusion
from tbe.common.buildcfg import get_L1_info
from tbe.common.utils import log
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.platform.platform_info import intrinsic_check_support
from tbe.common.buildcfg import get_current_build_config
from te.platform.fusion_manager import fusion_manager
from tbe.common.context import get_context
from tbe.common.rl_bank import bank_manager
from tbe.common import buildcfg
from tbe.dsl.compute.conv_compute import ConvParam  # 'pylint: disable=C0412
from tbe.common.rl_bank import rl_bank
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.instrinsic import cce_emitinsn_params
from tbe.common.utils import shape_to_list
from tbe.common.utils import check_kernel_name

from .util import gen_dfs_tensor_map
from .util import generic_dispatch
from .cce_schedule_declarations import OpPatterns
from .cce_schedule_declarations import OpSubPatterns
from .cce_schedule_declarations import OpSpecTypes
from .cce_schedule_mappings import OpPatternRecognizer

from .concat_schedule import CceConcatOp
from .conv_schedule import CceConvOp
from .conv_schedule import reget_tensor_list
from .conv2d_backprop_input_schedule import CceConv2dBackpropInputOp
from .conv2d_backprop_filter_schedule import CceConv2dBackpropFilterOp
from .conv3d_backprop_input_schedule import CceConv3dBackpropInputOp
from .conv3d_backprop_filter_schedule import CceConv3dBackpropFilterOp
from .layer_norm_cube_schedule import CceLayerNormCubeOp
from .layer_norm_cube_schedule import reget_layernorm_multioutput
from .depthwise_conv2d_schedule import depthwise_conv2d_schedule
from .elewise_schedule import CceOp
from .elewise_speel_schedule import CceSpeelOp
from .mmad_schedule import mmad_schedule
from .gemm_schedule import gemm_schedule
from .pooling2d_schedule import pooling2d_schedule
from .pooling3d_schedule import pooling3d_schedule
from .segment_schedule import CceSegmentOp
from .segment_speel_schedule import CceSegmentSpeelOp
from .inplace_schedule import CceInplaceOp
from ..compute import cast
from .pure_broadcast_schedule import PureBroadcastSchedule
from .elewise_schedule_new import ElewiseSchedule
from .bn_update_schedule import bn_update_schedule
from .bn_reduce_schedule import bn_reduce_schedule
from .bn_grad_reduce_schedule import BnGradReduceSchedule
from .elewise_multi_schedule import ElewiseMultiSchedule
from .reduce_multi_schedule import ReduceMultiSchedule
from .reduce_mean_mid_reduce_high_performance_schedule import \
    reduce_mean_mid_reduce_high_performance_schedule
from .workspace_multi_schedule import WorkspaceMultiSchedule
from .softmax_cross_entropy_with_logits_schedule import logits_schedule
from .l2_normalize_schedule import L2NormalizeSchedule
from .softmax_schedule import SoftmaxSchedule
from .l2_loss_schedule import l2_loss_schedule
from .reduce_atomic_schedule import ReduceAtomicSchedule
from .reduce_5hdc_schedule import Reduce5HDCSchedule
from .conv_schedule import check_quantfuse_doubleout
from .l2loss_mul_addn_schedule import l2loss_mul_addn_schedule
from .conv3d_schedule import CceConv3dOp

from .bn_update_grad_shedule import bn_update_grad_schedule
from .bn_update_grad_schedule_nd import bn_update_grad_schedule_nd
from .layer_norm_grad_schedule import layer_norm_grad_schedule

from .ascend_anti_quant_schedule import ascend_anti_quant_schedule
from .ascend_quant_schedule import ascend_quant_schedule
from .ascend_dequant_schedule import ascend_dequant_schedule

from .read_select_schedule import read_select_schedule
from .write_select_schedule import write_select_schedule
from .strided_read_schedule import strided_read_schedule
from .strided_write_schedule import strided_write_schedule

from .ascend_dequant_s16_schedule import ascend_dequant_s16_schedule
from .ascend_requant_schedule import ascend_requant_schedule
from .ascend_requant_s16_schedule import ascend_requant_s16_schedule
from . import util as te_util  # 'pylint: disable=E0401
from .fixpipe_schedule import fixpipe_schedule
from .cosine_embedding_loss_schedule import \
    cosine_embedding_loss_schedule
from .dilation_schedule import dilation_schedule


def set_op_pattern(all_tags, op_info):
    """
    set op pattern to gemm when batchmatmul fused with fusedmuladd or reduce_sum
    """
    op_pattern = op_info["pattern"]
    if ("reduce_sum" in all_tags or "dropout_broadcast" in all_tags or
        "broadcast_reshape" in all_tags or "broadcast_mad" in all_tags):
        op_pattern = OpPatterns.MATMUL_PATTERN
    return op_pattern


def get_op_info(outs):  # 'pylint: disable=R0912, R0914, R0915
    """
    dfs the compute garph to get the op info, the fomrt as follows:
        op_info
        {
        pattern: "xxx"
        input_tensors : []
        mid_tensors : []
        output_tensors : []
        tensor_map: {input : [outputs]}
        }
    Parameters
    ----------
    outs : the outputs of op

    Returns
    -------
    op_info
    """

    # Initialize op info
    op_info = collections.OrderedDict()

    # Initialize the following dictionaries
    # input_tensors: All input tensors (Usually placeholders)
    # mid_tensors  : All tensors except input and output (Usually ComputeOps)
    # out_tensors  : All output tensors
    # tensor_map   : All input -> outputs mapping relations
    output_tensors = outs

    visited, input_tensors, mid_tensors, tensor_map = gen_dfs_tensor_map(outs)
    op_info['dfs_tensor_list'] = visited

    # Get op statistics
    op_statistics = OpPatternRecognizer.get_compute_statistics(visited)
    # Get op patterns
    # Add your new flag or pattern in cce_schedule_declarations.py
    # Add your new matching rule in cce_schedule_mappings.py
    # DO NOT MODIFY OpPatternRecognizer if not necessary!!!
    # Read comments in cce_schedule_mappings.py for more information
    patterns = OpPatternRecognizer.get_pattern(op_statistics,
                                               input_tensors,
                                               output_tensors,
                                               visited,
                                               tensor_map)
    op_pattern, op_subpattern, op_spectype = patterns
    op_info['pattern'] = op_pattern
    if op_subpattern:
        op_info['sub_pattern'] = op_subpattern
        op_subpattern = op_subpattern.value
    if op_spectype:
        op_info['type'] = op_spectype
        if op_spectype == OpSpecTypes.MVN and "op_name" in input_tensors[0].op.attrs and \
                input_tensors[0].op.attrs["op_name"] == "mvn_v2":
            op_info['type'] = OpSpecTypes.MVNV2
        op_spectype = op_spectype.value
    # write the tensors
    op_info['input_tensors'] = input_tensors
    op_info['mid_tensors'] = mid_tensors
    op_info['output_tensors'] = output_tensors

    # write the tensor map
    op_info['tensor_map'] = tensor_map

    return op_info


def remove_redundant_cast_op(res, outs):
    """
    ->....->res_fp32_1-->res_fp16-->res_fp32_2
    :return: res_fp32_1
    """
    res_dtype = res.dtype
    res_pre = res
    tag = res.op.tag
    if tag is not None:
        tag = tag.split("|")[0]
    while tag == 'elewise_single_cast':
        res_pre = res_pre.op.input_tensors[0]
        if isinstance(res_pre.op, tvm.tensor.PlaceholderOp) or res_pre in outs:
            break

        if res_dtype == res_pre.dtype:
            res = res_pre

        tag = res_pre.op.tag
        if tag is not None:
            tag = tag.split("|")[0]

    return res


def verify_compute_tensor(tensors):
    """
    verify compute tensor by rule:
    rule 1: only one tensor in compute, return False
    rule 2: any compute tensor shall be taggeg with 'elewise_single_cast', if correct return False
    otherwise return True
    tensors: target tensor which needs to verify
    """
    # rule 1
    if len(tensors) == 1:
        return False

    for tensor in tensors:
        if 'not_auto_cast' in tensor.op.tag or \
                'convolution' in tensor.op.tag or \
                'conv2d' in tensor.op.tag or \
                'conv3d' in tensor.op.tag or \
                'matmul' in tensor.op.tag:
            return False

    # rule 2
    for tensor in tensors:
        if tensor.op.tag != 'elewise_single_cast':
            return True
    return False


def rl_search_proc(outs, option):
    """
    rl_search schedule
    """
    # no special tensor list
    tensor_list = []
    # only support single output
    real_outs = outs
    if "rl_schedule_dict" in option:
        log.info("start to call offline rl_search.")
        from schedule_search.offline_schedule import offline_schedule  # 'pylint: disable=E0401
        schedule = offline_schedule(outs, option["rl_schedule_dict"])
    else:
        log.info("start to call online rl_search.")
        from schedule_search.online_infer import online_infer  # 'pylint: disable=E0401
        schedule = online_infer(outs, option)
    return schedule, tensor_list, real_outs


def get_all_tags(res):
    """
    get all tags
    :param res: tensor
    :return: list
    """
    tensor_tags = set()

    def get_tag(tenosr):
        """
        find all tag
        :param tensor: tensor
        :return: all tags
        """
        tensor_list = tenosr.op.input_tensors
        tensor_tags.add(tenosr.op.tag)
        for one_tensor in tensor_list:
            tensor_tags.add(one_tensor.op.tag)
            get_tag(one_tensor)

    get_tag(res)
    return tensor_tags


def is_cube_pattern(op_pattern):
    """
    check if is cube pattern
    op_pattern: input pattern
    return: bool
    """
    cube_patterns = [
        OpPatterns.MATMUL_PATTERN,
        OpPatterns.MATMUL_V2_PATTERN,
        OpPatterns.GEMM_PATTERN,
        OpPatterns.CONV2D_BACKPROP_INPUT_PATTERN,
        OpPatterns.CONV2D_BACKPROP_FILTER_PATTERN
    ]
    return op_pattern in cube_patterns


def _skip_cast_op(in_dtype, out, op_pattern):
    """
    check if need to add cast
    in_dtype: input dtype
    out: output tensor
    op_pattern: input pattern
    return bool
    """
    if in_dtype in ("float32", "float16") and out.dtype == "int32":
        return True
    # c = vesl(condition,a,b), c is the same as a and b,
    # condition_map[update_tensor]n is bool
    if in_dtype in ("bool", "uint64"):
        return True
    if not check_is_need_cast(out):
        return True
    support_l0c2out = intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if support_l0c2out and is_cube_pattern(op_pattern):
        return True
    return False


def schedule_cce(outs, option=None):  # 'pylint: disable=R0912, R0914, R0915
    """
    schedule cce
    """
    # for RL tune getting res
    ConvParam.conv_deq_req_double_out = False
    ConvParam.conv_reluv2_flag = False
    bank_manager.set_op_res(outs)

    outs = reget_tensor_list(outs)
    outs = reget_layernorm_multioutput(outs)

    cce_emitinsn_params.cceEmitParamsIns.clear_param()
    if isinstance(outs, (tuple, list)):
        if len(outs) > 1:
            if not check_support_muti_output(outs):
                dict_args = {"errCode": "E90003", "detailed_cause": "Only vector op support multi output now"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            out_tmp = outs
        else:
            out_tmp = [outs[0]]  # suppose input and output are the same
    else:
        out_tmp = [outs]
    outs = out_tmp
    origin_outs = copy.copy(outs)

    # if has redundant_cast_op, remove redundant
    for sub_out in outs:
        res = remove_redundant_cast_op(sub_out, outs)
        # has remove the redundant_cast_op
        if res != sub_out:
            sub_out = res

    # get op info
    op_info = get_op_info(outs)

    # set pattern
    op_pattern = op_info['pattern']
    if op_pattern == OpPatterns.MATMUL_PATTERN and len(outs) == 1:
        all_tags = get_all_tags(outs[0])
        op_info['pattern'] = set_op_pattern(all_tags, op_info)
    op_pattern = op_info['pattern']

    fusion_manager.set_current_op_pattern(op_pattern.value)

    # to list placeholder type tensor
    input_tensors = op_info["input_tensors"]
    # to list compute type tensor
    mid_tensors = op_info["mid_tensors"]
    output_tensors = op_info["output_tensors"]
    compute_tensors = mid_tensors + output_tensors

    tensor_map = op_info["tensor_map"]

    for index, out in enumerate(outs):
        # check whether needs to cast back data type into original type
        # suppose YES, compute tensor should be more than one
        if ((verify_compute_tensor(compute_tensors))  # 'pylint: disable=R0916
                and input_tensors
                and out.op.tag not in ["matmul", "matmul_gemv",
                                       "elewise_binary_logic|and",
                                       "gemm", "matmul_v2_gemv",
                                       "emit_insn_elewise_multiple_sel|bit",
                                       "NZ_trans_ND", "fixpipe_reform"]):
            in_dtype = input_tensors[0].dtype
            if _skip_cast_op(in_dtype, out, op_pattern):
                continue
            if in_dtype and out.dtype != in_dtype:
                real_out = cast.cast_to(out, in_dtype)

                def _dfs_tensor_graph(tensor):
                    """
                    dfs tensor graph
                    """
                    for current_tensor in list(
                            tensor.op.input_tensors):  # 'pylint: disable=cell-var-from-loop
                        if current_tensor in op_info["mid_tensors"] or \
                                current_tensor in op_info["input_tensors"]:
                            # record the tensor map input:[outputs]
                            if current_tensor in tensor_map.keys():
                                tensor_map[current_tensor].insert(0, tensor)
                            else:
                                tensor_map[current_tensor] = [tensor]
                            continue
                        else:
                            # record the tensor
                            if current_tensor == out:  # 'pylint: disable=cell-var-from-loop
                                input_tensors.insert(0, current_tensor)
                            else:
                                mid_tensors.insert(0, current_tensor)

                                _dfs_tensor_graph(current_tensor)

                        # record the tensor map input:[outputs]
                        if current_tensor in tensor_map.keys():
                            tensor_map[current_tensor].insert(0, tensor)
                        else:
                            tensor_map[current_tensor] = [tensor]

                _dfs_tensor_graph(real_out)
                outs[index] = real_out

    # retry use rl infer
    schedule = None
    if not get_current_build_config("enable_op_prebuild"):
        try:
            if option is not None and isinstance(option, dict):
                schedule, tensor_list, real_outs = rl_search_proc(outs, option)
            enable_bank_query = True
            if get_context() and (get_context().get_addition("enable_rl_bank_query") is False):
                enable_bank_query = False
            if schedule is None and enable_bank_query:
                ret, schedule = rl_bank.query_rl_bank(outs, op_info=op_info)
                if ret and isinstance(schedule, tvm.schedule.Schedule):
                    return schedule
        except Exception:  # 'pylint: disable=broad-except
            schedule = None

    # if not rl or rl is not supported, use auto_schedule.
    if schedule is None:
        schedule, tensor_list, real_outs = global_core_schedule(outs,
                                                                op_info=op_info)

    if schedule is None:
        schedule, tensor_list, real_outs = global_core_schedule(outs,
                                                                templet_name='speel',
                                                                op_info=op_info)
        if schedule is None:
            cce_op = CceOp(scope_ubuf, need_tensorize=True,
                           need_pragma=True)
            if len(outs) > 1:
                dict_args = {"errCode": "E90003",
                             "detailed_cause": f"cpu schedule not support multi-output, outputs num is [{len(outs)}]"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            schedule = cce_op.cpu_schedule(outs[0])
    schedule.cce_special = {}

    # spec_node_list
    schedule.cce_special["tensor_list"] = tensor_list
    # the origin out tensor list
    schedule.cce_special["orign_out_tensor"] = origin_outs
    # the real out tensor list
    schedule.cce_special["real_out_tensor"] = real_outs

    return schedule


def check_is_need_cast(out):
    """
    Check if tensor needs to do cast operation

    Parameters
    ----------
    out : output tensor

    Returns
    -------
    Bool : true or false
    """
    def crawl_anti_quant_tensor(out):
        queue = [out]
        visited = []
        while queue:
            head = queue.pop(0)
            for tensor in head.op.input_tensors:
                if tensor in visited:
                    continue
                if tensor.op.tag.startswith('anti_quant'):
                    return True
                visited.append(tensor)
                queue.append(tensor)
        return False

    str_list = out.op.tag.split("|")
    if 'emit_insn_elewise_binary_cmp' in str_list:
        return False
    if 'depthwise_conv2d' in str_list:
        return False
    if 'quant' in str_list:
        return False
    if out.op.tag.startswith('dequant'):
        return False
    if out.op.tag.startswith('anti_quant'):
        return False
    if out.op.tag.startswith('requant'):
        return False
    if out.op.tag.startswith("pooling2d_"):
        return False
    if out.op.tag.startswith("pooling3d_"):
        return False
    if out.op.name.find("write_select") >= 0 or \
            out.op.name.find("strided_write") >= 0:
        if crawl_anti_quant_tensor(out):
            return False
        pre_op_str_list = out.op.input_tensors[0].op.tag.split("|")
        if 'quant' in pre_op_str_list or "requant_remove_pad" in pre_op_str_list:
            return False
        if "dequant_remove_pad" in pre_op_str_list:
            return False
    return True


def check_support_muti_output(outs):
    """
    check wether support muti_output, only support all elewise/reduce/broadcast compute now
    :param outs:output list
    :return: True or Flase
    """
    operation_list = []
    for tensor in outs:
        operation_list.append(tensor)

    # dfs all compute node to check wheather has non-elewise/reduce/broadcast operation
    while operation_list:
        tmp_operation_list = []

        for sub_opt in operation_list:
            tag = sub_opt.op.tag
            if 'matmul' in tag:
                return True
            if "cube_layer_norm" in tag:
                return True
            if "tail_block_pretreatment" in tag:
                return True
            if "elewise" not in tag and "broadcast" not in tag and \
                    "reduce" not in tag and not ConvParam.convbn1_flag and \
                    not ConvParam.conv_deq_req_double_out and \
                    not ConvParam.conv_reluv2_flag and \
                    "requant_s16" not in tag:
                return False

            for sub_input in sub_opt.op.input_tensors:
                if not isinstance(sub_input.op, tvm.tensor.PlaceholderOp) and \
                        sub_input not in tmp_operation_list:
                    tmp_operation_list.append(sub_input)

        operation_list = tmp_operation_list

    return True


def decl_memory(buffer_scope):
    """
    decl memory
    """
    fun = tvm.get_global_func("tvm.info.mem.%s" % buffer_scope, True)
    if fun is not None:
        return

    try:
        @tvm.register_func(
            "tvm.info.mem.%s" % buffer_scope)  # 'pylint: disable=unused-variable, protected-access
        def mem_info_ub_buffer():  # 'pylint: disable=unused-variable,
            return tvm.make.node("MemoryInfo",
                                 unit_bits=32 * 8,
                                 max_simd_bits=32 * 8,
                                 max_num_bits=get_soc_spec("UB_SIZE") * 8, 
                                 head_address=tvm.const(0, 'int32'))
    except tvm._ffi.base.TVMError:  # 'pylint: disable=W0212
        dict_args = {"errCode": "E90003", "detailed_cause": "declare memory failed!"}
        raise RuntimeError(dict_args, get_error_message(dict_args))


def check_spec_node(tensor):
    """
    check whether the tensor is not special node
    --> none elewise/broadcast/transpose
    --> ExternOp, such as dim_conv
    """
    if tensor.op.tag:
        str_list = tensor.op.tag.split("|")
        if ('elewise' not in str_list[0].split('_')) and (
                'transpose' not in str_list[0].split('_')) \
                and ('poolinginput' not in str_list[0].split('_')):
            return True
        return False
    if isinstance((tensor.op), tvm.tensor.ExternOp):
        return True
    return False


def check_spec_node_matmul_before(tensor):
    """
    a=abs(A) ; c=matmul(a,b);then a is spec
    or
    a=abs(A) ; c=pooling2d(a);then a is spec
    """
    if tensor.op.tag:
        str_list = tensor.op.tag.split("|")
        list_tag = tensor.op.input_tensors[0].op.tag
        if 'transpose' in str_list[0].split('_') or \
                'poolinginput' in str_list[0].split('_'):
            if 'elewise' in list_tag.split('_'):
                return True

    return False


PATTERN_SCHEDULE_FUNC_MAP = {
    OpPatterns.ASCEND_DEQUANT_S16_PATTERN: ascend_dequant_s16_schedule,
    OpPatterns.ASCEND_REQUANT_PATTERN: ascend_requant_schedule,
    OpPatterns.ASCEND_REQUANT_S16_PATTERN: ascend_requant_s16_schedule,
    OpPatterns.ASCEND_QUANT_PATTERN: ascend_quant_schedule,
    OpPatterns.ASCEND_DEQUANT_PATTERN: ascend_dequant_schedule,
    OpPatterns.ASCEND_ANTI_QUANT_PATTERN: ascend_anti_quant_schedule,
    OpPatterns.STRIDED_READ_PATTERN: strided_read_schedule,
    OpPatterns.STRIDED_WRITE_PATTERN: strided_write_schedule,
    OpPatterns.BN_REDUCE_PATTERN: bn_reduce_schedule,
    OpPatterns.BN_UPDATE_PATTERN: bn_update_schedule,
    OpPatterns.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_PATTERN: logits_schedule,
    OpPatterns.LAYER_NORM_GRAD_PATTERN: layer_norm_grad_schedule,
    OpPatterns.L2LOSS_MUL_ADDN_PATTERN: l2loss_mul_addn_schedule,
    OpPatterns.COSINE_EMBEDDING_LOSS_PATTERN: cosine_embedding_loss_schedule,
    OpPatterns.FIXPIPE_PATTERN: fixpipe_schedule,
}


def comm_pattern_schedule(pattern, op_info, outs):
    """
    comm_pattern_schedule for QUANT_PATTERN and BN_PATTERN
    """
    if pattern in [OpPatterns.ASCEND_QUANT_PATTERN,
                   OpPatterns.ASCEND_DEQUANT_PATTERN,
                   OpPatterns.ASCEND_ANTI_QUANT_PATTERN,
                   OpPatterns.STRIDED_READ_PATTERN,
                   OpPatterns.STRIDED_WRITE_PATTERN,
                   OpPatterns.ASCEND_REQUANT_PATTERN,
                   OpPatterns.ASCEND_DEQUANT_S16_PATTERN,
                   OpPatterns.ASCEND_REQUANT_S16_PATTERN,
                   OpPatterns.FIXPIPE_PATTERN]:
        spec_mid_list = []
        input_tensors = op_info["input_tensors"]
        sch_func = PATTERN_SCHEDULE_FUNC_MAP[pattern]
        schedule = sch_func(outs[0], input_tensors)
        return schedule, spec_mid_list, outs

    if pattern == OpPatterns.BN_UPDATE_PATTERN:
        spec_mid_list = []
        input_tensors = op_info["input_tensors"]
        sch_func = PATTERN_SCHEDULE_FUNC_MAP[pattern]
        schedule = sch_func(outs, input_tensors)
        return schedule, spec_mid_list, outs

    if pattern == OpPatterns.READ_SELECT_PATTERN:
        spec_mid_list = []
        input_tensors = op_info["input_tensors"]
        schedule = read_select_schedule(outs[0], input_tensors)
        return schedule, spec_mid_list, outs

    if pattern == OpPatterns.WRITE_SELECT_PATTERN:
        spec_mid_list = []
        input_tensors = op_info["input_tensors"]
        schedule = write_select_schedule(outs[0], input_tensors)
        return schedule, spec_mid_list, outs

    return None, [], outs


def global_core_schedule(  # 'pylint: disable=R0911, R0912, R0914, R0915
        outs, templet_name="global", op_info=None):
    """
    global core schedule
    """
    input_origin_outs = copy.copy(outs)

    def __update_tensor_map(tensor_map):
        """
        update the dst_tensor list of the tensor with more than one dst_tensor
        tensor_map = {input: [outputs(is spec node)]}
        """
        for sub_ten in tensor_map:
            if len(tensor_map[sub_ten]) > 1:
                for idx in range(len(tensor_map[sub_ten])):
                    tmp = tensor_map[sub_ten][idx]
                    while tmp in tensor_map:
                        if tmp not in tensor_map.keys():
                            break
                        if (op_info.get('type') == OpSpecTypes.BN_EXT2
                                or len(tensor_map[tmp]) == 1) \
                                and (not check_spec_node(tmp)):
                            tensor_map[sub_ten][idx] = tensor_map[tmp][0]
                            tmp = tensor_map[sub_ten][idx]
                        else:
                            break
            tensor_map[sub_ten] = list(set(tensor_map[sub_ten]))

    def __find_spec_node(tensor_map):  # 'pylint: disable=too-many-branches, too-many-statements
        '''
        1. not-elewise node
        2. elewise connet to two diff not-elewise node
        3. res
        4. ExternOp
        5. inputs of ExternOp
        '''
        res_list = []
        spec_mid_list = []
        # build a map to indicate which schedule should be assigned for each op
        sch_map = {}

        # first, base on the tensor map find the spec node
        for tensor_i in tensor_map:
            if (tensor_i not in spec_mid_list) and \
                    tensor_i.op.input_tensors:
                if check_spec_node(tensor_i):
                    spec_mid_list.append(tensor_i)
                if check_spec_node_matmul_before(tensor_i):
                    # a=abs(A) ; c=matmul(a,b);then a is spec
                    spec_mid_list.append(tensor_i.op.input_tensors[0])

            for tensor_j in tensor_map[tensor_i]:
                if tensor_j not in tensor_map:
                    if tensor_j not in res_list:
                        res_list.append(tensor_j)
                else:
                    if (tensor_j not in spec_mid_list) and \
                            tensor_j.op.input_tensors:
                        if check_spec_node(tensor_j):
                            spec_mid_list.append(tensor_j)

        # then, tree node such as a for b and a for c, a as spec node
        spec_node_found = True
        while spec_node_found:
            spec_node_found = False
            for tensor_i in tensor_map:
                if (tensor_i not in spec_mid_list) and \
                        tensor_i.op.input_tensors:
                    tmp_set = set(tensor_map[tensor_i]).intersection(
                        set(spec_mid_list + res_list))
                    if len(tmp_set) > 1:
                        spec_node_found = True
                        spec_mid_list.append(tensor_i)

        # then, rule for extern, find the input of externop as spec node
        tmp_spec_list = spec_mid_list + res_list
        for tensor_i in tmp_spec_list:
            if isinstance((tensor_i.op), tvm.tensor.ExternOp):
                for key in tensor_i.op.input_tensors:
                    if (not isinstance((key.op), tvm.tensor.PlaceholderOp)) \
                            and (key not in tmp_spec_list):
                        tmp_spec_list.append(key)
                        spec_mid_list.append(key)

        # add sch_map for spec node
        for key in spec_mid_list + res_list:
            if isinstance((key.op), tvm.tensor.ExternOp):
                sch_map[key] = "tvm-extern"
            else:
                sch_map[key] = key.op.tag.split('|')[0]

        # then, rules for elewise_single after reduce
        tmp_spec_mid_list = spec_mid_list[:]
        for tensor_i in tmp_spec_mid_list:
            if not isinstance((tensor_i.op), tvm.tensor.ExternOp):
                update_tensor = None
                if len(tensor_map[tensor_i]) < 2:
                    tmp = tensor_map[tensor_i][0]
                    if tensor_i.op.tag.count("broadcast"):
                        continue
                    # need to update tensor for elewise_single
                    while (tmp.op.tag.count("elewise_single") and
                           ((tmp not in spec_mid_list) or (tmp in res_list))):
                        update_tensor = tmp
                        if (tmp not in res_list) and (len(tensor_map[tmp]) < 2):
                            tmp = tensor_map[tmp][0]
                        else:
                            break
                    # if c=matmul(a,b);d=abs(c); then c is spec,
                    # can not remove from spec_mid_list
                    if (update_tensor is not None) and \
                            tensor_i.op.tag not in ("matmul",
                                                    "gemm",
                                                    "matmul_gemv",
                                                    "matmul_v2_gemv",
                                                    "pooling2d_avg",
                                                    "pooling2d_max",
                                                    "pooling2d_gap",
                                                    "pooling2d_gmp"):
                        sch_map[update_tensor] = sch_map.get(tensor_i)
                        if update_tensor not in res_list:
                            spec_mid_list.append(update_tensor)
                        spec_mid_list.remove(tensor_i)
                        sch_map.pop(tensor_i)

        # then, rules for broadcast
        tmp_spec_mid_list = spec_mid_list[:]
        for tensor_i in tmp_spec_mid_list:
            if tensor_i.op.tag.count("broadcast") and len(
                    tensor_map[tensor_i]) < 2:
                tmp = tensor_map[tensor_i][0]
                while tmp not in (spec_mid_list + res_list):  # 'pylint: disable=superfluous-parens
                    tmp = tensor_map[tmp][0]
                if tmp.op.tag.count("reduce") or tmp.op.tag.count("elewise"):
                    spec_mid_list.remove(tensor_i)
                    sch_map.pop(tensor_i)

        return spec_mid_list, res_list, sch_map

    def __gen_muti_output_spec_node_map(sepc_node_list, outs, tensor_map):
        """
        get every output can attach which spec_node
        1 if out is a spec node, the attched spec node is itself
        2 if out is not a spec node, we should find its attach spec node
        :param sepc_node_list:
        :param outs:the muti outs
        :param tensor_map:[intput:output_list]
        :return: map: {spec_node1:{out1, out2}, spec_node2:{out3}}
        """
        spec_node_to_output_map = {}
        for spec_node in sepc_node_list:
            spec_node_to_output_map[spec_node] = [spec_node]

        def __find_attatch_spec_node(tensor):
            """
             find the spec_node which the tensor attach
            :param tensor: tensor
            :return: the spec_node which the tensor attach
            """
            if tensor in sepc_node_list:
                return tensor

            if tensor in tensor_map.keys():
                tensor_list = tensor_map[tensor]
                for output in tensor_list:
                    spec_node = __find_attatch_spec_node(output)
                    if spec_node is not None:
                        return spec_node

            return None

        # find every out attached to spec
        for out in outs:
            spec_node = __find_attatch_spec_node(out)
            if spec_node is not None and \
                    out not in spec_node_to_output_map.get(spec_node):
                spec_node_to_output_map[spec_node].append(out)

        return spec_node_to_output_map

    def __specific_schedule(tensor, scope, sch_list):
        op_param_list = {}
        op_param_list = {
            'segment': [templet_name, tensor, scope, spec_node_list, sch_list],
            'inplace': [tensor, scope, spec_node_list, sch_list],
            'concat': [tensor, scope, spec_node_list, sch_list],
            'gemm': [tensor, sch_list],
            'matmul': [tensor, sch_list],
            'convolution': [tensor, scope, spec_node_list, sch_list],
            'conv2d_backprop_input': [tensor, spec_node_list, sch_list],
            'conv2d_backprop_filter': [tensor, spec_node_list, sch_list],
            'pooling2d': [tensor, sch_list],
            'pooling3d': [tensor, sch_list],
            'dilation': [tensor, sch_list]
        }

        schedule_dispatch = ScheduleDispatch()
        for key, val in op_param_list.items():
            # 'pylint: disable=too-many-function-args
            if key in sch_map[tensor]:
                return schedule_dispatch.handle_case(key, *val)
        return None

    def __reduce_schedule(tensor, scope, sch_list):
        need_enable_muticore = True
        # if there is spec node , can not enbale muti-core
        if len(spec_node_list) > 1:
            need_enable_muticore = False

        if templet_name == 'global':
            cce_op = CceOp(scope, need_tensorize=True, need_pragma=True,
                           need_enable_muticore=need_enable_muticore)
        elif templet_name == 'speel':
            cce_op = CceSpeelOp(scope, need_tensorize=True,
                                need_pragma=True,
                                need_enable_muticore=need_enable_muticore)

        return cce_op.core_schedule_reduce(spec_node_to_out_map[tensor],
                                           spec_node_list,
                                           sch_list, tensor_map)

    def __assign_schedule(tensor, scope, sch_list):
        """First call specific schedule.If Failed, then call reduce schedule"""

        res = __specific_schedule(tensor, scope, sch_list)

        if res is not None:
            return res

        return __reduce_schedule(tensor, scope, sch_list)

    # ugly, for complexity check
    def __do_schedule_wrapper(reduce_multi_sch, temp_outs):
        try:
            schedule_valid, sch = reduce_multi_sch.do_schedule(temp_outs,
                                                               None, [])
        except Exception:  # 'pylint: disable=broad-except
            return False, None
        return schedule_valid, sch

    # since a node may connect mul-node, so its traverse order should be order-
    # ed, it chould be achieved by python built-in orderedDict

    pattern = op_info["pattern"]
    tensor_map = op_info["tensor_map"]
    pre_pattern = None

    if pattern in [
            OpPatterns.ASCEND_QUANT_PATTERN,
            OpPatterns.ASCEND_DEQUANT_PATTERN,
            OpPatterns.ASCEND_ANTI_QUANT_PATTERN,
            OpPatterns.ASCEND_DEQUANT_S16_PATTERN,
            OpPatterns.ASCEND_REQUANT_PATTERN,
            OpPatterns.ASCEND_REQUANT_S16_PATTERN,
            OpPatterns.STRIDED_READ_PATTERN,
            OpPatterns.STRIDED_WRITE_PATTERN,
            OpPatterns.BN_UPDATE_PATTERN,
            OpPatterns.READ_SELECT_PATTERN,
            OpPatterns.WRITE_SELECT_PATTERN,
            OpPatterns.FIXPIPE_PATTERN,
    ]:
        outs_bak = list(outs)
        schedule, spec_mid_list, outs = comm_pattern_schedule(pattern, op_info, outs)

        if schedule is not None:
            return schedule, spec_mid_list, outs

        pattern = OpPatterns.ELEMWISE_PATTERN
        outs = outs_bak

    elif pattern in [OpPatterns.BN_REDUCE_PATTERN,
                     OpPatterns.LAYER_NORM_GRAD_PATTERN,
                     OpPatterns.L2LOSS_MUL_ADDN_PATTERN]:
        spec_mid_list = []
        input_tensors = op_info["input_tensors"]
        sch_func = PATTERN_SCHEDULE_FUNC_MAP[pattern]
        sch = sch_func(outs, input_tensors)
        if sch is not None:
            return sch, spec_mid_list, outs

        # if can't find valid sch, use common sch
        op_info = get_op_info(outs)
        tensor_map = op_info["tensor_map"]

    elif pattern in [OpPatterns.COSINE_EMBEDDING_LOSS_PATTERN,
                     OpPatterns.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_PATTERN]:
        input_tensors = op_info["input_tensors"]
        outs_bak = list(outs)
        sch_func = PATTERN_SCHEDULE_FUNC_MAP[pattern]
        sch, spec_mid_list = sch_func(outs, input_tensors)
        if sch is not None:
            return sch, spec_mid_list, outs

        # if can't find valid sch, use common sch
        pattern = OpPatterns.OPAQUE_PATTERN
        outs = outs_bak

    elif pattern == OpPatterns.SOFTMAX_PATTERN:
        spec_mid_list = []
        outs_bak = list(outs)
        schedule, spec_mid_list = SoftmaxSchedule().do_schedule(outs)
        if schedule is not None:
            return schedule, spec_mid_list, outs

        # if can't find valid sch, use old way process
        op_info["sub_pattern"] = OpPatterns.SOFTMAX_PATTERN
        pre_pattern = pattern
        pattern = OpPatterns.OPAQUE_PATTERN
        outs = outs_bak

    elif pattern == OpPatterns.L2_LOSS_PATTERN:
        spec_mid_list = []
        input_tensors = op_info["input_tensors"]
        outs_bak = list(outs)
        schedule = l2_loss_schedule(outs, input_tensors)
        if schedule is not None:
            return schedule, spec_mid_list, outs
        pattern = OpPatterns.OPAQUE_PATTERN
        outs = outs_bak

    elif pattern == OpPatterns.BN_UPDATE_GRAD_PATTERN:
        spec_mid_list = []
        input_tensors = op_info["input_tensors"]
        shape_x = shape_to_list(input_tensors[-1].shape)
        if len(shape_x) == 4:
            schedule = bn_update_grad_schedule_nd(outs, input_tensors)
        else:
            schedule = bn_update_grad_schedule(outs, input_tensors)

        if schedule:
            return schedule, spec_mid_list, outs

        # if can't find valid sch, use old way process
        op_info = get_op_info(outs)
        tensor_map = op_info["tensor_map"]
        pattern = op_info["pattern"]

    # solve reduce atomic compile performance
    tuple_reduce_flag = False
    for tensor in outs:
        if tensor.op.tag.find("tuple_reduce_sum") != -1:
            tuple_reduce_flag = True
            break
    is_reduce_multi_pattern = not tuple_reduce_flag and \
        len(outs[0].shape) != 1 and \
        pattern != OpPatterns.POOL2D_PATTERN and \
        pattern != OpPatterns.ELEMWISE_PATTERN and \
        op_info.get('type') != OpSpecTypes.MVNV2

    if is_reduce_multi_pattern:
        # try use multi reduce template
        reduce_multi_sch = ReduceMultiSchedule()
        temp_outs = copy.copy(input_origin_outs)
        reduce_multi_sch.set_op_type(op_info.get('type'), op_info.get("sub_pattern"))
        schedule_valid, sch = __do_schedule_wrapper(reduce_multi_sch,
                                                    temp_outs)
        if schedule_valid:
            return sch, [], temp_outs

    # A->B->res1-->res2, output:[res1, res2], we only need create res2's op, res2 contains res1
    schedule = tvm.create_schedule(
        [res.op for res in outs if res not in tensor_map])
    sch_list = [schedule]
    schedule_valid = True
    spec_node_list = []
    spec_mid_list = []
    sch_map = {}
    spec_node_to_out_map = {}

    if 'type' in op_info.keys():
        spec_type = op_info['type']
        if spec_type == OpSpecTypes.REDUCE_MEAN_2D_ALIGNED_MID_REDUCE_NO_CAST:
            schedule_valid = reduce_mean_mid_reduce_high_performance_schedule(outs, sch_list)
            if schedule_valid:
                return schedule, spec_mid_list, outs
    if 'sub_pattern' in op_info.keys() and pattern != OpPatterns.MATMUL_PATTERN:
        sub_pattern = op_info['sub_pattern']
        if sub_pattern == OpSubPatterns.REDUCE_ATOMIC_PATTERN:
            atomic_sch = ReduceAtomicSchedule()
            schedule_valid = atomic_sch.do_schedule(outs, sch_list, [])
            if schedule_valid:
                op_info['pattern'] = OpSubPatterns.REDUCE_ATOMIC_PATTERN
                return schedule, spec_mid_list, outs
        if sub_pattern == OpSubPatterns.REDUCE_5HDC_PATTERN:
            fhdc_sch = Reduce5HDCSchedule()
            schedule_valid = fhdc_sch.do_schedule(outs, sch_list, [])
            if schedule_valid:
                op_info['pattern'] = OpSubPatterns.REDUCE_5HDC_PATTERN
                return schedule, spec_mid_list, outs
    if pattern == OpPatterns.L2_NORMALIZE_PATTERN:
        l2_normalize_sch = L2NormalizeSchedule()
        spec_mid_list = []
        outs_bak = list(outs)
        schedule_valid = l2_normalize_sch.do_schedule(outs, schedule, [])
        if schedule_valid:
            return schedule, spec_mid_list, outs
        pattern = OpPatterns.OPAQUE_PATTERN
        outs = outs_bak

    if pattern == OpPatterns.BN_GRAD_REDUCE_PATTERN:
        bn_grad_reduce_sch = BnGradReduceSchedule()
        outs_bak = list(outs)
        schedule_valid = bn_grad_reduce_sch.do_schedule(outs, sch_list, [])
        if schedule_valid:
            return schedule, spec_mid_list, outs_bak
        outs = outs_bak

    if pattern == OpPatterns.ELEMWISE_PATTERN:
        if len(op_info['output_tensors']) > 1:
            elewise_multi_sch = ElewiseMultiSchedule()
            elewise_multi_sch.set_op_type(op_info.get('type'), op_info.get("sub_pattern"))
            temp_outs = copy.copy(input_origin_outs)
            schedule_valid = elewise_multi_sch.do_schedule(temp_outs,
                                                           sch_list,
                                                           spec_mid_list)
            if schedule_valid:
                outs = temp_outs
            else:
                # elemwise_pattern has no spec_node, to remove spec_node
                __update_tensor_map(tensor_map)
                spec_mid_list, res_list, sch_map = __find_spec_node(tensor_map)
                spec_node_list = spec_mid_list + res_list
                spec_node_to_out_map = __gen_muti_output_spec_node_map(
                    spec_node_list, outs, tensor_map)
                # split tag for input tensor
                for out in outs:
                    if not out.op.input_tensors:
                        spec_node_list.append(out)
                        spec_node_to_out_map[out] = [out]
                        sch_map[out] = out.op.tag.split('|')[0]

            need_enable_muticore = True
            # there is spec node, can not enable muti core
            if len(spec_node_list) > 1:
                need_enable_muticore = False
            schedule_index = 0
            for tensor_i in spec_node_list:
                scope_name = scope_ubuf
                if schedule_index > 0:
                    scope_name = scope_name + str(schedule_index)
                    decl_memory(scope_name)
                schedule_index += 1
                elewise_sch = ElewiseSchedule(need_enable_muticore)
                schedule_valid = elewise_sch.do_schedule(
                    spec_node_to_out_map[tensor_i], sch_list, spec_node_list)
        else:
            elewise_sch = ElewiseSchedule()
            elewise_sch.set_op_type(op_info.get('type'), op_info.get("sub_pattern"))
            schedule_valid = elewise_sch.do_schedule(input_origin_outs,
                                                     sch_list,
                                                     spec_node_list)
    elif pattern == OpPatterns.REDUCE_PATTERN:
        if len(op_info['output_tensors']) > 1:
            __update_tensor_map(tensor_map)
            spec_mid_list, res_list, sch_map = __find_spec_node(tensor_map)
            spec_node_list = spec_mid_list + res_list
            spec_node_to_out_map = __gen_muti_output_spec_node_map(
                spec_node_list, outs, tensor_map)

            # split tag for input tensor
            for out in outs:
                if not out.op.input_tensors:
                    spec_node_list.append(out)
                    spec_node_to_out_map[out] = [out]
                    sch_map[out] = out.op.tag.split('|')[0]

            need_enable_muticore = True
            # there is spec node, can not enable muti core
            if len(spec_node_list) > 1:
                need_enable_muticore = False
            schedule_index = 0
            for tensor_i in spec_node_list:
                scope_name = scope_ubuf
                if schedule_index > 0:
                    scope_name = scope_name + str(schedule_index)
                    decl_memory(scope_name)
                schedule_index += 1

                if templet_name == 'global':
                    cce_op = CceOp(scope_name, need_tensorize=True,
                                   need_pragma=True,
                                   need_enable_muticore=need_enable_muticore)
                elif templet_name == 'speel':
                    cce_op = CceSpeelOp(scope_name, need_tensorize=True,
                                        need_pragma=True,
                                        need_enable_muticore=need_enable_muticore)

                schedule_valid, _ = cce_op.core_schedule_reduce(
                    spec_node_to_out_map[tensor_i],
                    spec_node_list, sch_list, tensor_map)
        else:
            need_enable_muticore = True

            if templet_name == 'global':
                cce_op = CceOp(scope_ubuf, need_tensorize=True,
                               need_pragma=True,
                               need_enable_muticore=need_enable_muticore)
            elif templet_name == 'speel':
                cce_op = CceSpeelOp(scope_ubuf, need_tensorize=True,
                                    need_pragma=True,
                                    need_enable_muticore=need_enable_muticore)

            schedule_valid, _ = cce_op.core_schedule_reduce(
                input_origin_outs, outs, sch_list, tensor_map)
    elif pattern == OpPatterns.SEGMENT_PATTERN:
        if templet_name == 'global':
            cce_segment_op = CceSegmentOp(scope_ubuf,
                                          need_tensorize=True, need_pragma=True)
        elif templet_name == 'speel':
            cce_segment_op = CceSegmentSpeelOp(scope_ubuf,
                                               need_tensorize=True,
                                               need_pragma=True)
        schedule_valid = cce_segment_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.INPLACE_PATTERN:
        cce_inplace_op = CceInplaceOp(scope_ubuf)
        schedule_valid = cce_inplace_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.PURE_BROADCAST_PATTERN:
        pure_broadcast_schedule = PureBroadcastSchedule()
        sch_list = [pure_broadcast_schedule.do_schedule(outs[0], sch_list[0])]
        outs = pure_broadcast_schedule.get_real_outs()
    elif pattern == OpPatterns.CONV_PATTERN:
        cce_conv_op = CceConvOp()
        if ConvParam.convbn1_flag:
            cce_conv_op.schedule(outs[-1], outs, sch_list,
                                 ConvParam.convbn1_flag)
        else:
            cce_conv_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.DEPTHWISECONV_PATTERN:
        cce_depthwise_conv_op = depthwise_conv2d_schedule(outs[0])
        sch_list[0] = cce_depthwise_conv_op
    elif pattern == OpPatterns.CONV3D_PATTERN:
        cce_conv3d_op = CceConv3dOp(scope_ubuf, need_tensorize=True,
                                    need_pragma=True)
        schedule_valid = cce_conv3d_op.do_schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.CONV2D_BACKPROP_INPUT_PATTERN:
        cce_conv2d_backprop_input_op = CceConv2dBackpropInputOp(
            scope_ubuf,
            need_tensorize=True, need_pragma=True)
        cce_conv2d_backprop_input_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.CONV3D_BACKPROP_INPUT_PATTERN:
        cce_conv3d_backprop_input_op = CceConv3dBackpropInputOp(
            scope_ubuf,
            need_tensorize=True, need_pragma=True)
        cce_conv3d_backprop_input_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.CONV3D_BACKPROP_FILTER_PATTERN:
        cce_conv3d_backprop_filter_op = CceConv3dBackpropFilterOp(
            scope_ubuf,
            need_tensorize=True, need_pragma=True)
        cce_conv3d_backprop_filter_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.CONV2D_BACKPROP_FILTER_PATTERN:
        cce_conv2d_backprop_filter_op = CceConv2dBackpropFilterOp(
            scope_ubuf,
            need_tensorize=True, need_pragma=True)
        cce_conv2d_backprop_filter_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.CUBE_LAYER_NORM:
        cce_cube_layer_norm_op = CceLayerNormCubeOp(scope_ubuf)
        cce_cube_layer_norm_op.schedule(outs[0], outs, sch_list)
    elif pattern == OpPatterns.MATMUL_PATTERN:
        mmad_schedule(outs, sch_list)  # 'pylint: disable=W0631
    elif pattern == OpPatterns.GEMM_PATTERN:
        gemm_schedule(outs[0], sch_list)
    elif pattern == OpPatterns.POOL2D_PATTERN:
        pooling2d_schedule(outs[0], sch_list)
    elif pattern == OpPatterns.POOL3D_PATTERN:
        pooling3d_schedule(outs[0], sch_list)
    else:
        __update_tensor_map(tensor_map)

        spec_mid_list, res_list, sch_map = __find_spec_node(tensor_map)

        # res_list is the leaf node
        # spec_mid_list is the node that will produce workspace.
        spec_node_list = spec_mid_list + res_list
        # when has muti outs, we should know the out in outs follow which spec_node,
        # they should in one schedule
        spec_node_to_out_map = __gen_muti_output_spec_node_map(
            spec_node_list, outs, tensor_map)

        workspace_info = {}
        workspace_multi_schedule = WorkspaceMultiSchedule(
            pre_pattern == OpPatterns.SOFTMAX_PATTERN or op_info.get('type') == OpSpecTypes.MVN)
        workspace_multi_schedule.set_op_type(op_info.get('type'))
        workspace_multi_schedule.update_schedule(spec_node_list, sch_list,
                                                 spec_mid_list)
        # split tag for input tensor
        for out in outs:
            if not out.op.input_tensors:
                spec_node_list.append(out)
                spec_node_to_out_map[out] = [out]
                sch_map[out] = out.op.tag.split('|')[0]

        schedule_index = 0
        for tensor_i in spec_node_list:
            if isinstance(tensor_i.op, tvm.tensor.ExternOp):
                continue
            scope_name = scope_ubuf
            if schedule_index > 0:
                scope_name = scope_name + str(schedule_index)
                decl_memory(scope_name)
            schedule_index += 1

            schedule_valid = __assign_schedule(tensor_i, scope_name, sch_list)
            if isinstance(schedule_valid, tuple):
                schedule_valid, cur_info = schedule_valid
                workspace_info[tensor_i] = cur_info
            if not schedule_valid:
                break
        workspace_multi_schedule.do_schedule(workspace_info)

    schedule = sch_list[0]

    if schedule_valid:
        if len(sch_list) > 1:
            return schedule, spec_mid_list, sch_list[1:]
        return schedule, spec_mid_list, outs
    return None, spec_mid_list, outs


def cce_build_code(  # 'pylint: disable=R0912, R0914, R0915
        sch, config_map=None):
    """
    API of building or printing lower code, just can be used when device is CCE

    Parameters
    ----------
    sch : tvm.schedule
        schedule to build or to print lower code

    config_map : dict, default is {} and use default configration

        key_words:

            print_ir : if need print lower IR code, default is True

            need_build : if need build, default is True

            name : kernel name, default is cce_op

    Returns
    -------
    None
    """
    warnings.warn("cce_build_code is expired, please replace it with the func build in cce",
                  DeprecationWarning)
    if get_current_build_config("enable_op_prebuild") and \
            not get_soc_spec("CUBE_VECTOR_SPLIT"):
        te_util.L1CommonParam.l1_fusion_tensors_map = None
        return

    def _write_workspace_info(workspace_list, kernel_name):
        """
        write workspace info
        """
        def write_code(wkspace_dict, fname):
            """
            write code
            """
            fname = os.path.realpath(fname)
            if os.path.exists(fname):
                with open(fname, "r") as file_name:
                    load_dict = json.load(file_name)

                load_dict.update(wkspace_dict)
                with open(fname, "w") as file_name:
                    json.dump(load_dict, file_name,
                                sort_keys=True, indent=4,
                                separators=(',', ':'))
            else:
                dict_args = {"errCode": "E90001",
                             "detailed_cause": f"The file [{fname}] does not exist, please check the path."}
                raise RuntimeError(dict_args, get_error_message(dict_args))

        def _shape_to_list(shape):
            """
            translate tvm.shape to list type in python
            """
            tmp = []
            for i in shape:
                tmp.append(i.value)
            return tmp

        def get_data_width(dtype):
            """
            get data width
            """
            m_sea = re.search(r'\d+', dtype)
            if m_sea:
                return int(m_sea.group(0)) // 8
            return 0

        num = len(workspace_list)
        if num:
            shape_list = [_shape_to_list(i.shape) for i in workspace_list]
            total_size = [functools_reduce(lambda x, y: x * y, list_i) for
                          list_i in shape_list]

            for i, element in enumerate(workspace_list):
                total_size[i] = total_size[i] * get_data_width(element.dtype)

            addr_type_list = []
            for tensor_w in workspace_list:
                if sch[tensor_w].scope == scope_cbuf_fusion:
                    addr_type_list.append(1)
                else:
                    addr_type_list.append(0)

            wkspace_dict = {"workspace": {"num": num,
                                          "size": total_size,
                                          "type": addr_type_list}}

            kernel_meta_dir = get_current_build_config("kernel_meta_parent_dir") + "/kernel_meta/"
            write_code(wkspace_dict,
                       os.path.join(kernel_meta_dir, "%s.json" % kernel_name))

    def _gen_build_map():
        fusion_config_map = config_map.get("fusion_build_config", {})
        build_map = {}
        for attr, value in itertools.chain(config_map.items(),
                                           fusion_config_map.items()):
            if tvm.build_module.BuildConfig.is_build_config(attr):
                build_map[attr] = value
        if te_util.L1CommonParam.l1_fusion_tensors_map is not None:
            build_map["dummy_placeholder"] = True
        if hasattr(sch, "tbe_compile_para") and sch.tbe_compile_para is not None:
            para_list = ("read_write_bank_conflict", "out_of_order", "pipeline_opt")
            for para in para_list:
                if para in sch.tbe_compile_para:
                    build_map[para] = sch.tbe_compile_para[para]

        return build_map

    def get_l1_tensors(input_tensors, l1_tensors_map):
        """
        get L1 tensors order
        """
        if l1_tensors_map is None:
            return []

        l1_tensors_list = []
        for ten_i in input_tensors:
            if ten_i in l1_tensors_map:
                l1_tensors_list.append(l1_tensors_map.get(ten_i, tvm.var("dummy")))

        return l1_tensors_list

    def _build(sch, tensor_list, name="cce_op", ):
        """
        do TVM build

        Parameters
        ----------

        Returns
        -------
        No return
        """
        device = "cce"

        with buildcfg.build_config(**build_map):
            tvm.build(sch, tensor_list, device, name=name)

    if config_map is None:
        config_map = {}
    elif "name" in config_map:
        check_kernel_name(config_map["name"])

    # for RL tune getting tensor_list
    bank_manager.set_tensor_list(config_map.get("tensor_list", []))

    config_map.setdefault("l1_fusion_option",
                          get_L1_info("L1_fusion_enabled"))
    config_map.setdefault("l2_fusion_option",
                          get_L1_info("L2_fusion_enabled"))
    build_map = _gen_build_map()
    local_config_map = {"print_ir": False,
                        "need_build": True,
                        "name": "cce_op",
                        "tensor_list": None}

    for key in local_config_map:
        key_exist_flag = (config_map.get(key) is not None) and \
                         (isinstance(config_map[key], type(local_config_map.get(key))) or
                          (local_config_map.get(key) is None))
        if key_exist_flag:
            local_config_map[key] = config_map.get(key)

    config_tensor_list = local_config_map.get("tensor_list")

    if config_tensor_list is None:
        dict_args = {"errCode": "E90001",
                     "detailed_cause": "Please infer correct parameter of tensor list through the key of 'tensor_list'"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    real_out_tensors = sch.cce_special["real_out_tensor"]
    orign_out_tensors = sch.cce_special["orign_out_tensor"]

    # update the config_tensor_list:update 1 auto_cast tensor 2 compute
    # group tensor
    config_tensor_list_tmp = []
    for tensor in config_tensor_list:  # 'pylint: disable=not-an-iterable
        if tensor not in orign_out_tensors:
            config_tensor_list_tmp.append(tensor)
        else:
            index = orign_out_tensors.index(tensor)
            config_tensor_list_tmp.append(real_out_tensors[index])

    # update special_tensor_list:if the spec node is a output, no need
    # to use worlspace
    special_tensor_list = []
    for tensor in sch.cce_special["tensor_list"]:
        if tensor not in config_tensor_list_tmp:
            special_tensor_list.append(tensor)

    tensor_list = config_tensor_list_tmp + special_tensor_list

    tensor_list = check_quantfuse_doubleout(tensor_list, sch)
    ConvParam.conv_deq_req_double_out = False

    l1_fusion_tensors = get_l1_tensors(tensor_list, te_util.L1CommonParam.l1_fusion_tensors_map)
    te_util.L1CommonParam.l1_fusion_tensors_map = None

    tensor_list = tensor_list + l1_fusion_tensors
    _build(sch, tensor_list, local_config_map.get("name"))

    with buildcfg.build_config(**build_map):
        if not get_current_build_config("enable_op_prebuild"):
            _write_workspace_info(special_tensor_list, local_config_map.get("name"))

    cce_emitinsn_params.cceEmitParamsIns.clear_param()


class ScheduleDispatch:
    """Schedule Dispatch"""
    @generic_dispatch(key=1)
    def handle_case(self, case):
        """handle case"""
        dict_args = {"errCode": "E90003", "detailed_cause": f"Unknown key {key} in generic_dispatch"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    @handle_case.register('segment')
    def _(self, case, templet_name, tensor, scope, spec_node_list, sch_list):
        if templet_name == 'global':
            cce_op = CceSegmentOp(scope, need_tensorize=True, need_pragma=True)
        if templet_name == 'speel':
            cce_op = CceSegmentSpeelOp(scope, need_tensorize=True, need_pragma=True)
        return cce_op.schedule(tensor, spec_node_list, sch_list)

    @handle_case.register('inplace')
    def _(self, case, tensor, scope, spec_node_list, sch_list):
        cce_op = CceInplaceOp(scope)
        return cce_op.schedule(tensor, spec_node_list, sch_list)

    @handle_case.register('concat')
    def _(self, case, tensor, scope, spec_node_list, sch_list):
        cce_op = CceConcatOp(scope, need_tensorize=True, need_pragma=True)
        return cce_op.schedule(tensor, spec_node_list, sch_list)

    @handle_case.register('gemm')
    def _(self, case, tensor, sch_list):
        return gemm_schedule(tensor, sch_list)

    @handle_case.register('matmul')
    def _(self, case, tensor, sch_list):
        return mmad_schedule(tensor, sch_list)

    @handle_case.register('convolution')
    def _(self, case, tensor, scope, spec_node_list, sch_list):
        conv2d_op = CceConvOp()
        return conv2d_op.schedule(tensor, spec_node_list, sch_list)

    @handle_case.register('conv2d_backprop_input')
    def _(self, case, tensor, spec_node_list, sch_list):
        cce_op = CceConv2dBackpropInputOp(scope_ubuf, need_tensorize=True, need_pragma=True)
        return cce_op.schedule(tensor, spec_node_list, sch_list)

    @handle_case.register('conv2d_backprop_filter')
    def _(self, case, tensor, spec_node_list, sch_list):
        cce_op = CceConv2dBackpropInputOp(scope_ubuf, need_tensorize=True, need_pragma=True)
        return cce_op.schedule(tensor, spec_node_list, sch_list)

    @handle_case.register('pooling2d')
    def _(self, case, tensor, sch_list):
        return pooling2d_schedule(tensor, sch_list)

    @handle_case.register('pooling3d')
    def _(self, case, tensor, sch_list):
        return pooling3d_schedule(tensor, sch_list)

    @handle_case.register('dilation')
    def _(self, case, tensor, sch_list):
        return dilation_schedule(tensor, sch_list)
