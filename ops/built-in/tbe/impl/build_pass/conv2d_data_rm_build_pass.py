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
conv2d_data_rm_build_pass
"""
import copy
import time
import te.platform as tbe_platform

CONV = "Conv2D"
DATA = "Data"
STRIDED_READ = "StrideRead"
DATA_RM = "conv2d_data_rm"

DEQUANT_PATTERN = "dequant"
QUANT_PATTERN = "quant"
ELTWISE_PATTERN = "ElemWise"
STRIDED_WRITE_PATTERN = "strided_write"
STRIDED_READ_PATTERN = "strided_read"

INPUT_DESC = "input_desc"
OUTPUT_DESC = "output_desc"
PATTERN_STR = "pattern"
TYPE_STR = "type"
NAME_STR = "name"
L1_FUSION_TYPE = "L1_fusion_type"
UNDERLINE = "_"
NAME_SUFFIX = "__0"
NAME_PREFIX = "ori_"
FUNC_NAME = "func_name"
INDEX = "id"
OUTPUT_INDEX = "output_index"
ORIGINAL_NAME = "ori_name"
INVALID_PATTERN = "Opaque"
PREBUILD_ATTRS = "prebuild_outs_attrs"
KWDS_ARGS = "kwds_args"
LIST_ARGS = "list_args"
MODULE_NAME = "module_name"
OP_MODULE = "impl.conv2d_data_rm"
OP_LIST_KEY_OPTIONS = "options"
PARAM_KEY_DATA_RM = "invalid_data_rm"

L1_FUSION_DISABLE = -1
ELTWISE_INPUT_SIZE_MAX = 1
OUTPUT_DESC_SIZE_MAX = 2
HEAD_STRUCK_OP_SIZE_ONE = 1
HEAD_STRUCK_OP_SIZE_TWO = 2
TAIL_OP_SIZE = 1
DOUBLE_OUTPUT = 2
SINGLE_OUTPUT = 1

def get_next_op(op_list, current_op):
    """
    get next node based on the current node

    Parameters
    ----------
    op_list:list
    op info of fusion op
    current_op:dict
    one op in op_list

    Returns
    -------
    next op of current op
    """
    for output_desc in current_op[OUTPUT_DESC]:
        for op_node in op_list:
            if op_node[TYPE_STR] == DATA:
                continue
            for input_desc in op_node[INPUT_DESC]:
                if output_desc[NAME_STR] == input_desc[NAME_STR]:
                    return op_node
    return []

def get_pre_op(op_list, current_op):
    """
    get pre node based on the current node

    Parameters
    ----------
    op_list:list
    op info of fusion op
    current_op:dict
    one op in op_list

    Returns
    -------
    pre op of current op
    """
    for input_desc in current_op[INPUT_DESC]:
        for op_node in op_list:
            for output_desc in op_node[OUTPUT_DESC]:
                if output_desc[NAME_STR] == input_desc[NAME_STR] and\
                    op_node[TYPE_STR] != DATA:
                    return op_node
    return []

def total_struct_check(op_list, head_op, tail_op, head_struck_oplist, tail_op_list):
    """
    check struct of op_list whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct
    head_struck_oplist:list
    head struct ops in op_list
    tail_op_list:list
    all last ops in op_list

    Returns
    -------
    True or False
    """
    double_output_oplist = (DEQUANT_PATTERN, ELTWISE_PATTERN)
    tail_oplist = (QUANT_PATTERN, ELTWISE_PATTERN, STRIDED_WRITE_PATTERN)
    head_oplist = (CONV, STRIDED_READ)
    strided_read_flag = False
    strided_write_flag = False
    for op in op_list:
        if OUTPUT_DESC in op:
            for output_desc in op[OUTPUT_DESC]:
                if L1_FUSION_TYPE in output_desc and output_desc[L1_FUSION_TYPE] != L1_FUSION_DISABLE:
                    return False
                if op[TYPE_STR] == DATA:
                    continue
                if op[PATTERN_STR] == ELTWISE_PATTERN and len(op[INPUT_DESC]) > ELTWISE_INPUT_SIZE_MAX:
                    return False
                if len(op[OUTPUT_DESC]) == OUTPUT_DESC_SIZE_MAX:
                    if op[PATTERN_STR] not in double_output_oplist or\
                        strided_read_flag ==True:
                        return False
                    tail_op_list.append(op)
                    continue
                next_op = get_next_op(op_list, op)
                if op[TYPE_STR] == STRIDED_READ:
                    strided_read_flag = True
                    if next_op and next_op[TYPE_STR] == CONV:
                        head_op.update(op)
                        head_struck_oplist.append(op)
                    else:
                        return False
                if op[TYPE_STR] == CONV:
                    head_op.update(op)
                    head_struck_oplist.append(op)
                if not next_op:
                    if op[PATTERN_STR] not in tail_oplist:
                        return False
                    if op[PATTERN_STR] == STRIDED_WRITE_PATTERN:
                        strided_write_flag == True
                    tail_op_list.append(op)
                    tail_op.update(op)
    if strided_write_flag and len(tail_op_list) == DOUBLE_OUTPUT:
        return False
    if len(head_struck_oplist) == HEAD_STRUCK_OP_SIZE_ONE and\
        head_struck_oplist[0][TYPE_STR] == CONV:
        return True
    if len(head_struck_oplist) == HEAD_STRUCK_OP_SIZE_TWO and\
        head_struck_oplist[0][TYPE_STR] in head_oplist and\
        head_struck_oplist[1][TYPE_STR] in head_oplist:
        return True
    return False

def pattern1_match(op_list, head_op, tail_op):
    """
    check pattern: (stridedread) + conv + dequant + eltwise*N + (stridedwrite)

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct

    Returns
    -------
    True or False
    """
    if head_op[PATTERN_STR] != DEQUANT_PATTERN or tail_op[PATTERN_STR] != ELTWISE_PATTERN:
        return False
    while 1:
        next_op = get_next_op(op_list, head_op)
        if not next_op or next_op[PATTERN_STR] != ELTWISE_PATTERN:
            return False
        if next_op[NAME_STR] == tail_op[NAME_STR]:
            return True
        head_op = next_olltllyp

def pattern2_match(op_list, head_op, tail_op):
    """
    check pattern: (stridedread) + conv + (dequant) + (eltwise*N) + quant + (stridedwrite)

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct

    Returns
    -------
    True or False
    """
    if tail_op[PATTERN_STR] != QUANT_PATTERN:
        return False
    while 1:
        next_op = get_next_op(op_list, head_op)
        if not next_op or (next_op[PATTERN_STR] != ELTWISE_PATTERN and next_op[PATTERN_STR] != QUANT_PATTERN):
            return False
        if next_op[NAME_STR] == tail_op[NAME_STR]:
            return True
        head_op = next_op

def pattern3_match(op_list, head_op, tail_op, head_struct_oplist, tail_op_list):
    """
    check pattern: conv + dequant + (eltwise*N) + quant
    double output: one output from quant, the other from dequant or eltwise

    Parameters
    ----------
    op_list:list
    op info of fusion op
    head_op:dict
    last op of common head_struct
    tail_op:dict
    first op of common tail_struct
    head_struck_oplist:list
    head struct ops in op_list
    tail_op_list:list
    all last ops in op_list

    Returns
    -------
    True or False
    """
    if head_op[PATTERN_STR] != DEQUANT_PATTERN or tail_op[PATTERN_STR] != QUANT_PATTERN:
        return False
    while 1:
        next_op = get_next_op(op_list, head_op)
        if not next_op or (next_op[PATTERN_STR] != ELTWISE_PATTERN and next_op[PATTERN_STR] != QUANT_PATTERN):
            return False
        if next_op[NAME_STR] == tail_op[NAME_STR]:
            return True
        head_op = tail_op

def can_user_define_compute(op_list, tail_op_list):
    """
    check the op_list whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    tail_op_list:list
    all last op in op_list

    Returns
    -------
    True or False
    """
    head_op = {}
    tail_op = {}
    head_struct_oplist = []
    if total_struct_check(op_list, head_op, tail_op, head_struct_oplist, tail_op_list):
        if head_op:
            next_op = get_next_op(op_list, head_op)
            if next_op[PATTERN_STR] == DEQUANT_PATTERN:
                head_op = next_op
        if tail_op and tail_op[PATTERN_STR] == STRIDED_WRITE_PATTERN:
            tail_op = get_pre_op(op_list, tail_op)

        if len(tail_op_list) == SINGLE_OUTPUT:
            if pattern1_match(op_list, head_op, tail_op):
                return True
            if pattern2_match(op_list, head_op, tail_op):
                return True
        if len(tail_op_list) == DOUBLE_OUTPUT:
            if pattern3_match(op_list, head_op, tail_op, head_struct_oplist, tail_op_list):
                return True
    return False

def set_rm_in_options(op_list, rm_flag=False):
    """
    set invalid_data_rm in options as True or False

    Parameters
    ----------
    op_list:list
    op info of fusion op
    rm_flag:bool
    whether user define compue

    Returns
    -------
    None
    """
    for op_node in op_list:
        if op_node[TYPE_STR] == CONV:
            op_node[OP_LIST_KEY_OPTIONS] = {PARAM_KEY_DATA_RM: rm_flag}
            return

def add_rm_op_in_op_list(op_list, tail_op_list):
    """
    add remove op in oplist

    Parameters
    ----------
    op_list:list
    op info of fusion op
    tail_op_list:list
    all last op in op_list

    Returns
    -------
    None
    """
    for last_op in tail_op_list:
        rm_op = copy.deepcopy(last_op)
        del rm_op[INPUT_DESC]
        del rm_op[OUTPUT_DESC]
        name = DATA_RM + UNDERLINE + str(time.time())
        if len(last_op[OUTPUT_DESC]) == DOUBLE_OUTPUT:
            last_op_desc_flag = True
            for output_desc in last_op[OUTPUT_DESC]:
                for op in op_list:
                    if INPUT_DESC not in op:
                        continue
                    else:
                        for input_desc in op[INPUT_DESC]:
                            if output_desc[NAME_STR] == input_desc[NAME_STR]:
                                last_op_desc_flag = False
                                break
                if last_op_desc_flag:
                    rm_op[INPUT_DESC] = [copy.deepcopy(last_op[OUTPUT_DESC])]
                    rm_op[OUTPUT_DESC] = [copy.deepcopy(last_op[OUTPUT_DESC])]
                    output_desc[NAME_STR] = name + NAME_SUFFIX
                last_op_desc_flag = True
        else:
            rm_op[INPUT_DESC] = copy.deepcopy(last_op[OUTPUT_DESC])
            rm_op[OUTPUT_DESC] = copy.deepcopy(last_op[OUTPUT_DESC])
            output_desc[NAME_STR] = name + NAME_SUFFIX
        rm_op[FUNC_NAME] = DATA_RM
        rm_op[INDEX] = rm_op[INDEX] * 2
        del rm_op[INPUT_DESC][0][OUTPUT_INDEX]
        rm_op[OUTPUT_DESC][0][OUTPUT_INDEX] = 0
        rm_op[MODULE_NAME] = OP_MODULE
        rm_op[NAME_STR] = name
        if not rm_op[ORIGINAL_NAME]:
            rm_op[ORIGINAL_NAME] = [NAME_PREFIX + name]
        else:
            rm_op[ORIGINAL_NAME][0] = NAME_PREFIX + name
        rm_op[INPUT_DESC][0][NAME_STR] = name + NAME_SUFFIX
        rm_op[PATTERN_STR] = INVALID_PATTERN
        if PREBUILD_ATTRS in rm_op:
            rm_op[PREBUILD_ATTRS][KWDS_ARGS] = {}
            rm_op[PREBUILD_ATTRS][LIST_ARGS] = []
        rm_op[TYPE_STR] = DATA_RM
        op_list.append(rm_op)

@tbe_platform.fusion_manager.fusion_manager.register_build_pass()
def conv2d_data_rm_build_pass(op_list):
    """
    user-defined compute if necessary

    Parameters
    ----------
    op_list:list
    op info of fusion op

    Returns
    -------
    None
    """
    tail_op_list = []
    res = can_user_define_compute(op_list, tail_op_list)
    if res:
        set_rm_in_options(op_list, True)
        add_rm_op_in_op_list(op_list, tail_op_list)
    else:
        set_rm_in_options(op_list, False)
