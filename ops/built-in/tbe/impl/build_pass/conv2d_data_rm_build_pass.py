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

def head_struct_check(op_list):
    """
    check the head of op_list Whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op

    Returns
    -------
    True or False
    """
    for op_node in op_list:
        if op_node["type"] == "Data" or op_node["type"] == "Conv2D" or\
           op_node["type"] == "StridedRead":
            if op_node["type"] == "Conv2D":
                return True
            if op_node["type"] == "StridedRead":
                consistent_flag = False
                for op in op_list:
                    if op["type"] == "Conv2D":
                        for input_desc in op["input_desc"]:
                            if input_desc["name"] == op_node["output_desc"][0]["name"]:
                                consistent_flag = True
                if consistent_flag:
                    return True
                else:
                    return False
        else:
            return False
    return False

def get_pre_op(op_list, current_op):
    """
    get the precursor node based on the current node

    Parameters
    ----------
    op_list:list
    op info of fusion op
    current_op:dict
    one op in op_list

    Returns
    -------
    pre_op of current op
    """
    for input_desc in current_op["input_desc"]:
        for op_node in op_list:
            for output_desc in op_node["output_desc"]:
                if output_desc["name"] == input_desc["name"] and\
                    op_node["type"] != "Data":
                    return op_node
    return []

def middle_struct_check(op_list, last_op_output_desc_list):
    """
    check the middle of op_list Whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    last_op_output_desc_list:list
    output_desc of last op in op_list

    Returns
    -------
    True or False
    """
    rm_middle_struct_oplist = ["dequant", "quant", "ElemWise"]
    rm_double_output_oplist = ["dequant", "ElemWise"]
    last_op_list = []
    for last_op_output_desc in last_op_output_desc_list:
        current_op = []
        for op in op_list:
            for output_desc in op["output_desc"]:
                if output_desc == last_op_output_desc:
                    current_op = op
        if not current_op:
            return False
        else:
            last_op_list.append(copy.deepcopy(current_op))
        while 1:
            pre_op = get_pre_op(op_list, current_op)
            if not pre_op:
                return False
            if pre_op["type"] == "Conv2D":
                break
            if pre_op["pattern"] in rm_middle_struct_oplist:
                if pre_op["pattern"] == "ElemWise" and len(pre_op["input_desc"]) > 1:
                    return False
            else:
                return False
            current_op = pre_op
    if len(last_op_list) == 2:
        if (last_op_list[0]["pattern"] != "quant" and last_op_list[1]["pattern"] != "quant") or\
           (last_op_list[0]["pattern"] not in rm_double_output_oplist and\
            last_op_list[1]["pattern"] not in rm_double_output_oplist):
            return False
    return True

def tail_struct_check(op_list, last_op_output_desc_list):
    """
    check the tail of op_list Whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    last_op_output_desc_list:list
    output_desc of last op in op_list

    Returns
    -------
    True or False
    """
    rm_tail_struct_oplist = ["quant", "ElemWise", "strided_write"]
    rm_double_output_oplist = ["dequant", "ElemWise"]
    for op_node in op_list:
        if "pattern" in op_node and op_node["pattern"] == "ElemWise" and\
           len(op_node["input_desc"]) > 1:
            return False
        last_op_flag = True
        if len(op_node["output_desc"]) == 1:
            for op in op_list:
                if "input_desc" not in op:
                    continue
                else:
                    for input_desc in op["input_desc"]:
                        if op_node["output_desc"][0]["name"] == input_desc["name"]:
                            last_op_flag = False
                            break
            if last_op_flag:
                if op_node["pattern"] not in rm_tail_struct_oplist:
                    return False
                last_op_output_desc_list.append(copy.deepcopy(op_node["output_desc"][0]))
        else:
            if op_node["pattern"] not in rm_double_output_oplist:
                return False
            for output_desc in op_node["output_desc"]:
                for op in op_list:
                    if "input_desc" not in op:
                        continue
                    else:
                        for input_desc in op["input_desc"]:
                            if output_desc["name"] == input_desc["name"]:
                                last_op_flag = False
                                break
                if last_op_flag:
                    last_op_output_desc_list.append(copy.deepcopy(output_desc))
                last_op_flag = True
    if len(last_op_output_desc_list) > 2 or len(last_op_output_desc_list) == 0:
        return False
    return True

def can_user_define_compute(op_list, last_op_output_desc_list):
    """
    check the op_list Whether it meets the conditions

    Parameters
    ----------
    op_list:list
    op info of fusion op
    last_op_output_desc_list:list
    output_desc of last op in op_list

    Returns
    -------
    True or False
    """
    for op in op_list:
        if "output_desc" in op:
            for output_desc in op["output_desc"]:
                if "L1_fusion_type" in output_desc and output_desc["L1_fusion_type"] != -1:
                    return False
    if head_struct_check(op_list):
        rm_invalid_op_list = ["ElemWise", "strided_write", "strided_read"]
        invalid_flag = True
        for op_node in op_list:
            if op_node["type"] != "Data":
                if op_node["type"] != "Conv2D" and\
                  "pattern" in op_node and op_node["pattern"] not in rm_invalid_op_list:
                    invalid_flag = False
        if invalid_flag:
            return False
        if tail_struct_check(op_list, last_op_output_desc_list):
            if middle_struct_check(op_list, last_op_output_desc_list):
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
    Whether user define compute

    Returns
    -------
    None
    """
    for op_node in op_list:
        if op_node["type"] == "Conv2D":
            op_node["options"] = {"invalid_data_rm": rm_flag}
            return

def add_rm_op_in_op_list(op_list, last_op_output_desc_list):
    """
    add remove op in oplist

    Parameters
    ----------
    op_list:list
    op info of fusion op
    last_op_output_desc_list:list
    output_desc of last op in op_list

    Returns
    -------
    None
    """
    for last_op_output_desc in last_op_output_desc_list:
        last_op = []
        name = "conv2d_data_rm_" + str(time.time())
        for op in op_list:
            for output_desc in op["output_desc"]:
                if output_desc == last_op_output_desc:
                    output_desc["name"] = name + "__0"
                    last_op = op
        if not last_op:
            return
        rm_op = copy.deepcopy(last_op)
        del rm_op["input_desc"]
        del rm_op["output_desc"]
        rm_op["input_desc"] = [copy.deepcopy(last_op_output_desc)]
        rm_op["output_desc"] = [copy.deepcopy(last_op_output_desc)]
        rm_op["func_name"] = "conv2d_data_rm"
        rm_op["id"] = rm_op["id"] * 2
        del rm_op["input_desc"][0]["output_index"]
        rm_op["output_desc"][0]["output_index"] = 0
        rm_op["module_name"] = "impl.conv2d_data_rm"
        rm_op["name"] = name
        if not rm_op["ori_name"]:
            rm_op["ori_name"] = ["ori_" + name]
        else:
            rm_op["ori_name"][0] = "ori_" + name
        rm_op["input_desc"][0]["name"] = name + "__0"
        rm_op["pattern"] = "Opaque"
        rm_op["prebuild_outs_attrs"]["kwds_args"] = {}
        rm_op["prebuild_outs_attrs"]["list_args"] = []
        rm_op["type"] = "conv2d_data_rm"
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
    last_op_output_desc_list = []
    res = can_user_define_compute(op_list, last_op_output_desc_list)
    if res:
        set_rm_in_options(op_list, True)
        add_rm_op_in_op_list(op_list, last_op_output_desc_list)
    else:
        set_rm_in_options(op_list, False)
