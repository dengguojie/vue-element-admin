#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import collections


class OpInfo:
    """
    OpInfo store the op informat for generate the op files,
    parsed_input_info and parsed_output_info is dicts,eg:
    {name:
        {
        ir_type_list:[],
        param_type:""required,
        format_list:[]
        }
    }
    """
    def __init__(self):
        self.op_type = ""
        self.fix_op_type = ""
        self.parsed_input_info = collections.OrderedDict()
        self.parsed_output_info = collections.OrderedDict()
        self.parsed_attr_info = []

