#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves op file content.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
# [TODO]
MAIN_CPP_CONTENT = """
/**
* @file main.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

int main()
{{
  {testcase_call_sentences}  
}}
"""

TESTCASE_CALL = """
    if (!{function_name}()) {{
        cout << "{function_name} execute failed!";
    }}
"""
# [TODO]

TESTCASE_FUNCTION = """
OP_TEST({op_name}, {testcase_name})
{{
    {testcase_content}
}}

"""

TESTCASE_CONTENT = """
    std::string opType = "{op_name}";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{{input_shape_data}}};
    opTestDesc.inputDataType = {{{input_data_type}}};
    opTestDesc.inputFormat = {{{input_format}}};
    opTestDesc.inputFilePath = {{{input_file_path}}};
    // output parameter init
    opTestDesc.outputShape = {{{output_shape_data}}};
    opTestDesc.outputDataType = {{{output_data_type}}};
    opTestDesc.outputFormat = {{{output_format}}};
    opTestDesc.outputFilePath = {{{output_file_path}}};
    // attr parameter init
    {all_attr_code_snippet}
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc), opTestDesc, "{testcase_name}");
"""
