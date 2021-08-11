#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves op file content.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
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
    opTestDesc.inputConst = {{{is_const}}};
    // output parameter init
    opTestDesc.outputShape = {{{output_shape_data}}};
    opTestDesc.outputDataType = {{{output_data_type}}};
    opTestDesc.outputFormat = {{{output_format}}};
    opTestDesc.outputFilePath = {{{output_file_path}}};
    // attr parameter init
    {all_attr_code_snippet}
    // set deviceId
    const uint32_t deviceId = {device_id};
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "{testcase_name}");
"""

# -----mindspore test .py file--------------------------
PYTEST_INI_CONTEN = """
[pytest]
log_cli = 1
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)
log_cli_date_format=%Y-%m-%d %H:%M:%S
"""
TESTCASE_IMPORT_CONTENT = """import numpy as np
import pytest
import time
import logging
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor

# Import the definition of the {op_name} primtive.
from {import_op} import {op_name}
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id={device_id})
logger = logging.getLogger(__name__)

"""

TESTCASE_CLASS_CONTENT_NO_ATTR = """
class Net(nn.Cell):
    \"""Net definition\"""

    def __init__(self):
        super(Net, self).__init__()
        self.{op_lower} = {op_name}()

    def construct(self,{input_args}):
        return self.{op_lower}({inputs})
"""

TESTCASE_CLASS_CONTENT_WITH_ATTR_CONSTRUCT = """
    def construct(self, {inputs}):
        return self.{op_lower}({inputs})
"""

TESTCASE_CLASS_CONTENT_WITH_ATTR = """
class Net(nn.Cell):
    \"""Net definition\"""

    def __init__(self):
        super(Net, self).__init__()
        self.{op_lower} = {op_name}({attr_value})
    {attr_constrct}
"""
TESTCASE_TEST_NET_INPUT = """
    {input_name} = np.fromfile('{file}', np.{np_type})
    {input_name}.shape = {op_shape}
"""

TESTCASE_TEST_TENSOR = """Tensor({input_name})"""
TESTCASE_TEST_NET_OUTPUT = """{output_name} = {op_lower}_test({tensor})
"""

TESTCASE_TEST_NET = """
def {subcase}():
    {inputs}
    {op_lower}_test = Net()
    
    start = time.time()
    
    {outputs}
    end = time.time()
    
    print("running time: %.2f s" %(end-start))
"""