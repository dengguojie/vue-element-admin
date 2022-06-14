/**
* @file testcase.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "op_test.h"
#include "op_execute.h"
using namespace OpTest;


OP_TEST(Adds, Test_Adds_001_case_001_ND_float16)
{
    
    std::string opType = "Adds";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{4}, {1}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Adds_001_case_001_ND_float16_input_0", "test_data/data/Test_Adds_001_case_001_ND_float16_input_1"};
    opTestDesc.inputConst = {false, false};
    // output parameter init
    opTestDesc.outputShape = {{4}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Adds_001_case_001_ND_float16_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Adds_001_case_001_ND_float16");

}

