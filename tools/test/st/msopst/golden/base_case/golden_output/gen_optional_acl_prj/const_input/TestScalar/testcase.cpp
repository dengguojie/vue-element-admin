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


OP_TEST(TestScalar, Test_Op_001_case_001)
{
    
    std::string opType = "TestScalar";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{}};
    opTestDesc.inputDataType = {ACL_INT32};
    opTestDesc.inputFormat = {(aclFormat)1};
    opTestDesc.inputFilePath = {"test_data/data/Test_Op_001_case_001_input_0"};
    opTestDesc.inputConst = {true};
    // output parameter init
    opTestDesc.outputShape = {{1}};
    opTestDesc.outputDataType = {ACL_FLOAT};
    opTestDesc.outputFormat = {(aclFormat)1};
    opTestDesc.outputFilePath = {"result_files/Test_Op_001_case_001_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Op_001_case_001");

}

