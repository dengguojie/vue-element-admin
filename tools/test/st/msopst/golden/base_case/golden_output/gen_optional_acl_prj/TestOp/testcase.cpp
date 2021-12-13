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


OP_TEST(TestOp, Test_Op_001_case_001)
{
    
    std::string opType = "TestOp";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1}};
    opTestDesc.inputDataType = {ACL_INT64};
    opTestDesc.inputFormat = {(aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Op_001_case_001_input_0"};
    opTestDesc.inputConst = {true};
    // output parameter init
    opTestDesc.outputShape = {{1}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Op_001_case_001_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_DTYPE, "dtype"};
    attr0.dtypeAttr = ACL_FLOAT16;
    opTestDesc.opAttrVec.push_back(attr0);
    OpTestAttr attr1 = {OP_INT, "seed"};
    attr1.intAttr = 0;
    opTestDesc.opAttrVec.push_back(attr1);
    OpTestAttr attr2 = {OP_INT, "seed2"};
    attr2.intAttr = 0;
    opTestDesc.opAttrVec.push_back(attr2);

    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Op_001_case_001");

}

