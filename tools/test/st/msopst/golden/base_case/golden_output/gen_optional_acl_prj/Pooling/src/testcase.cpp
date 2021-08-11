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


OP_TEST(Pooling, Test_Pooling_001_case_001)
{
    
    std::string opType = "Pooling";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2, 64, 64, 16}, {}, {}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED};
    opTestDesc.inputFormat = {(aclFormat)3, (aclFormat)-1, (aclFormat)-1};
    opTestDesc.inputFilePath = {"test_data/data/Test_Pooling_001_case_001_input_0", "", ""};
    opTestDesc.inputConst = {false, false, false};
    // output parameter init
    opTestDesc.outputShape = {{1, 2, 64, 64, 16}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)3};
    opTestDesc.outputFilePath = {"result_files/Test_Pooling_001_case_001_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_LIST_INT, "window"};
    attr0.listIntAttr = {3, 3};
    opTestDesc.opAttrVec.push_back(attr0);
    OpTestAttr attr1 = {OP_LIST_INT, "stride"};
    attr1.listIntAttr = {2, 2};
    opTestDesc.opAttrVec.push_back(attr1);
    OpTestAttr attr2 = {OP_INT, "mode"};
    attr2.intAttr = 1;
    opTestDesc.opAttrVec.push_back(attr2);
    OpTestAttr attr3 = {OP_INT, "offset_x"};
    attr3.intAttr = 0;
    opTestDesc.opAttrVec.push_back(attr3);
    OpTestAttr attr4 = {OP_LIST_INT, "pad"};
    attr4.listIntAttr = {0, 0, 0, 0};
    opTestDesc.opAttrVec.push_back(attr4);
    OpTestAttr attr5 = {OP_BOOL, "global_pooling"};
    attr5.boolAttr = 0;
    opTestDesc.opAttrVec.push_back(attr5);
    OpTestAttr attr6 = {OP_INT, "ceil_mode"};
    attr6.intAttr = 0;
    opTestDesc.opAttrVec.push_back(attr6);
    OpTestAttr attr7 = {OP_LIST_INT, "dilation"};
    attr7.listIntAttr = {1, 1, 1, 1};
    opTestDesc.opAttrVec.push_back(attr7);

    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Pooling_001_case_001");

}

