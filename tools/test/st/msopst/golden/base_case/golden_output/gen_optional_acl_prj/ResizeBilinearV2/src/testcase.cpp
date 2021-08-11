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


OP_TEST(ResizeBilinearV2, Test_ResizeBilinearV2_001_case_001)
{
    
    std::string opType = "ResizeBilinearV2";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{4, 16, 16, 16}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_INT32};
    opTestDesc.inputFormat = {(aclFormat)1, (aclFormat)1};
    opTestDesc.inputFilePath = {"test_data/data/Test_ResizeBilinearV2_001_case_001_input_0", "test_data/data/Test_ResizeBilinearV2_001_case_001_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{4, 48, 48, 16}};
    opTestDesc.outputDataType = {ACL_FLOAT};
    opTestDesc.outputFormat = {(aclFormat)1};
    opTestDesc.outputFilePath = {"result_files/Test_ResizeBilinearV2_001_case_001_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_BOOL, "align_corners"};
    attr0.boolAttr = 0;
    opTestDesc.opAttrVec.push_back(attr0);
    OpTestAttr attr1 = {OP_BOOL, "half_pixel_centers"};
    attr1.boolAttr = 0;
    opTestDesc.opAttrVec.push_back(attr1);

    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_ResizeBilinearV2_001_case_001");

}

