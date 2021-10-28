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


OP_TEST(Add, Test_Add_001_fuzz_case_001)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_001_input_0", "test_data/data/Test_Add_001_fuzz_case_001_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_001_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_001");

}


OP_TEST(Add, Test_Add_001_fuzz_case_002)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_002_input_0", "test_data/data/Test_Add_001_fuzz_case_002_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_002_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_002");

}


OP_TEST(Add, Test_Add_001_fuzz_case_003)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_003_input_0", "test_data/data/Test_Add_001_fuzz_case_003_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_003_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_003");

}


OP_TEST(Add, Test_Add_001_fuzz_case_004)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_004_input_0", "test_data/data/Test_Add_001_fuzz_case_004_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_004_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_004");

}


OP_TEST(Add, Test_Add_001_fuzz_case_005)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_005_input_0", "test_data/data/Test_Add_001_fuzz_case_005_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_005_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_005");

}


OP_TEST(Add, Test_Add_001_fuzz_case_006)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_006_input_0", "test_data/data/Test_Add_001_fuzz_case_006_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_006_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_006");

}


OP_TEST(Add, Test_Add_001_fuzz_case_007)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_007_input_0", "test_data/data/Test_Add_001_fuzz_case_007_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_007_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_007");

}


OP_TEST(Add, Test_Add_001_fuzz_case_008)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_008_input_0", "test_data/data/Test_Add_001_fuzz_case_008_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_008_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_008");

}


OP_TEST(Add, Test_Add_001_fuzz_case_009)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_009_input_0", "test_data/data/Test_Add_001_fuzz_case_009_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_009_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_009");

}


OP_TEST(Add, Test_Add_001_fuzz_case_010)
{
    
    std::string opType = "Add";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 2}, {2}};
    opTestDesc.inputDataType = {ACL_FLOAT16, ACL_FLOAT16};
    opTestDesc.inputFormat = {(aclFormat)2, (aclFormat)2};
    opTestDesc.inputFilePath = {"test_data/data/Test_Add_001_fuzz_case_010_input_0", "test_data/data/Test_Add_001_fuzz_case_010_input_1"};
    opTestDesc.inputConst = {false, true};
    // output parameter init
    opTestDesc.outputShape = {{1, 2}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {(aclFormat)2};
    opTestDesc.outputFilePath = {"result_files/Test_Add_001_fuzz_case_010_output_0"};
    // attr parameter init
    
    // set deviceId
    const uint32_t deviceId = 0;
    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc, deviceId), opTestDesc, "Test_Add_001_fuzz_case_010");

}

