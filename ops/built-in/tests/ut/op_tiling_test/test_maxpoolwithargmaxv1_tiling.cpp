#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class MaxPoolWithArgmaxV1Tiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "MaxPoolWithArgmaxV1Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaxPoolWithArgmaxV1Tiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }
    return result;
}

TEST_F(MaxPoolWithArgmaxV1Tiling, maxpool_with_argmax_v1_tiling_0) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{2, 4, 42, 42, 16};
    std::vector<int64_t> output0{2, 4, 41, 41, 16};
    std::vector<int64_t> output1{2, 4, 4, 107, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
	opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 8 1 1 2 4 42 42 1764 8 21 21 441 28 441 7 23 22 2 0 565 564 1 ");
}

TEST_F(MaxPoolWithArgmaxV1Tiling, maxpool_with_argmax_v1_tiling_1) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{2, 4, 32, 32, 16};
    std::vector<int64_t> output0{2, 4, 31, 31, 16};
    std::vector<int64_t> output1{2, 4, 4, 62, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
	opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 8 1 1 2 4 32 32 1024 8 16 16 256 16 256 0 29 28 2 1 565 564 1 ");
}

TEST_F(MaxPoolWithArgmaxV1Tiling, maxpool_with_argmax_v1_tiling_2) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952, \"l1_size\": 1048576, \"kernel_h\": 2,\"kernel_w\": 3, \"stride_h\": 3, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 1}}";

    std::vector<int64_t> input0{2, 4, 42, 42, 16};
    std::vector<int64_t> output0{2, 4, 41, 41, 16};
    std::vector<int64_t> output1{2, 4, 4, 107, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
	opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 8 1 1 2 4 42 42 1764 8 22 15 330 21 330 6 42 42 2 0 719 718 1 ");
}