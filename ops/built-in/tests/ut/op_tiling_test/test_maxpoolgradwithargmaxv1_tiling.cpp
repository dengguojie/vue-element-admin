#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class MaxPoolGradWithArgmaxV1Tiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "MaxPoolGradWithArgmaxV1Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaxPoolGradWithArgmaxV1Tiling TearDown" << std::endl;
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

TEST_F(MaxPoolGradWithArgmaxV1Tiling, maxpoolgrad_with_argmax_v1_tiling_0) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0, \"dtype_size\": 2}}";

    std::vector<int64_t> input0{1, 4, 112, 560, 16};
    std::vector<int64_t> input1{1, 4, 113, 561, 16};
    std::vector<int64_t> input2{1, 4, 4, 3964, 16};
	std::vector<int64_t> output0{1, 4, 112, 560, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
	TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
	TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "1234561";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "4 32 1 0 112 560 113 561 2 2 1 1 112 560 1 1 0 385 5 5 0 192 2 2 15 15 8 ");
}

TEST_F(MaxPoolGradWithArgmaxV1Tiling, maxpoolgrad_with_argmax_v1_tiling_1) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0, \"dtype_size\": 2}}";

    std::vector<int64_t> input0{1, 4, 112, 560, 16};
    std::vector<int64_t> input1{1, 4, 56, 280, 16};
    std::vector<int64_t> input2{1, 4, 9, 981, 16};
	std::vector<int64_t> output0{1, 4, 112, 560, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
	TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
	TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "1234563";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "3 32 1 0 112 560 56 280 2 2 1 1 112 560 1 1 3 577 0 0 1 288 0 0 7 14 8 ");
}

TEST_F(MaxPoolGradWithArgmaxV1Tiling, maxpoolgrad_with_argmax_v1_tiling_2) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0, \"dtype_size\": 2}}";

    std::vector<int64_t> input0{2, 16, 112, 560, 16};
    std::vector<int64_t> input1{2, 16, 56, 280, 16};
    std::vector<int64_t> input2{2, 16, 9, 981, 16};
	std::vector<int64_t> output0{2, 16, 112, 560, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
	TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
	TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "1234563";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 32 1 0 112 560 56 280 2 2 1 1 112 560 1 1 3 577 0 0 0 0 0 0 56 56 1 ");
}

TEST_F(MaxPoolGradWithArgmaxV1Tiling, maxpoolgrad_with_argmax_v1_tiling_3) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0, \"dtype_size\": 2}}";

    std::vector<int64_t> input0{2, 16, 112, 112, 16};
    std::vector<int64_t> input1{2, 16, 56, 56, 16};
    std::vector<int64_t> input2{2, 16, 9, 197, 16};
	std::vector<int64_t> output0{2, 16, 112, 112, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
	TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
	TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "1234563";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 32 1 0 112 112 56 56 2 2 1 1 112 112 1 1 15 129 0 0 7 64 0 0 56 56 1 ");
}

TEST_F(MaxPoolGradWithArgmaxV1Tiling, maxpoolgrad_with_argmax_v1_tiling_4) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0, \"dtype_size\": 2}}";

    std::vector<int64_t> input0{1, 16, 42, 42, 16};
    std::vector<int64_t> input1{1, 16, 21, 21, 16};
    std::vector<int64_t> input2{1, 16, 9, 29, 16};
	std::vector<int64_t> output0{1, 16, 42, 42, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
	TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
	TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "1234563";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 16 1 0 42 42 21 21 2 2 1 1 42 42 1 1 31 65 0 0 15 32 0 0 11 11 2 ");
}

TEST_F(MaxPoolGradWithArgmaxV1Tiling, maxpoolgrad_with_argmax_v1_tiling_5) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo2 = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144, \"l1_size\": 1048576, \"kernel_h\": 2,\"kernel_w\": 2, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0, \"dtype_size\": 2}}";

    std::vector<int64_t> input0{2, 4, 112, 112, 16};
    std::vector<int64_t> input1{2, 4, 57, 57, 16};
    std::vector<int64_t> input2{2, 4, 4, 205, 16};
	std::vector<int64_t> output0{2, 4, 112, 112, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
	TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
	TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo2;
    op_compile_info.key = "1234564";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "5 32 1 0 112 112 57 57 2 2 1 1 112 112 0 0 0 128 14 14 0 64 7 7 14 15 4 ");
}

TEST_F(MaxPoolGradWithArgmaxV1Tiling, maxpoolgrad_with_argmax_v1_tiling_6) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 220000, \"l1_size\": 1048576, \"kernel_h\": 3,\"kernel_w\": 3, \"stride_h\": 2, \"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0, \"dtype_size\": 2}}";

    std::vector<int64_t> input0{2, 16, 112, 560, 16};
    std::vector<int64_t> input1{2, 16, 56, 280, 16};
    std::vector<int64_t> input2{2, 16, 9, 981, 16};
	std::vector<int64_t> output0{2, 16, 112, 560, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
	TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
	TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "1234563";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 32 1 0 112 560 56 280 2 2 1 1 112 560 1 1 3 577 0 0 0 0 0 0 56 56 1 ");
}