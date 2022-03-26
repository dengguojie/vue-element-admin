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

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 1, "
        "\"strides_w\": 1, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{4, 5, 35, 35, 16};
    std::vector<int64_t> output0{4, 5, 35, 35, 16};
    std::vector<int64_t> output1{4, 5, 1, 78, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v1_tiling_0";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 32 766 754 35 35 35 35 35 35 0 0 0 0 1 1 1 0 766 0 754 20 16 768 48 1248 ");
}

TEST_F(MaxPoolWithArgmaxV1Tiling, maxpool_with_argmax_v1_tiling_1) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 2, \"ksize_w\": 2, \"strides_h\": 2, "
        "\"strides_w\": 2, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{4, 5, 35, 35, 16};
    std::vector<int64_t> output0{4, 5, 17, 17, 16};
    std::vector<int64_t> output1{4, 5, 4, 20, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v1_tiling_1";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "4 20 1 1 35 35 17 17 34 34 0 0 0 0 1 15 1 1 2 1 2 20 16 16 32 320 ");
}

TEST_F(MaxPoolWithArgmaxV1Tiling, maxpool_with_argmax_v1_tiling_2) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV1";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 2, \"ksize_w\": 2, \"strides_h\": 2, "
        "\"strides_w\": 2, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{4, 5, 100, 100, 16};
    std::vector<int64_t> output0{4, 5, 50, 50, 16};
    std::vector<int64_t> output1{4, 5, 4, 158, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v1_tiling_2";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "4 20 1 1 100 100 50 50 100 100 0 0 0 0 1 5 1 10 0 10 0 20 16 0 64 2528 ");
}