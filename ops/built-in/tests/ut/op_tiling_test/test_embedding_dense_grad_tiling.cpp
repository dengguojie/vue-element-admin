#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class EmbeddingDenseGradTiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "EmbeddingDenseGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "EmbeddingDenseGradTiling TearDown" << std::endl;
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

TEST_F(EmbeddingDenseGradTiling, embedding_dense_grad_tiling_0) {
    using namespace optiling;
    std::string op_name = "EmbeddingDenseGrad";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"num_weights\": 20000, \"padding_idx\": 20, \"scale_grad_by_freq\": 1}}";

    std::vector<int64_t> input0{20000, 512};
    std::vector<int64_t> input1{20000};
    std::vector<int64_t> output{20000, 512};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "20000 512 1 32 ");
}

TEST_F(EmbeddingDenseGradTiling, embedding_dense_grad_tiling_1) {
    using namespace optiling;
    std::string op_name = "EmbeddingDenseGrad";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"num_weights\": 20000, \"padding_idx\": 20, \"scale_grad_by_freq\": 1}}";

    std::vector<int64_t> input0{30000, 1024};
    std::vector<int64_t> input1{30000};
    std::vector<int64_t> output{20000, 1024};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "30000 1024 1 32 ");
}

TEST_F(EmbeddingDenseGradTiling, embedding_dense_grad_tiling_2) {
    using namespace optiling;
    std::string op_name = "EmbeddingDenseGrad";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"num_weights\": 20000, \"padding_idx\": 10, \"scale_grad_by_freq\": 0}}";

    std::vector<int64_t> input0{10000, 768};
    std::vector<int64_t> input1{10000};
    std::vector<int64_t> output{20000, 768};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "10000 768 1 32 ");
}

