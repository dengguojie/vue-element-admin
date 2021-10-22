#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class LayerNormXBackpropV2Tiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "LayerNormXBackpropV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LayerNormXBackpropV2Tiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t))
    {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

TEST_F(LayerNormXBackpropV2Tiling, LayerNormXBackpropV2_tiling_test_1)
{
    using namespace optiling;
    std::string op_name = "LayerNormXBackpropV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
    std::string compileInfo = R"({
                        "_pattern": "Layer_norm_x_backprop_v2",
                        "UB_SIZE":262112,
                        "CORE_NUM":32,
                        "MAX_DTYPE": 4,
                        "COEXISTING_QUANTITY": 7,
                        "_vars": {"10000": ["dim_0", "dim_1"]},
                        "_normal_vars": {"10000": []},
                        "_attr_vars": {"10000": []},
                        "_custom_vars": {"10000": ["dim_0", "dim_1"]}
                        })";

    std::vector<std::vector<int64_t>> inputs{
        {13, 32, 512},
        {13, 32, 512},
        {13, 32, 1},
        {13, 32, 1},
        {512}};

    std::vector<std::vector<int64_t>> outputs{
        {13, 32, 512},
        {13, 32, 512}};

    std::vector<std::string> input_types{"float32", "float32", "float32", "float32", "float32"};
    std::string output_type = {"float32", "float32"};
    std::string data_format = "ND";

    TeOpParas opParas;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        TeOpTensor tensor_input;
        TeOpTensorArg tensor_arg;
        tensor_input.shape = inputs[i];
        tensor_input.dtype = input_types[i];
        tensor_input.format = data_format;
        tensor_arg.tensor.push_back(tensor_input);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.inputs.push_back(tensor_arg);
    }
    for (size_t i = 0; i < outputs.size(); i++)
    {
        TeOpTensor tensor_output;
        TeOpTensorArg tensor_arg;
        tensor_output.shape = outputs[i];
        tensor_output.dtype = output_type;
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "LayerNormXBackpropV2_tiling_test_1";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 10000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "13 32 ");
}
