#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class LayerNormTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "LayerNormTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LayerNormTiling TearDown" << std::endl;
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

TEST_F(LayerNormTiling, LayerNorm_tiling_test_1)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"2290000": [], "290000": [], "3780000": [], "480000": [], "780000": [], "90000":[]},
                        "_custom_vars": {
                        "2290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "90000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"2290000": [], "290000": [], "3780000": [], "480000": [], "780000": [], "90000":[]}, 
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "2290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "90000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 512,
                        "pattern_info": [9],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{
        {11, 12, 512},
        {512},
        {512}};

    std::vector<std::vector<int64_t>> outputs{
        {11, 12, 512},
        {11, 12, 1},
        {11, 12, 1}};

    std::vector<std::string> input_types{"float16", "float16", "float16"};
    std::vector<std::string> output_types{"float16", "float16", "float16"};
    std::string data_format = "NCHWC";

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
        tensor_output.dtype = output_types[i];
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "LayerNorm_tiling_test_1";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(runInfo.tiling_key, 90000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "11 12 512 989855744 16 1 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_2)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"130000": [], "560000": [], "580000": [], "860000": [], "880000": []},
                        "_custom_vars": {
                        "130000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"130000": [], "560000": [], "580000": [], "860000": [], "880000": []}, 
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "130000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 512,
                        "pattern_info": [13],
                        "reduce_axis": [1, 2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{
        {34, 309, 512},
        {512},
        {512}};

    std::vector<std::vector<int64_t>> outputs{
        {34, 309, 512},
        {34, 1, 1},
        {34, 1, 1}};

    std::vector<std::string> input_types{"float16", "float16", "float16"};
    std::vector<std::string> output_types{"float16", "float16", "float16"};
    std::string data_format = "NCHWC";

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
        tensor_output.dtype = output_types[i];
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "LayerNorm_tiling_test_2";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 3);
    EXPECT_EQ(runInfo.tiling_key, 560000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 919869235 16 20 16 ");
}