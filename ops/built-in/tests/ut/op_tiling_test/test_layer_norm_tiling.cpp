#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "register/op_tiling_registry.h"

using namespace std;

class LayerNormTiling : public testing::Test
{
protected:
    static void SetUpTestCase() { std::cout << "LayerNormTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LayerNormTiling TearDown" << std::endl; }
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
                        "_attr_vars": {"1940000": [], "270000": [], "670000": [], "6710000": []},
                        "_custom_vars": {
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "6710000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940000": [], "270000": [], "670000": [], "6710000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "6710000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{{11, 12, 512}, {512}, {512}};

    std::vector<std::vector<int64_t>> outputs{{11, 12, 512}, {11, 12, 1}, {11, 12, 1}};

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
    EXPECT_EQ(runInfo.tiling_key, 670000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "11 12 512 989855744 11 1 12 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_2)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": []},
                        "_custom_vars": {
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [39],
                        "reduce_axis": [1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{{1024, 30, 512}, {512}, {512}};

    std::vector<std::vector<int64_t>> outputs{{1024, 30, 512}, {1024, 1, 1}, {1024, 1, 1}};

    std::vector<std::string> input_types{"float32", "float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32", "float32"};
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
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 1480000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "1024 30 512 948471945 32 1 20 23 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_3)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": []},
                        "_custom_vars": {
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [39],
                        "reduce_axis": [1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};

    std::vector<std::vector<int64_t>> outputs{{34, 309, 512}, {34, 1, 1}, {34, 1, 1}};

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
    op_compile_info.key = "LayerNorm_tiling_test_3";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 3);
    EXPECT_EQ(runInfo.tiling_key, 1480000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 919869235 16 1 20 16 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_4)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"1260000": [],"1320000": [],"1380000": [],"1960000": [],"2020000": [],"2080000": [],"2660000": [],"2720000": [],"2780000": []},
                        "_custom_vars": {
                        "1260000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "1320000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "1380000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "1960000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2080000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2660000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1260000": [],"1320000": [],"1380000": [],"1960000": [],"2020000": [],"2080000": [],"2660000": [],"2720000": [],"2780000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1260000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "1320000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "1380000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "1960000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2080000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2660000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"],
                        "2780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_facto_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [63],
                        "reduce_axis": [0,1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};

    std::vector<std::vector<int64_t>> outputs{{34, 309, 512}, {1, 1, 1}, {1, 1, 1}};

    std::vector<std::string> input_types{"float32", "float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32", "float32"};
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
    op_compile_info.key = "LayerNorm_tiling_test_4";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(runInfo.tiling_key, 2020000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 877108573 34 1 20 34 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_5)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": []},
                        "_custom_vars": {
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "1540000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "2240000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "390000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [39],
                        "reduce_axis": [1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};

    std::vector<std::vector<int64_t>> outputs{{34, 309, 512}, {34, 1, 1}, {34, 1, 1}};

    std::vector<std::string> input_types{"float32", "float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32", "float32"};
    std::string data_format = "NCHW";

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
    op_compile_info.key = "LayerNorm_tiling_test_5";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 5);
    EXPECT_EQ(runInfo.tiling_key, 1480000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 919869235 8 1 20 8 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_6)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"1940000": [], "270000": [], "670000": [], "6710000": []},
                        "_custom_vars": {
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "6710000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940000": [], "270000": [], "670000": [], "6710000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "6710000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{{20, 304, 512}, {512}, {512}};

    std::vector<std::vector<int64_t>> outputs{{20, 304, 512}, {20, 304, 1}, {20, 304, 1}};

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
    op_compile_info.key = "LayerNorm_tiling_test_6";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 671000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "20 304 512 989855744 2 16 19 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_7)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"1940000": [], "270000": [], "670000": [], "6710000": []},
                        "_custom_vars": {
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "6710000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"1940000": [], "270000": [], "670000": [], "6710000": []},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "270000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "670000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"], 
                        "6710000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "block_factor_1", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{{49, 304, 512}, {512}, {512}};

    std::vector<std::vector<int64_t>> outputs{{49, 304, 512}, {49, 304, 1}, {49, 304, 1}};

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
    op_compile_info.key = "LayerNorm_tiling_test_7";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 784);
    EXPECT_EQ(runInfo.tiling_key, 671000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "49 304 512 989855744 49 16 19 0 ");
}