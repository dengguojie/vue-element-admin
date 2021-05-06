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
                        "_attr_vars": {"180000": [], "90000": [], "480000": [], "290000": [], "780000": [], "490000":[], "3480000":[],"2290000":[], "3780000":[],"2490000":[],"6780000":[],"4490000":[]},
                        "_custom_vars": {
                        "180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "90000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"180000": [], "90000": [], "480000": [], "290000": [], "780000": [], "490000":[], "3480000":[],"2290000":[], "3780000":[],"2490000":[],"6780000":[],"4490000":[]}, 
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "90000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
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
    EXPECT_EQ(runInfo.tiling_key, 290000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "11 12 512 989855744 11 12 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_2)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"260000": [], "280000": [], "130000": [], "560000": [], "580000": [], "330000":[], "860000":[],"880000":[], "530000":[],"3560000":[],"3580000":[],"2330000":[], "3860000":[], "3880000":[], "2530000":[], "6860000":[], "6880000":[], "4530000":[]},
                        "_custom_vars": {
                        "260000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "280000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "130000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "330000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "530000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2330000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2530000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4530000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"260000": [], "280000": [], "130000": [], "560000": [], "580000": [], "330000":[], "860000":[],"880000":[], "530000":[],"3560000":[],"3580000":[],"2330000":[], "3860000":[], "3880000":[], "2530000":[], "6860000":[], "6880000":[], "4530000":[]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "260000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "280000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "130000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "330000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "530000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2330000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2530000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4530000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [13],
                        "reduce_axis": [1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{
        {1024, 30, 512},
        {512},
        {512}};

    std::vector<std::vector<int64_t>> outputs{
        {1024, 30, 512},
        {1024, 1, 1},
        {1024, 1, 1}};

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
    EXPECT_EQ(runInfo.tiling_key, 560000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "1024 30 512 948471945 32 20 23 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_3)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"180000": [], "90000": [], "480000": [], "290000": [], "780000": [], "490000":[], "3480000":[],"2290000":[], "3780000":[],"2490000":[],"6780000":[],"4490000":[]},
                        "_custom_vars": {
                        "180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "90000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"180000": [], "90000": [], "480000": [], "290000": [], "780000": [], "490000":[], "3480000":[],"2290000":[], "3780000":[],"2490000":[],"6780000":[],"4490000":[]}, 
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "180000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "90000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3480000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2290000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6780000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4490000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [9],
                        "reduce_axis": [2],
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
    op_compile_info.key = "LayerNorm_tiling_test_3";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(runInfo.tiling_key, 290000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 989855744 34 20 0 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_4)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"420000": [], "440000": [], "460000": [], "210000": [], "720000": [], "740000":[], "760000":[],"410000":[], "1020000":[],"1040000":[],"1060000":[], "610000":[], "3720000":[], "3740000":[], "3760000":[], "2410000":[], "4020000":[], "4040000":[], "4060000":[], "2610000":[], "7020000":[], "7040000":[], "7060000":[],"4610000":[]},
                        "_custom_vars": {
                        "420000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "440000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "460000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "210000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "740000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "760000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "410000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "1020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "1040000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "1060000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "610000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3740000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3760000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2410000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4040000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4060000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2610000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "7020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "7040000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "7060000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4610000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"420000": [], "440000": [], "460000": [], "210000": [], "720000": [], "740000":[], "760000":[],"410000":[], "1020000":[],"1040000":[],"1060000":[], "610000":[], "3720000":[], "3740000":[], "3760000":[], "2410000":[], "4020000":[], "4040000":[], "4060000":[], "2610000":[], "7020000":[], "7040000":[], "7060000":[],"4610000":[]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "420000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "440000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "460000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "210000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "740000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "760000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "410000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "1020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "1040000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "1060000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "610000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3720000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3740000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "3760000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2410000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4040000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4060000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "2610000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"],
                        "7020000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "7040000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "7060000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "4610000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [21],
                        "reduce_axis": [0,1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "ub_info":[16384]})";

    std::vector<std::vector<int64_t>> inputs{
        {34, 309, 512},
        {512},
        {512}};

    std::vector<std::vector<int64_t>> outputs{
        {34, 309, 512},
        {1, 1, 1},
        {1, 1, 1}};

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
    EXPECT_EQ(runInfo.tiling_key, 740000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 877108573 34 20 34 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_5)
{
    using namespace optiling;
    std::string op_name = "LayerNorm";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_attr_vars": {"130000": [], "560000": [], "580000": [], "860000":[],"880000":[], "3560000":[],"3580000":[], "3860000":[], "3880000":[], "6860000":[], "6880000":[]},
                        "_custom_vars": {
                        "130000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "_normal_vars": {"130000": [], "560000": [], "580000": [], "860000":[],"880000":[], "3560000":[],"3580000":[], "3860000":[], "3880000":[], "6860000":[], "6880000":[]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "130000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3560000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3580000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "3880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6860000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"], 
                        "6880000": ["dim0_0", "dim0_1", "dim0_2", "mean_cof", "block_factor", "ub_factor", "ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "pattern_info": [13],
                        "reduce_axis": [1,2],
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
    EXPECT_EQ(runInfo.tiling_key, 560000);
    EXPECT_EQ(to_string(runInfo.tiling_data), "34 309 512 919869235 8 20 8 ");
}