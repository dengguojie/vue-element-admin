#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class SoftmaxCrossEntropyWithLogitsTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsTiling TearDown" << std::endl;
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

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_1)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": -1},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, true, false],
                        "base_info": {
                        "130": [262144, 4, 10, 30],
                        "140": [262144, 4, 10, 30],
                        "230": [262144, 4, 10, 30],
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {
                        "112130000": [10000, 10100],
                        "112140000": [10000, 10100],
                        "112230000": [10000, 10100],
                        "0": [10000, 10100]},
                        "_vars": {
                        "112130000": ["_dim0_0", "_dim1_0", "block_factor_0", "ub_factor_0"],
                        "112140000": ["_dim0_0", "_dim1_0", "block_factor_0", "ub_factor_0"],
                        "112230000": ["_dim0_0", "_dim1_0", "block_factor_0", "ub_factor_0"],
                        "0": ["_dim0_0", "_dim1_0", "block_factor_0", "ub_factor_0"]},
                        "_normal_vars": {
                        "112130000": ["_dim0_0", "_dim1_0"],
                        "112140000": ["_dim0_0", "_dim1_0"],
                        "112230000": ["_dim0_0", "_dim1_0"],
                        "0": ["_dim0_0", "_dim1_0"]},
                        "_attr_vars": {
                        "112130000": [],
                        "112140000": [],
                        "112230000": [],
                        "0": []},
                        "_custom_vars": {
                        "112130000": ["block_factor_0", "ub_factor_0"],
                        "112140000": ["block_factor_0", "ub_factor_0"],
                        "112230000": ["block_factor_0", "ub_factor_0"],
                        "0": ["block_factor_0", "ub_factor_0"]}})";

    std::vector<std::vector<int64_t>> inputs{
        {98, 8},
        {98, 8}};

    std::vector<std::vector<int64_t>> outputs{
        {98,},
        {98, 8}};

    std::vector<std::string> input_types{"float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32"};
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
        tensor_output.dtype = output_types[i];
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_1";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(runInfo.tiling_key, 112130000);
    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 8 8 98 98 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_2)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": 1024, "labels_shape1": -1},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, true, false],
                        "base_info": {
                        "130": [262144, 4, 10, 30],
                        "140": [262144, 4, 10, 30],
                        "230": [262144, 4, 10, 30],
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": [10000, 10100]},
                        "_vars": {"0": ["_dim0_0", "_dim1_0", "block_factor_0", "ub_factor_0"]},
                        "_normal_vars": {"0": ["_dim0_0", "_dim1_0"]},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["block_factor_0", "ub_factor_0"]}})";

    std::vector<std::vector<int64_t>> inputs{
        {1024, 8},
        {1024, 8}};

    std::vector<std::vector<int64_t>> outputs{
        {1024,},
        {1024, 8}};

    std::vector<std::string> input_types{"float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32"};
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
        tensor_output.dtype = output_types[i];
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_2";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(runInfo.tiling_key, 112130000);
    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "1024 8 1024 819 ");
}


TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_3)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 8, "features_shape1": -1, "labels_shape0": 8, "labels_shape1": -1},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "130": [262144, 4, 10, 32],
                        "140": [262144, 4, 10, 32],
                        "230": [262144, 4, 10, 32],
                        "000": [262144, 4, 10, 32]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim0_0", "dim1_0", "block_factor_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim0_0", "dim1_0", "block_factor_0", "ub_factor_0"]}})";

    std::vector<std::vector<int64_t>> inputs{
        {8, 80},
        {8, 80}};

    std::vector<std::vector<int64_t>> outputs{
        {8,},
        {8, 80}};

    std::vector<std::string> input_types{"float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32"};
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
        tensor_output.dtype = output_types[i];
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_3";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(runInfo.tiling_key, 0);
    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "8 80 80 8 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_4)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": 1024, "labels_shape1": -1},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "130": [262144, 4, 10, 32],
                        "140": [262144, 4, 10, 32],
                        "230": [262144, 4, 10, 32],
                        "000": [262144, 4, 10, 32]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim0_0", "dim1_0", "block_factor_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim0_0", "dim1_0", "block_factor_0", "ub_factor_0"]}})";

    std::vector<std::vector<int64_t>> inputs{
        {1024, 4800},
        {1024, 4800}};

    std::vector<std::vector<int64_t>> outputs{
        {1024,},
        {1024, 4800}};

    std::vector<std::string> input_types{"float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32"};
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
        tensor_output.dtype = output_types[i];
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_3";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(runInfo.tiling_key, 0);
    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "1024 4800 1024 1 ");
}
