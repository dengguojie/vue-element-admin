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
                        "range": {"features_range0_l":71, "features_range0_r":114, "features_range1_l": 7, "features_range1_r": 10,
                                  "labels_range0_l":71, "labels_range0_r":114, "labels_range1_l": 7, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_0", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_2)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":98, "features_range0_r":98, "features_range1_l": 7, "features_range1_r": 10,
                                  "labels_range0_l":98, "labels_range0_r":98, "labels_range1_l": 7, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_2";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_3)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_1_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_3";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_4)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":2, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":2, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_4";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_5)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_5";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_6)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_6";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_7)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": 8},
                        "range": {"features_range0_l":2, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":2, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_7";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_8)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": 98, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_8";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_9)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": 98, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_9";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_10)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": 8, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_1_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_10";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_11)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": 8, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":2, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":2, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_11";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_12)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_1_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_1_0", "dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_12";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_13)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_13";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 1-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_14)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": -1, "labels_shape0": 98, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_14";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 2-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_15)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": 8, "labels_shape0": -1, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_0", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_1_0", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_15";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 2-2
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_16)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": 8, "labels_shape0": -1, "labels_shape1": 8},
                        "range": {"features_range0_l":2, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":2, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_16";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 3-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_17)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": -1},
                        "range": {"features_range0_l":2, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":2, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_17";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 4-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_18)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": -1, "labels_shape0": -1, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_1_0", "dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_18";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 5-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_19)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": -1, "labels_shape0": 98, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_1", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_19";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 5-2
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_20)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": -1, "labels_shape0": 98, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_20";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// two known case 6-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_21)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 8, "labels_shape0": -1, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_1_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_1_0", "dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_21";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// one known case 1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_22)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": -1, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_0", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_0", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_22";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// one known case 2
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_23)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": -1, "labels_shape0": 98, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_0_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_0_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_23";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// one known case 3
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_24)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 8, "labels_shape0": -1, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_1_0", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_1_0", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_24";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// one known case 4
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_25)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": -1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["dim_1_1", "block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["dim_1_1", "block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_25";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}

// no known case 1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_26)
{
    using namespace optiling;
    std::string op_name = "SoftmaxCrossEntropyWithLogits";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": 98},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10},
                        "common_info" : {
                        "ub_size" : 262144,
                        "core_num" : 32},
                        "flag_info": [false, false, true, false, false],
                        "base_info": {
                        "000": [262144, 4, 10, 30]},
                        "elewise_vars": {"0": []},
                        "_vars": {"0": ["block_nparts_0", "ub_factor_0"]},
                        "_normal_vars": {"0": []},
                        "_attr_vars": {"0": []},
                        "_custom_vars": {"0": ["block_nparts_0", "ub_factor_0"]}})";

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
    op_compile_info.key = "SoftmaxCrossEntropyWithLogits_tiling_test_26";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 12);

    std::cout << "to_string(runInfo.tiling_data)" << to_string(runInfo.tiling_data) << std::endl;
    EXPECT_EQ(to_string(runInfo.tiling_data), "98 98 8 8 12 8 ");
}
