#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>
#define private public
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "nn_norm_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class SoftmaxCrossEntropyWithLogitsTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "SoftmaxCrossEntropyWithLogitsTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "SoftmaxCrossEntropyWithLogitsTiling TearDown" << std::endl; }
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

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_1) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_2) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_3) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_4) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_5) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_6) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_7) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_8) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_9) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_10) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_11) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_12) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_13) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 1-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_14) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 2-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_15) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 2-2
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_16) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 3-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_17) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 4-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_18) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 5-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_19) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 5-2
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_20) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// two known case 6-1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_21) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// one known case 1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_22) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// one known case 2
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_23) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// one known case 3
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_24) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// one known case 4
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_25) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// no known case 1
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_26) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": 8},
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 8 12 8 ");
}

// no known broadcast case 1  key8
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_27) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": 1},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 1},
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

  std::vector<std::vector<int64_t>> inputs{{98, 8}, {98, 1}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(runInfo.GetTilingKey(), 8);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 8 1 12 8 ");
}

// no known broadcast case 2  key2
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_28) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 1, "labels_shape0": 98, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 1,
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

  std::vector<std::vector<int64_t>> inputs{{98, 1}, {98, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(runInfo.GetTilingKey(), 2);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 98 1 8 12 8 ");
}

// no known broadcast case 3  key6
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_29) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 1, "labels_shape0": 1, "labels_shape1": 8},
                        "range": {"features_range0_l":1, "features_range0_r":98, "features_range1_l": 1, "features_range1_r": 1,
                                  "labels_range0_l":1, "labels_range0_r":1, "labels_range1_l": 2, "labels_range1_r": 10},
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

  std::vector<std::vector<int64_t>> inputs{{98, 1}, {1, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(runInfo.GetTilingKey(), 6);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "98 1 1 8 12 8 ");
}

// no known broadcast case 4  key9
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_30) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 1, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": 1},
                        "range": {"features_range0_l":1, "features_range0_r":1, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 1},
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

  std::vector<std::vector<int64_t>> inputs{{1, 8}, {98, 1}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(runInfo.GetTilingKey(), 9);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 98 8 1 12 8 ");
}
// no known broadcast case 4  key9
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_31) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 1, "features_shape1": 8, "labels_shape0": 98, "labels_shape1": 1},
                        "range": {"features_range0_l":1, "features_range0_r":1, "features_range1_l": 2, "features_range1_r": 10,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 1, "labels_range1_r": 1},
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

  std::vector<std::vector<int64_t>> inputs{{1, 8}, {98, 1}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(runInfo.GetTilingKey(), 9);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 98 8 1 12 8 ");
}

// no known large c_size case
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_32) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 98, "features_shape1": 8000, "labels_shape0": 98, "labels_shape1": 8000},
                        "range": {"features_range0_l":98, "features_range0_r":98, "features_range1_l": 2, "features_range1_r": 10000,
                                  "labels_range0_l":1, "labels_range0_r":98, "labels_range1_l": 2, "labels_range1_r": 10000},
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

  std::vector<std::vector<int64_t>> inputs{{98, 8000}, {98, 8000}};

  std::vector<std::vector<int64_t>> outputs{{
                                                98,
                                            },
                                            {98, 8000}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

// no known no multi-core case
TEST_F(SoftmaxCrossEntropyWithLogitsTiling, SoftmaxCrossEntropyWithLogits_tiling_test_33) {
  std::string op_name = "SoftmaxCrossEntropyWithLogits";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "SoftmaxCrossEntropyWithLogits",
                        "ori_shape": {"features_shape0": 7, "features_shape1": 8, "labels_shape0": 7, "labels_shape1": 8},
                        "range": {"features_range0_l":7, "features_range0_r":7, "features_range1_l": 8, "features_range1_r": 8,
                                  "labels_range0_l":7, "labels_range0_r":7, "labels_range1_l": 8, "labels_range1_r": 8},
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

  std::vector<std::vector<int64_t>> inputs{{7, 8}, {7, 8}};

  std::vector<std::vector<int64_t>> outputs{{
                                                7,
                                            },
                                            {7, 8}};

  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::SoftmaxCrossEntropyWithLogits(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, features);
  TENSOR_INPUT(opParas, tensor_inputB, labels);
  TENSOR_OUTPUT(opParas, tensor_outputA, loss);
  TENSOR_OUTPUT(opParas, tensor_outputB, backprop);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 7 8 8 1 7 ");
}
