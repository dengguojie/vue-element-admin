/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_nll_loss_grad_tiling.cpp
 * \brief dynamic tiling test of nll_loss_grad
 */
#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "math_ops.h"
#include "array_ops.h"

class NLLLossTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NLLLossTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NLLLossTiling TearDown" << std::endl;
  }
};

static std::string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  std::string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

using namespace ge;
#include "test_common.h"
/*
.INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(total_weight, TensorType({DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
*/

TEST_F(NLLLossTiling, NLLLoss_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {16, 32};
  std::vector<int64_t> input_target_shape = {16,};
  std::vector<int64_t> input_weight_shape = {32,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);

  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 16 16 32 1 0 1 1 0 1 59392 1856 32 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {16, 32};
  std::vector<int64_t> input_target_shape = {16,};
  std::vector<int64_t> input_weight_shape = {32,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);

  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"none\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 2 16 32 8 0 8 8 0 8 59392 1856 32 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {1, 3991};
  std::vector<int64_t> input_target_shape = {1,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 1 1 3991 0 0 0 1 0 1 59872 16 3992 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {220, 3991};
  std::vector<int64_t> input_target_shape = {220,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"none\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 28 220 3991 8 1 0 4 0 4 59872 16 3992 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {220, 3991};
  std::vector<int64_t> input_target_shape = {220,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 32 220 3991 7 0 7 3 0 3 59872 16 3992 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {7, 39};
  std::vector<int64_t> input_target_shape = {7,};
  std::vector<int64_t> input_weight_shape = {39,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"none\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 1 7 39 0 0 0 7 0 7 60296 1552 40 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling7) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 3991};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 32 2000 3991 63 4 3 47 3 2 59872 16 3992 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling8) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "1 32 2000 15003 63 21 0 47 15 2 45016 8 15008 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling9) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"REDUCTION\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling10) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"CORE\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling11) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"UB\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling12) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 100, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"UB\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling13) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000, 100};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling14) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {1199,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling15) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {1100,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(NLLLossTiling, NLLLoss_tiling16) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::NLLLoss("NLLLoss");
  std::vector<int64_t> input_x_shape = {220, 200000};
  std::vector<int64_t> input_target_shape = {220,};
  std::vector<int64_t> input_weight_shape = {200000,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::DataType dtype_target = ge::DT_INT32;


  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(dtype);

  TensorDesc tensorInputTarget;
  tensorInputTarget.SetShape(ge::Shape(input_target_shape));
  tensorInputTarget.SetDataType(dtype_target);

  TensorDesc tensorInputWeight;
  tensorInputWeight.SetShape(ge::Shape(input_weight_shape));
  tensorInputWeight.SetDataType(dtype);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputTarget, target);
  TENSOR_INPUT(opParas, tensorInputWeight, weight);
  TensorDesc tensorOutputY;
  tensorOutputY.SetShape(ge::Shape(output_y_shape));
  tensorOutputY.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputY, y);

  TensorDesc tensorOutputTotalWeight;
  tensorOutputTotalWeight.SetShape(ge::Shape(output_total_weight_shape));
  tensorOutputTotalWeight.SetDataType(dtype);
  TENSOR_OUTPUT(opParas, tensorOutputTotalWeight, total_weight);

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  std::cout << "new case NLLLossTilingData: " << to_string(runInfo.GetAllTilingData()) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.GetAllTilingData()),
      "2 32 220 200000 7 0 7 3 0 3 8 21669 8 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}