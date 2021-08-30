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
#include "register/op_tiling_registry.h"

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

TEST_F(NLLLossTiling, NLLLoss_tiling1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {16, 32};
  std::vector<int64_t> input_target_shape = {16,};
  std::vector<int64_t> input_weight_shape = {32,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 16 16 32 1 0 1 1 0 1 59392 1856 32 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {16, 32};
  std::vector<int64_t> input_target_shape = {16,};
  std::vector<int64_t> input_weight_shape = {32,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"none\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 2 16 32 8 0 8 8 0 8 59392 1856 32 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling3) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {1, 3991};
  std::vector<int64_t> input_target_shape = {1,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 1 1 3991 0 0 0 1 0 1 59872 16 3992 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling4) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {220, 3991};
  std::vector<int64_t> input_target_shape = {220,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"none\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 28 220 3991 8 1 0 4 0 4 59872 16 3992 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling5) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {220, 3991};
  std::vector<int64_t> input_target_shape = {220,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 32 220 3991 7 0 7 3 0 3 59872 16 3992 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling6) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {7, 39};
  std::vector<int64_t> input_target_shape = {7,};
  std::vector<int64_t> input_weight_shape = {39,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"none\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 1 7 39 0 0 0 7 0 7 60296 1552 40 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling7) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 3991};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {3991,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 32 2000 3991 63 4 3 47 3 2 59872 16 3992 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling8) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "1 32 2000 15003 63 21 0 47 15 2 45016 8 15008 ");
}

TEST_F(NLLLossTiling, NLLLoss_tiling9) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"REDUCTION\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(NLLLossTiling, NLLLoss_tiling10) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"CORE\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(NLLLossTiling, NLLLoss_tiling11) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"UB\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(NLLLossTiling, NLLLoss_tiling12) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 100, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"UB\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(NLLLossTiling, NLLLoss_tiling13) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000, 100};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(NLLLossTiling, NLLLoss_tiling14) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {2000,};
  std::vector<int64_t> input_weight_shape = {1199,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(NLLLossTiling, NLLLoss_tiling15) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {2000, 15003};
  std::vector<int64_t> input_target_shape = {1100,};
  std::vector<int64_t> input_weight_shape = {15003,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(NLLLossTiling, NLLLoss_tiling16) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLoss");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorOutputArgTotalWeight, tensorOutputArgY;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {220, 200000};
  std::vector<int64_t> input_target_shape = {220,};
  std::vector<int64_t> input_weight_shape = {200000,};
  std::vector<int64_t> output_total_weight_shape = {1,};
  std::vector<int64_t> output_y_shape = {1,};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;

  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;

  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;


  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);

  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);

  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);

  TeOpTensor tensorOutputY;
  tensorOutputY.shape = output_y_shape;
  tensorOutputY.dtype = dtype;
  tensorOutputArgY.tensor.push_back(tensorOutputY);
  tensorOutputArgY.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgY);
  
  TeOpTensor tensorOutputTotalWeight;
  tensorOutputTotalWeight.shape = output_total_weight_shape;
  tensorOutputTotalWeight.dtype = dtype;
  tensorOutputArgTotalWeight.tensor.push_back(tensorOutputTotalWeight);
  tensorOutputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputArgTotalWeight);
  
  opParas.op_type = "NLLLoss";
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65024, \"core_num\": 32, \"reduction\": \"sum\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "new case NLLLossTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "2 32 220 200000 7 0 7 3 0 3 8 21669 8 ");
}