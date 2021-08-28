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

class NLLLossGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NLLLossGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NLLLossGradTiling TearDown" << std::endl;
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

TEST_F(NLLLossGradTiling, NLLLossGrad_tiling1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgYGrad, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorInputArgTotalWeight, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {16, 32};
  std::vector<int64_t> input_y_grad_shape = {16};
  std::vector<int64_t> input_target_shape = {16};
  std::vector<int64_t> input_weight_shape = {32};
  std::vector<int64_t> input_total_weight_shape = {1};
  std::vector<int64_t> output_shape = {16, 32};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;
  TeOpTensor tensorInputYGrad;
  tensorInputYGrad.shape = input_y_grad_shape;
  tensorInputYGrad.dtype = dtype;
  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;
  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;
  TeOpTensor tensorInputTotalWeight;
  tensorInputTotalWeight.shape = input_total_weight_shape;
  tensorInputTotalWeight.dtype = dtype;
  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);
  tensorInputArgYGrad.tensor.push_back(tensorInputYGrad);
  tensorInputArgYGrad.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgYGrad);
  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);
  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);
  tensorInputArgTotalWeight.tensor.push_back(tensorInputTotalWeight);
  tensorInputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTotalWeight);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "NLLLossGrad";
  std::string compileInfo1 =
      "{\"vars\": {\"reduction\": \"None\", \"dtype\": \"float32\", \"dtype_weight\": \"int32\", \"ignore_idx\": 5, "
      "\"ub_size\": 252928, \"block_dim\": 2}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo1;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossGradTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(
      to_string(runInfo.tiling_data),
      "32 16 0 5 512 512 16 16 1 32 0 2 8 8 1 1 0 256 256 256 64 64 64 64 4 1 1 1 1 1 1 4 4 32 32 64 2000 0 0 0 0 0 ");
}

TEST_F(NLLLossGradTiling, NLLLossGrad_tiling2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgYGrad, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorInputArgTotalWeight, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {40, 255658};
  std::vector<int64_t> input_y_grad_shape = {40};
  std::vector<int64_t> input_target_shape = {40};
  std::vector<int64_t> input_weight_shape = {255658};
  std::vector<int64_t> input_total_weight_shape = {1};
  std::vector<int64_t> output_shape = {40, 255658};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;
  TeOpTensor tensorInputYGrad;
  tensorInputYGrad.shape = input_y_grad_shape;
  tensorInputYGrad.dtype = dtype;
  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;
  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;
  TeOpTensor tensorInputTotalWeight;
  tensorInputTotalWeight.shape = input_total_weight_shape;
  tensorInputTotalWeight.dtype = dtype;
  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);
  tensorInputArgYGrad.tensor.push_back(tensorInputYGrad);
  tensorInputArgYGrad.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgYGrad);
  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);
  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);
  tensorInputArgTotalWeight.tensor.push_back(tensorInputTotalWeight);
  tensorInputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTotalWeight);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "NLLLossGrad";
  std::string compileInfo1 =
      "{\"vars\": {\"reduction\": \"None\", \"dtype\": \"float32\", \"dtype_weight\": \"int32\", \"ignore_idx\": "
      "215907, "
      "\"ub_size\": 260096, \"block_dim\": 32}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo1;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossGradTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "255658 40 0 215907 10226320 10226320 40 40 1 255658 1 32 0 0 2 0 0 1 0 64960 64 64 64 8 0 0 0 0 0 0 0 0 0 "
            "8120 7598 64 2001 64960 4 1015 950 64960 ");
}

TEST_F(NLLLossGradTiling, NLLLossGrad_tiling3) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgYGrad, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorInputArgTotalWeight, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {1020, 15003};
  std::vector<int64_t> input_y_grad_shape = {1};
  std::vector<int64_t> input_target_shape = {1020};
  std::vector<int64_t> input_weight_shape = {15003};
  std::vector<int64_t> input_total_weight_shape = {1};
  std::vector<int64_t> output_shape = {1020, 15003};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;
  TeOpTensor tensorInputYGrad;
  tensorInputYGrad.shape = input_y_grad_shape;
  tensorInputYGrad.dtype = dtype;
  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;
  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;
  TeOpTensor tensorInputTotalWeight;
  tensorInputTotalWeight.shape = input_total_weight_shape;
  tensorInputTotalWeight.dtype = dtype;
  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);
  tensorInputArgYGrad.tensor.push_back(tensorInputYGrad);
  tensorInputArgYGrad.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgYGrad);
  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);
  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);
  tensorInputArgTotalWeight.tensor.push_back(tensorInputTotalWeight);
  tensorInputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTotalWeight);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "NLLLossGrad";
  std::string compileInfo1 =
      "{\"vars\": {\"reduction\": \"mean\", \"dtype\": \"float32\", \"dtype_weight\": \"int32\", \"ignore_idx\": "
      "-1, "
      "\"ub_size\": 260096, \"block_dim\": 32}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo1;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossGradTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "15003 1020 1 -1 15303060 15303060 1 1020 1 15003 0 32 3 3 11 340 28 45009 45009 45056 64 15040 64 64 1876 "
            "1 1 1 1 1 1 704 704 "
            "5627 5627 64 2000 0 0 0 0 0 ");
}

TEST_F(NLLLossGradTiling, NLLLossGrad_tiling4) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("NLLLossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputArgX, tensorInputArgYGrad, tensorInputArgTarget, tensorInputArgWeight;
  TeOpTensorArg tensorInputArgTotalWeight, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_x_shape = {7, 7243};
  std::vector<int64_t> input_y_grad_shape = {1};
  std::vector<int64_t> input_target_shape = {7};
  std::vector<int64_t> input_weight_shape = {7243};
  std::vector<int64_t> input_total_weight_shape = {1};
  std::vector<int64_t> output_shape = {7, 7243};
  std::string dtype = "float32";
  std::string dtype_target = "int32";

  TeOpTensor tensorInputX;
  tensorInputX.shape = input_x_shape;
  tensorInputX.dtype = dtype;
  TeOpTensor tensorInputYGrad;
  tensorInputYGrad.shape = input_y_grad_shape;
  tensorInputYGrad.dtype = dtype;
  TeOpTensor tensorInputTarget;
  tensorInputTarget.shape = input_target_shape;
  tensorInputTarget.dtype = dtype_target;
  TeOpTensor tensorInputWeight;
  tensorInputWeight.shape = input_weight_shape;
  tensorInputWeight.dtype = dtype;
  TeOpTensor tensorInputTotalWeight;
  tensorInputTotalWeight.shape = input_total_weight_shape;
  tensorInputTotalWeight.dtype = dtype;
  tensorInputArgX.tensor.push_back(tensorInputX);
  tensorInputArgX.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgX);
  tensorInputArgYGrad.tensor.push_back(tensorInputYGrad);
  tensorInputArgYGrad.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgYGrad);
  tensorInputArgTarget.tensor.push_back(tensorInputTarget);
  tensorInputArgTarget.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTarget);
  tensorInputArgWeight.tensor.push_back(tensorInputWeight);
  tensorInputArgWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgWeight);
  tensorInputArgTotalWeight.tensor.push_back(tensorInputTotalWeight);
  tensorInputArgTotalWeight.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputArgTotalWeight);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "NLLLossGrad";
  std::string compileInfo1 =
      "{\"vars\": {\"reduction\": \"sum\", \"dtype\": \"float32\", \"dtype_weight\": \"int32\", \"ignore_idx\": "
      "1, "
      "\"ub_size\": 260096, \"block_dim\": 32}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo1;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "NLLLossGradTilingData: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "7243 7 0 1 50701 50701 1 7 1 7243 0 7 1 1 1 1 0 7243 7243 7296 64 7296 64 64 906 "
            "1 1 1 1 1 1 114 114 "
            "906 906 64 2000 0 0 0 0 0 ");
}
