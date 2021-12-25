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
 * \file test_tabulate_fusion_grad_tiling.cpp
 * \brief dynamic tiling test of TabulateFusionGrad
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace optiling;

class TabulateFusionGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TabulateFusionGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TabulateFusionGradTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
  return result;
}

static TeOpTensorArg SimpleTensorArg(const std::vector<int64_t>& shape, const std::string& dtype) {
  TeOpTensor tensor;
  tensor.shape = shape;
  tensor.dtype = dtype;

  TeOpTensorArg tensorArg;
  tensorArg.tensor.push_back(tensor);
  tensorArg.arg_type = TensorArgType::TA_SINGLE;

  return tensorArg;
}

static void RunTestTiling(const std::vector<int64_t>& tableShape, const std::string& tableDtype,
                          const std::vector<int64_t>& tableInfoShape, const std::string& tableInfoDtype,
                          const std::vector<int64_t>& emXShape, const std::string& emXDtype,
                          const std::vector<int64_t>& emShape, const std::string& emDtype,
                          const std::vector<int64_t>& dyShape, const std::string& dyDtype,
                          const std::vector<int64_t>& descriptorShape, const std::string& descriptorDtype,
                          const std::vector<int64_t>& dyDemXShape, const std::string& dyDemXDtype,
                          const std::vector<int64_t>& dyDemShape, const std::string& dyDemDtype,
                          const std::string& compileInfo, const std::string& compileInfoKey,
                          const std::string& expectTiling) {
  std::string opName = "TabulateFusionGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("TabulateFusionGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg table = SimpleTensorArg(tableShape, tableDtype);
  TeOpTensorArg tableInfo = SimpleTensorArg(tableInfoShape, tableInfoDtype);
  TeOpTensorArg emX = SimpleTensorArg(emXShape, emXDtype);
  TeOpTensorArg em = SimpleTensorArg(emShape, emDtype);
  TeOpTensorArg dy = SimpleTensorArg(dyShape, dyDtype);
  TeOpTensorArg descriptor = SimpleTensorArg(descriptorShape, descriptorDtype);

  TeOpTensorArg dyDemX = SimpleTensorArg(dyDemXShape, dyDemXDtype);
  TeOpTensorArg dyDem = SimpleTensorArg(dyDemShape, dyDemDtype);

  TeOpParas opParas;
  opParas.inputs.push_back(table);
  opParas.inputs.push_back(tableInfo);
  opParas.inputs.push_back(emX);
  opParas.inputs.push_back(em);
  opParas.inputs.push_back(dy);
  opParas.inputs.push_back(descriptor);
  opParas.outputs.push_back(dyDemX);
  opParas.outputs.push_back(dyDem);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo;
  opCompileInfo.key = compileInfoKey;
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), expectTiling);
}

void RunSimpleTest(const int64_t& runIndex, const int64_t& nloc, const int64_t& nnei, const int64_t& lastLayerSize,
                   const int64_t& splitCount, const int64_t& splitIndex, const int64_t& coreNum,
                   const std::string& expectTiling) {
  stringstream compileInfoKey;
  compileInfoKey << "tabulate_fusion_grad.key." << runIndex;

  stringstream compileInfo;
  compileInfo << "{\"vars\": {\"core_num\": " << coreNum
              << ", \"split_count\": " << splitCount
              << ", \"split_index\": " << splitIndex << "}}";

  RunTestTiling({1024, lastLayerSize * 6}, "float32",
                {6,}, "float32",
                {nloc, nnei}, "float32",
                {nloc, nnei, 4}, "float32",
                {nloc, 4, lastLayerSize}, "float32",
                {nloc, 4, lastLayerSize}, "float32",
                {nloc, nnei}, "float32",
                {nloc, nnei, 4}, "float32",
                compileInfo.str(), compileInfoKey.str(), expectTiling);
}

TEST_F(TabulateFusionGradTiling, tabulate_fusion_grad_tiling_001) {
  RunSimpleTest(10, 12288, 138, 128, 1, 0, 8, "12288 138 128 0 12288 0 8 1536 1536 ");
  RunSimpleTest(11, 12288, 138, 128, 2, 0, 8, "12288 138 128 0 6144 0 8 768 768 ");
  RunSimpleTest(12, 12288, 138, 128, 2, 1, 8, "12288 138 128 6144 6144 0 8 768 768 ");

  RunSimpleTest(20, 6144, 1800, 128, 1, 0, 8, "6144 1800 128 0 6144 0 8 768 768 ");
  RunSimpleTest(21, 6144, 1800, 128, 2, 0, 8, "6144 1800 128 0 3072 0 8 384 384 ");
  RunSimpleTest(22, 6144, 1800, 128, 2, 1, 8, "6144 1800 128 3072 3072 0 8 384 384 ");
}
