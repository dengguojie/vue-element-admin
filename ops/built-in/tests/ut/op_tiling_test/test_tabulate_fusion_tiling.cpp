/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file test_tabulate_fusion_tiling.cpp
 * \brief dynamic tiling test of TabulateFusion
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace optiling;

class TabulateFusionTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TabulateFusionTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TabulateFusionTiling TearDown" << std::endl;
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
  tensorArg.arg_type = TA_SINGLE;

  return tensorArg;
}

static void RunTestTiling(const std::vector<int64_t>& tableShape, const std::string& tableDtype,
                          const std::vector<int64_t>& tableInfoShape, const std::string& tableInfoDtype,
                          const std::vector<int64_t>& emXShape, const std::string& emXDtype,
                          const std::vector<int64_t>& emShape, const std::string& emDtype,
                          const std::vector<int64_t>& descriptorShape, const std::string& descriptorDtype,
                          const std::string& compileInfo, const std::string& compileInfoKey,
                          const std::string& expectTiling) {
  std::string opName = "TabulateFusion";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("TabulateFusion");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg table = SimpleTensorArg(tableShape, tableDtype);
  TeOpTensorArg table_info = SimpleTensorArg(tableInfoShape, tableInfoDtype);
  TeOpTensorArg em_x = SimpleTensorArg(emXShape, emXDtype);
  TeOpTensorArg em = SimpleTensorArg(emShape, emDtype);
  TeOpTensorArg descriptor = SimpleTensorArg(descriptorShape, descriptorDtype);

  TeOpParas opParas;
  opParas.inputs.push_back(table);
  opParas.inputs.push_back(table_info);
  opParas.inputs.push_back(em_x);
  opParas.inputs.push_back(em);
  opParas.outputs.push_back(descriptor);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo;
  opCompileInfo.key = compileInfoKey;
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), expectTiling);
}

void RunSimpleTest(const int64_t& runIndex, const int64_t& nloc, const int64_t& nnei, const int32_t& last_layer_size,
                   const int64_t& table_dim0, const int64_t& splitCount, const int64_t& splitIndex,
                   const int64_t& one_portion_ub_elems, const int64_t& coreNum, const std::string& expectTiling) {
  stringstream compileInfoKey;
  compileInfoKey << "tabulate_fusion.key." << runIndex;

  stringstream compileInfo;
  compileInfo << "{\"vars\": {\"core_num\": " << coreNum
              << ", \"last_layer_size\": " << last_layer_size
              << ", \"one_portion_ub_elems\": " << one_portion_ub_elems
              << ", \"split_count\": " << splitCount
              << ", \"split_index\": " << splitIndex << "}}";

  RunTestTiling({table_dim0, last_layer_size * 6}, "float32",
                {6}, "float32",
                {nloc, nnei}, "float32",
                {nloc, nnei, 4}, "int32",
                {nloc, 4, last_layer_size}, "float32",
                compileInfo.str(), compileInfoKey.str(), expectTiling);
}

TEST_F(TabulateFusionTiling, prod_tabulate_fusion_tiling_001) {
  RunSimpleTest(0, 8192, 46, 100, 1360, 1, 0, 14848, 8, "8 0 46 1024 1024 128 8 0 8 0 ");
  RunSimpleTest(1, 8192, 46, 100, 1360, 2, 0, 14848, 8, "8 0 46 512 512 128 4 0 4 0 ");
  RunSimpleTest(2, 8192, 46, 100, 1360, 2, 1, 14848, 7, "7 4096 46 586 580 128 4 74 4 68 ");
}
