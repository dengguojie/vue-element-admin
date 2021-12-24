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
 * \file test_prod_force_se_a_tiling.cpp
 * \brief dynamic tiling test of ProdForceSeA
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace optiling;

class ProdForceSeATiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdForceSeATiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdForceSeATiling TearDown" << std::endl;
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

static void RunTestTiling(const std::vector<int64_t>& netShape, const std::string& netDtype,
                          const std::vector<int64_t>& inShape, const std::string& inDtype,
                          const std::vector<int64_t>& nlistShape, const std::string& nlistDtype,
                          const std::vector<int64_t>& natomsShape, const std::string& natomsDtype,
                          const std::vector<int64_t>& forceShape, const std::string& forceDtype,
                          const std::string& compileInfo, const std::string& compileInfoKey,
                          const std::string& expectTiling) {
  std::string opName = "ProdForceSeA";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ProdForceSeA");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg netDeriv = SimpleTensorArg(netShape, netDtype);
  TeOpTensorArg inDeriv = SimpleTensorArg(inShape, inDtype);
  TeOpTensorArg nlist = SimpleTensorArg(nlistShape, nlistDtype);
  TeOpTensorArg natoms = SimpleTensorArg(natomsShape, natomsDtype);

  TeOpTensorArg force = SimpleTensorArg(forceShape, forceDtype);

  TeOpParas opParas;
  opParas.inputs.push_back(netDeriv);
  opParas.inputs.push_back(inDeriv);
  opParas.inputs.push_back(nlist);
  opParas.inputs.push_back(natoms);
  opParas.outputs.push_back(force);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo;
  opCompileInfo.key = compileInfoKey;
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), expectTiling);
}

void ForceRunSimpleTest(const int64_t& runIndex, const int64_t& nframes, const int64_t& nloc, const int64_t& nnei,
                   const int64_t& nall, const int64_t& splitCount, const int64_t& splitIndex,
                   const int64_t& coreNum, const std::string& expectTiling) {
  stringstream compileInfoKey;
  compileInfoKey << "prod_force_se_a.key." << runIndex;

  stringstream compileInfo;
  int64_t n_r_sel = 0;
  compileInfo << "{\"vars\": {\"core_nums\": " << coreNum
              << ", \"n_a_sel\": " << nnei
              << ", \"n_r_sel\": " << n_r_sel
              << ", \"split_count\": " << splitCount
              << ", \"split_index\": " << splitIndex << "}}";

  RunTestTiling({nframes, nloc * nnei * 4}, "float32",
                {nframes, nloc * nnei * 4 * 3}, "float32",
                {nframes, nloc * nnei}, "int32",
                {4,}, "int32",
                {nframes, 3, nall}, "float32",
                compileInfo.str(), compileInfoKey.str(), expectTiling);
}

TEST_F(ProdForceSeATiling, prod_force_se_a_tiling_001) {
  ForceRunSimpleTest(0, 1, 12288, 138, 28328, 1, 0, 8, "12288 28328 1536 0 0 1 8 ");
  ForceRunSimpleTest(1, 1, 12288, 138, 28328, 2, 0, 8, "12288 28328 820 0 0 1 8 ");
  ForceRunSimpleTest(2, 1, 12288, 138, 28328, 2, 1, 7, "12288 28328 820 808 6560 1 7 ");
}
TEST_F(ProdForceSeATiling, prod_force_se_a_tiling_002) {
  ForceRunSimpleTest(3, 1, 65, 8, 10000, 1, 0, 8, "65 10000 9 2 0 1 8 ");
  ForceRunSimpleTest(3, 1, 65, 8, 10000, 1, 0, 8, "65 10000 9 2 0 1 8 ");
}
