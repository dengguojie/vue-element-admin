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
 * \file test_prod_env_mat_a_tiling.cpp
 * \brief dynamic tiling test of ProdEnvMatA
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace optiling;

class ProdEnvMatATiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdEnvMatATiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdEnvMatATiling TearDown" << std::endl;
  }
};

static string pro_to_string(const std::stringstream& tiling_data) {
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

static TeOpTensorArg ProdSimpleTensorArg(const std::vector<int64_t>& shape, const std::string& dtype) {
  TeOpTensor tensor;
  tensor.shape = shape;
  tensor.dtype = dtype;

  TeOpTensorArg tensorArg;
  tensorArg.tensor.push_back(tensor);
  tensorArg.arg_type = TensorArgType::TA_SINGLE;

  return tensorArg;
}

static void ProdRunTestTiling(const std::vector<int64_t>& coordShape, const std::string& coordDtype,
                          const std::vector<int64_t>& typeShape, const std::string& typeDtype,
                          const std::vector<int64_t>& natomsShape, const std::string& natomsDtype,
                          const std::vector<int64_t>& boxShape, const std::string& boxDtype,
                          const std::vector<int64_t>& meshShape, const std::string& meshDtype,
                          const std::vector<int64_t>& davgShape, const std::string& davgDtype,
                          const std::vector<int64_t>& dstdShape, const std::string& dstdDtype,
                          const std::vector<int64_t>& descrptShape, const std::string& descrptDtype,
                          const std::vector<int64_t>& descrptDerivShape, const std::string& descrptDerivDtype,
                          const std::vector<int64_t>& rijShape, const std::string& rijDtype,
                          const std::vector<int64_t>& nlistShape, const std::string& nlistDtype,
                          const std::string& compileInfo, const std::string& compileInfoKey,
                          const std::string& expectTiling) {
  std::string opName = "ProdEnvMatA";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ProdEnvMatA");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg coord = ProdSimpleTensorArg(coordShape, coordDtype);
  TeOpTensorArg type = ProdSimpleTensorArg(typeShape, typeDtype);
  TeOpTensorArg natoms = ProdSimpleTensorArg(natomsShape, natomsDtype);
  TeOpTensorArg box = ProdSimpleTensorArg(boxShape, boxDtype);
  TeOpTensorArg mesh = ProdSimpleTensorArg(meshShape, meshDtype);

  TeOpTensorArg davg = ProdSimpleTensorArg(davgShape, davgDtype);
  TeOpTensorArg dstd = ProdSimpleTensorArg(dstdShape, dstdDtype);

  TeOpTensorArg descrpt = ProdSimpleTensorArg(descrptShape, descrptDtype);
  TeOpTensorArg descrpt_deriv = ProdSimpleTensorArg(descrptDerivShape, descrptDerivDtype);
  TeOpTensorArg rij = ProdSimpleTensorArg(rijShape, rijDtype);
  TeOpTensorArg nlist = ProdSimpleTensorArg(nlistShape, nlistDtype);

  TeOpParas opParas;
  opParas.inputs.push_back(coord);
  opParas.inputs.push_back(type);
  opParas.inputs.push_back(natoms);
  opParas.inputs.push_back(box);
  opParas.inputs.push_back(mesh);
  opParas.inputs.push_back(davg);
  opParas.inputs.push_back(dstd);
  opParas.outputs.push_back(descrpt);
  opParas.outputs.push_back(descrpt_deriv);
  opParas.outputs.push_back(rij);
  opParas.outputs.push_back(nlist);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo;
  opCompileInfo.key = compileInfoKey;
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(pro_to_string(runInfo.tiling_data), expectTiling);
}

void ProRunSimpleTest(const int64_t& runIndex, const int64_t& nsample, const int64_t& nloc, const int64_t& nnei,
                      const int64_t& nall, const int64_t& splitCount, const int64_t& splitIndex,
                      const int64_t& coreNum, const std::string& expectTiling) {
  stringstream compileInfoKey;
  compileInfoKey << "prod_env_mat_a.key." << runIndex;

  stringstream compileInfo;
  compileInfo << "{\"vars\": {\"core_num\": " << coreNum
              << ", \"split_count\": " << splitCount
              << ", \"split_index\": " << splitIndex << "}}";

  ProdRunTestTiling({nsample, nall * 3}, "float32",
                    {nsample, nall}, "int32",
                    {4,}, "int32",
                    {nsample, 9}, "float32",
                    {1 + 1026 * nloc,}, "int32",
                    {2, nnei * 4}, "float32",
                    {2, nnei * 4}, "float32",
                    {nsample, nloc * nnei * 4}, "float32",
                    {nsample, nloc * nnei * 12}, "float32",
                    {nsample, nloc * nnei * 3}, "float32",
                    {nsample, nloc * nnei}, "int32",
                    compileInfo.str(), compileInfoKey.str(), expectTiling);
}

TEST_F(ProdEnvMatATiling, prod_env_mat_a_tiling_001) {

  ProRunSimpleTest(0, 1, 12288, 138, 28328, 1, 0, 8, "");
}
