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
 * \file test_prod_env_mat_a_calc_descrpt_tiling.cpp
 * \brief dynamic tiling test of ProdEnvMatACalcDescrpt
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace optiling;

class ProdEnvMatACalcDescrptTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdEnvMatACalcDescrptTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdEnvMatACalcDescrptTiling TearDown" << std::endl;
  }
};

static string ProdEnvMatCalcDescrptTilingToString(const std::stringstream& tiling_data) {
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

void ProdEnvMatCalcDescrptTilingSimpleTest(const int64_t& runIndex, const int64_t& nsample, const int64_t& nloc,
                                           const int64_t& nnei, const int64_t& coreNum,
                                           const std::string& expectTiling) {
  stringstream compileInfoKey;
  compileInfoKey << "prod_env_mat_a_calc_descrpt.key." << runIndex;

  stringstream compileInfo;
  compileInfo << "{\"vars\": {\"core_num\": " << coreNum << "}}";

  std::string opName = "ProdEnvMatACalcDescrpt";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ProdEnvMatACalcDescrpt");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg distance;
  TeOpTensorArg rijX;
  TeOpTensorArg rijY;
  TeOpTensorArg rijZ;
  TeOpTensorArg type;
  TeOpTensorArg natoms;
  TeOpTensorArg mesh;
  TeOpTensorArg davg;
  TeOpTensorArg dstd;

  TeOpTensorArg descrpt;
  TeOpTensorArg descrptDeriv;

  TeOpParas opParas;
  opParas.op_type = opName;
  opParas.inputs.push_back(distance);
  opParas.inputs.push_back(rijX);
  opParas.inputs.push_back(rijY);
  opParas.inputs.push_back(rijZ);
  opParas.inputs.push_back(type);
  opParas.inputs.push_back(natoms);
  opParas.inputs.push_back(mesh);
  opParas.inputs.push_back(davg);
  opParas.inputs.push_back(dstd);
  opParas.outputs.push_back(descrpt);
  opParas.outputs.push_back(descrptDeriv);

  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo.str();
  opCompileInfo.key = compileInfoKey.str();
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(ProdEnvMatCalcDescrptTilingToString(runInfo.tiling_data), expectTiling);
}

TEST_F(ProdEnvMatACalcDescrptTiling, prod_env_mat_a_calc_descrpt_tiling_001) {

  ProdEnvMatCalcDescrptTilingSimpleTest(0, 1, 12288, 138, 8, "");
}
