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
 * \file test_prod_virial_se_a_tiling.cpp
 * \brief dynamic tiling test of ProdVirialSeA
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace optiling;

class ProdVirialSeATiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ProdVirialSeATiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ProdVirialSeATiling TearDown" << std::endl;
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

static void RunTestTiling(const std::vector<int64_t>& netShape, const std::string& netDtype,
                          const std::vector<int64_t>& inShape, const std::string& inDtype,
                          const std::vector<int64_t>& rijShape, const std::string& rijDtype,
                          const std::vector<int64_t>& nlistShape, const std::string& nlistDtype,
                          const std::vector<int64_t>& natomsShape, const std::string& natomsDtype,
                          const std::vector<int64_t>& virialShape, const std::string& virialDtype,
                          const std::vector<int64_t>& atomVirialShape, const std::string& atomVirialDtype,
                          const std::string& compileInfo, const std::string& compileInfoKey,
                          const std::string& expectTiling) {
  std::string opName = "ProdVirialSeA";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ProdVirialSeA");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg netDeriv = SimpleTensorArg(netShape, netDtype);
  TeOpTensorArg inDeriv = SimpleTensorArg(inShape, inDtype);
  TeOpTensorArg rij = SimpleTensorArg(rijShape, rijDtype);
  TeOpTensorArg nlist = SimpleTensorArg(nlistShape, nlistDtype);
  TeOpTensorArg natoms = SimpleTensorArg(natomsShape, natomsDtype);

  TeOpTensorArg virial = SimpleTensorArg(virialShape, virialDtype);
  TeOpTensorArg atomVirial = SimpleTensorArg(atomVirialShape, atomVirialDtype);

  TeOpParas opParas;
  opParas.inputs.push_back(netDeriv);
  opParas.inputs.push_back(inDeriv);
  opParas.inputs.push_back(rij);
  opParas.inputs.push_back(nlist);
  opParas.inputs.push_back(natoms);
  opParas.outputs.push_back(virial);
  opParas.outputs.push_back(atomVirial);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo;
  opCompileInfo.key = compileInfoKey;
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), expectTiling);
}

void RunSimpleTest(const int64_t& runIndex, const int64_t& nframes, const int64_t& nloc, const int64_t& nnei,
                   const int64_t& nall, const int64_t& splitCount, const int64_t& splitIndex,
                   const int64_t& coreNum, const std::string& expectTiling) {
  stringstream compileInfoKey;
  compileInfoKey << "prod_virial_se_a.key." << runIndex;

  stringstream compileInfo;
  compileInfo << "{\"vars\": {\"core_num\": " << coreNum
              << ", \"split_count\": " << splitCount
              << ", \"split_index\": " << splitIndex << "}}";

  RunTestTiling({nframes, nloc * nnei * 4}, "float32",
                {nframes, nloc * nnei * 4 * 3}, "float32",
                {nframes, nloc * nnei * 3}, "float32",
                {nframes, nloc * nnei}, "int32",
                {4,}, "int32",
                {nframes, 9}, "float32",
                {nframes, nall * 9}, "float32",
                compileInfo.str(), compileInfoKey.str(), expectTiling);
}

TEST_F(ProdVirialSeATiling, prod_virial_se_a_tiling_001) {
  // std::map<std::string, std::string> tilings = {
  //     "1 1 1 1 1 1 ",     "1 1 1 1 2 2 ",     "1 1 1 1 3 3 ",     "1 1 1 1 4 4 ",     "1 1 1 1 5 5 ",
  //     "1 1 1 1 6 6 ",     "1 1 1 1 7 7 ",     "1 1 1 1 8 8 ",     "2 1 1 1 5 9 ",     "2 2 1 1 5 10 ",
  //     "2 1 1 1 6 11 ",    "2 2 1 1 6 12 ",    "2 1 1 1 7 13 ",    "2 2 1 1 7 14 ",    "2 1 1 1 8 15 ",
  //     "2 2 1 1 8 16 ",    "3 2 1 1 6 17 ",    "3 3 1 1 6 18 ",    "3 1 1 1 7 19 ",    "3 2 1 1 7 20 ",
  //     "3 3 1 1 7 21 ",    "3 1 1 1 8 22 ",    "3 2 1 1 8 23 ",    "3 3 1 1 8 24 ",    "4 1 1 1 7 25 ",
  // };
  // size_t testIdx = 0
  // for (size_t nframes = 0; nframes < 5; ++nframes) {
  //   for (size_t nloc = 0; nloc < 64; ++nloc) {
  //     for (size_t nASel = 0; nASel < 64; ++nASel) {
  //       for (size_t nRSel = 0; nRSel < 1; ++nRSel) {
  //         size_t nnei = nASel + nRSel;
  //         for (size_t nall = 0; nall < 64; ++nall) {
  //           RunSimpleTest(testIdx, nframes, nloc, nnei, nall, 1, 0, 8, tilings[testIdx]);
  //           RunSimpleTest(testIdx, nframes, nloc, nnei, nall, 2, 0, 8, tilings[testIdx]);
  //           RunSimpleTest(testIdx, nframes, nloc, nnei, nall, 2, 1, 7, tilings[testIdx]);
  //         }
  //       }
  //     }
  //   }
  // }

  RunSimpleTest(0, 1, 12288, 138, 28328, 1, 0, 8, "1695744 28328 0 6624 0 8 828 828 ");
  RunSimpleTest(1, 1, 12288, 138, 28328, 2, 0, 8, "1695744 28328 0 3536 0 8 442 442 ");
  RunSimpleTest(2, 1, 12288, 138, 28328, 2, 1, 7, "1695744 28328 3536 3088 1 6 442 441 ");

  RunSimpleTest(10, 1, 6144, 1800, 19000, 1, 0, 8, "11059200 19000 0 43200 0 8 5400 5400 ");
  RunSimpleTest(11, 1, 6144, 1800, 19000, 2, 0, 8, "11059200 19000 0 23040 0 8 2880 2880 ");
  RunSimpleTest(12, 1, 6144, 1800, 19000, 2, 1, 7, "11059200 19000 23040 20160 0 7 2880 2880 ");
}
