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
 * \file test_gen_adc_tiling.cpp
 * \brief dynamic tiling test of GenADC
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace optiling;

class GenADCTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GenADCTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GenADCTiling TearDown" << std::endl;
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

static void RunTestTiling(const std::vector<int64_t>& queryShape, const std::string& queryDtype,
                          const std::vector<int64_t>& codeBookShape, const std::string& codeBookDtype,
                          const std::vector<int64_t>& centroidsShape, const std::string& centroidsDtype,
                          const std::vector<int64_t>& bucketListShape, const std::string& bucketListDtype,
                          const std::vector<int64_t>& adcTablesShape, const std::string& adcTablesDtype,
                          const std::string& compileInfo, const std::string& compileInfoKey,
                          const std::string& expectTiling) {
  std::string opName = "GenADC";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GenADC");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensor queryTensor;
  queryTensor.shape = queryShape;
  queryTensor.dtype = queryDtype;

  TeOpTensor codeBookTensor;
  codeBookTensor.shape = codeBookShape;
  codeBookTensor.dtype = codeBookDtype;

  TeOpTensor centroidsTensor;
  centroidsTensor.shape = centroidsShape;
  centroidsTensor.dtype = centroidsDtype;

  TeOpTensor bucketListTensor;
  bucketListTensor.shape = bucketListShape;
  bucketListTensor.dtype = bucketListDtype;

  TeOpTensor adcTablesTensor;
  adcTablesTensor.shape = adcTablesShape;
  adcTablesTensor.dtype = adcTablesDtype;

  TeOpTensorArg tensorArgQuery;
  tensorArgQuery.tensor.push_back(queryTensor);
  tensorArgQuery.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCodeBook;
  tensorArgCodeBook.tensor.push_back(codeBookTensor);
  tensorArgCodeBook.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCentroids;
  tensorArgCentroids.tensor.push_back(centroidsTensor);
  tensorArgCentroids.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgBucketList;
  tensorArgBucketList.tensor.push_back(bucketListTensor);
  tensorArgBucketList.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgAdcTables;
  tensorArgAdcTables.tensor.push_back(adcTablesTensor);
  tensorArgAdcTables.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorArgQuery);
  opParas.inputs.push_back(tensorArgCodeBook);
  opParas.inputs.push_back(tensorArgCentroids);
  opParas.inputs.push_back(tensorArgBucketList);
  opParas.outputs.push_back(tensorArgAdcTables);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo;
  opCompileInfo.key = compileInfoKey;
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), expectTiling);
}

void RunSimpleTest(const std::string& bucketListDtype, const int64_t& bucketsNum, const std::string& expectTiling) {
  stringstream compileInfoKey;
  compileInfoKey << "gen_adc.key." << bucketListDtype << "." << bucketsNum;

  RunTestTiling({32}, "float16", {16, 256, 2}, "float16", {1000000, 32}, "float16", {bucketsNum}, bucketListDtype,
                {bucketsNum, 16, 256}, "float16", "{\"vars\": {\"core_num\": 8}}", compileInfoKey.str(), expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_001) {
  std::vector<std::string> tilings = {
      "1 1 1 1 1 1 ",     "1 1 1 1 2 2 ",     "1 1 1 1 3 3 ",     "1 1 1 1 4 4 ",     "1 1 1 1 5 5 ",
      "1 1 1 1 6 6 ",     "1 1 1 1 7 7 ",     "1 1 1 1 8 8 ",     "2 1 1 1 5 9 ",     "2 2 1 1 5 10 ",
      "2 1 1 1 6 11 ",    "2 2 1 1 6 12 ",    "2 1 1 1 7 13 ",    "2 2 1 1 7 14 ",    "2 1 1 1 8 15 ",
      "2 2 1 1 8 16 ",    "3 2 1 1 6 17 ",    "3 3 1 1 6 18 ",    "3 1 1 1 7 19 ",    "3 2 1 1 7 20 ",
      "3 3 1 1 7 21 ",    "3 1 1 1 8 22 ",    "3 2 1 1 8 23 ",    "3 3 1 1 8 24 ",    "4 1 1 1 7 25 ",
      "4 2 1 1 7 26 ",    "4 3 1 1 7 27 ",    "4 4 1 1 7 28 ",    "4 1 1 1 8 29 ",    "4 2 1 1 8 30 ",
      "4 3 1 1 8 31 ",    "4 4 1 1 8 32 ",    "5 3 1 1 7 33 ",    "5 4 1 1 7 34 ",    "5 5 1 1 7 35 ",
      "5 1 1 1 8 36 ",    "5 2 1 1 8 37 ",    "5 3 1 1 8 38 ",    "5 4 1 1 8 39 ",    "5 5 1 1 8 40 ",
      "6 5 1 1 7 41 ",    "6 6 1 1 7 42 ",    "6 1 1 1 8 43 ",    "6 2 1 1 8 44 ",    "6 3 1 1 8 45 ",
      "6 4 1 1 8 46 ",    "6 5 1 1 8 47 ",    "6 6 1 1 8 48 ",    "7 7 1 1 7 49 ",    "7 1 1 1 8 50 ",
      "7 2 1 1 8 51 ",    "7 3 1 1 8 52 ",    "7 4 1 1 8 53 ",    "7 5 1 1 8 54 ",    "7 6 1 1 8 55 ",
      "7 7 1 1 8 56 ",    "8 1 1 1 8 57 ",    "8 2 1 1 8 58 ",    "8 3 1 1 8 59 ",    "8 4 1 1 8 60 ",
      "8 5 1 1 8 61 ",    "8 6 1 1 8 62 ",    "8 7 1 1 8 63 ",    "8 8 1 1 8 64 ",    "9 2 2 1 8 65 ",
      "9 3 2 1 8 66 ",    "9 4 2 1 8 67 ",    "9 5 2 1 8 68 ",    "9 6 2 1 8 69 ",    "9 7 2 1 8 70 ",
      "9 8 2 1 8 71 ",    "9 9 2 2 8 72 ",    "10 3 2 1 8 73 ",   "10 4 2 1 8 74 ",   "10 5 2 1 8 75 ",
      "10 6 2 1 8 76 ",   "10 7 2 1 8 77 ",   "10 8 2 1 8 78 ",   "10 9 2 2 8 79 ",   "10 10 2 2 8 80 ",
      "11 4 2 1 8 81 ",   "11 5 2 1 8 82 ",   "11 6 2 1 8 83 ",   "11 7 2 1 8 84 ",   "11 8 2 1 8 85 ",
      "11 9 2 2 8 86 ",   "11 10 2 2 8 87 ",  "11 11 2 2 8 88 ",  "12 5 2 1 8 89 ",   "12 6 2 1 8 90 ",
      "12 7 2 1 8 91 ",   "12 8 2 1 8 92 ",   "12 9 2 2 8 93 ",   "12 10 2 2 8 94 ",  "12 11 2 2 8 95 ",
      "12 12 2 2 8 96 ",  "13 6 2 1 8 97 ",   "13 7 2 1 8 98 ",   "13 8 2 1 8 99 ",   "13 9 2 2 8 100 ",
      "13 10 2 2 8 101 ", "13 11 2 2 8 102 ", "13 12 2 2 8 103 ", "13 13 2 2 8 104 ", "14 7 2 1 8 105 ",
      "14 8 2 1 8 106 ",  "14 9 2 2 8 107 ",  "14 10 2 2 8 108 ", "14 11 2 2 8 109 ", "14 12 2 2 8 110 ",
      "14 13 2 2 8 111 ", "14 14 2 2 8 112 ", "15 8 2 1 8 113 ",  "15 9 2 2 8 114 ",  "15 10 2 2 8 115 ",
      "15 11 2 2 8 116 ", "15 12 2 2 8 117 ", "15 13 2 2 8 118 ", "15 14 2 2 8 119 ", "15 15 2 2 8 120 ",
      "16 9 2 2 8 121 ",  "16 10 2 2 8 122 ", "16 11 2 2 8 123 ", "16 12 2 2 8 124 ", "16 13 2 2 8 125 ",
      "16 14 2 2 8 126 ", "16 15 2 2 8 127 ", "16 16 2 2 8 128 ", "17 10 3 2 8 129 ", "17 11 3 2 8 130 ",
      "17 12 3 2 8 131 ", "17 13 3 2 8 132 ", "17 14 3 2 8 133 ", "17 15 3 2 8 134 ", "17 16 3 2 8 135 ",
      "17 17 3 3 8 136 ", "18 11 3 2 8 137 ", "18 12 3 2 8 138 ", "18 13 3 2 8 139 ", "18 14 3 2 8 140 ",
      "18 15 3 2 8 141 ", "18 16 3 2 8 142 ", "18 17 3 3 8 143 ", "18 18 3 3 8 144 ", "19 12 3 2 8 145 ",
      "19 13 3 2 8 146 ", "19 14 3 2 8 147 ", "19 15 3 2 8 148 ", "19 16 3 2 8 149 ", "19 17 3 3 8 150 ",
      "19 18 3 3 8 151 ", "19 19 3 3 8 152 ", "20 13 3 2 8 153 ", "20 14 3 2 8 154 ", "20 15 3 2 8 155 ",
      "20 16 3 2 8 156 ", "20 17 3 3 8 157 ", "20 18 3 3 8 158 ", "20 19 3 3 8 159 ", "20 20 3 3 8 160 ",
      "21 14 3 2 8 161 ", "21 15 3 2 8 162 ", "21 16 3 2 8 163 ", "21 17 3 3 8 164 ", "21 18 3 3 8 165 ",
      "21 19 3 3 8 166 ", "21 20 3 3 8 167 ", "21 21 3 3 8 168 ", "22 15 3 2 8 169 ", "22 16 3 2 8 170 ",
      "22 17 3 3 8 171 ", "22 18 3 3 8 172 ", "22 19 3 3 8 173 ", "22 20 3 3 8 174 ", "22 21 3 3 8 175 ",
      "22 22 3 3 8 176 ", "23 16 3 2 8 177 ", "23 17 3 3 8 178 ", "23 18 3 3 8 179 ", "23 19 3 3 8 180 ",
      "23 20 3 3 8 181 ", "23 21 3 3 8 182 ", "23 22 3 3 8 183 ", "23 23 3 3 8 184 ", "24 17 3 3 8 185 ",
      "24 18 3 3 8 186 ", "24 19 3 3 8 187 ", "24 20 3 3 8 188 ", "24 21 3 3 8 189 ", "24 22 3 3 8 190 ",
      "24 23 3 3 8 191 ", "24 24 3 3 8 192 ", "25 18 4 3 8 193 ", "25 19 4 3 8 194 ", "25 20 4 3 8 195 ",
      "25 21 4 3 8 196 ", "25 22 4 3 8 197 ", "25 23 4 3 8 198 ", "25 24 4 3 8 199 ", "25 25 4 4 8 200 ",
  };

  for (int i = 1; i <= tilings.size(); ++i) {
    RunSimpleTest("int32", i, tilings[i - 1]);
  }
}

TEST_F(GenADCTiling, gen_adc_tiling_002) {
  std::vector<std::string> tilings = {
      "1 1 1 1 1 1 ",     "1 1 1 1 2 2 ",     "1 1 1 1 3 3 ",     "1 1 1 1 4 4 ",     "1 1 1 1 5 5 ",
      "1 1 1 1 6 6 ",     "1 1 1 1 7 7 ",     "1 1 1 1 8 8 ",     "2 1 1 1 5 9 ",     "2 2 1 1 5 10 ",
      "2 1 1 1 6 11 ",    "2 2 1 1 6 12 ",    "2 1 1 1 7 13 ",    "2 2 1 1 7 14 ",    "2 1 1 1 8 15 ",
      "2 2 1 1 8 16 ",    "3 2 1 1 6 17 ",    "3 3 1 1 6 18 ",    "3 1 1 1 7 19 ",    "3 2 1 1 7 20 ",
      "3 3 1 1 7 21 ",    "3 1 1 1 8 22 ",    "3 2 1 1 8 23 ",    "3 3 1 1 8 24 ",    "4 1 1 1 7 25 ",
      "4 2 1 1 7 26 ",    "4 3 1 1 7 27 ",    "4 4 1 1 7 28 ",    "4 1 1 1 8 29 ",    "4 2 1 1 8 30 ",
      "4 3 1 1 8 31 ",    "4 4 1 1 8 32 ",    "5 3 2 1 7 33 ",    "5 4 2 1 7 34 ",    "5 5 2 2 7 35 ",
      "5 1 2 1 8 36 ",    "5 2 2 1 8 37 ",    "5 3 2 1 8 38 ",    "5 4 2 1 8 39 ",    "5 5 2 2 8 40 ",
      "6 5 2 2 7 41 ",    "6 6 2 2 7 42 ",    "6 1 2 1 8 43 ",    "6 2 2 1 8 44 ",    "6 3 2 1 8 45 ",
      "6 4 2 1 8 46 ",    "6 5 2 2 8 47 ",    "6 6 2 2 8 48 ",    "7 7 2 2 7 49 ",    "7 1 2 1 8 50 ",
      "7 2 2 1 8 51 ",    "7 3 2 1 8 52 ",    "7 4 2 1 8 53 ",    "7 5 2 2 8 54 ",    "7 6 2 2 8 55 ",
      "7 7 2 2 8 56 ",    "8 1 2 1 8 57 ",    "8 2 2 1 8 58 ",    "8 3 2 1 8 59 ",    "8 4 2 1 8 60 ",
      "8 5 2 2 8 61 ",    "8 6 2 2 8 62 ",    "8 7 2 2 8 63 ",    "8 8 2 2 8 64 ",    "9 2 3 1 8 65 ",
      "9 3 3 1 8 66 ",    "9 4 3 1 8 67 ",    "9 5 3 2 8 68 ",    "9 6 3 2 8 69 ",    "9 7 3 2 8 70 ",
      "9 8 3 2 8 71 ",    "9 9 3 3 8 72 ",    "10 3 3 1 8 73 ",   "10 4 3 1 8 74 ",   "10 5 3 2 8 75 ",
      "10 6 3 2 8 76 ",   "10 7 3 2 8 77 ",   "10 8 3 2 8 78 ",   "10 9 3 3 8 79 ",   "10 10 3 3 8 80 ",
      "11 4 3 1 8 81 ",   "11 5 3 2 8 82 ",   "11 6 3 2 8 83 ",   "11 7 3 2 8 84 ",   "11 8 3 2 8 85 ",
      "11 9 3 3 8 86 ",   "11 10 3 3 8 87 ",  "11 11 3 3 8 88 ",  "12 5 3 2 8 89 ",   "12 6 3 2 8 90 ",
      "12 7 3 2 8 91 ",   "12 8 3 2 8 92 ",   "12 9 3 3 8 93 ",   "12 10 3 3 8 94 ",  "12 11 3 3 8 95 ",
      "12 12 3 3 8 96 ",  "13 6 4 2 8 97 ",   "13 7 4 2 8 98 ",   "13 8 4 2 8 99 ",   "13 9 4 3 8 100 ",
      "13 10 4 3 8 101 ", "13 11 4 3 8 102 ", "13 12 4 3 8 103 ", "13 13 4 4 8 104 ", "14 7 4 2 8 105 ",
      "14 8 4 2 8 106 ",  "14 9 4 3 8 107 ",  "14 10 4 3 8 108 ", "14 11 4 3 8 109 ", "14 12 4 3 8 110 ",
      "14 13 4 4 8 111 ", "14 14 4 4 8 112 ", "15 8 4 2 8 113 ",  "15 9 4 3 8 114 ",  "15 10 4 3 8 115 ",
      "15 11 4 3 8 116 ", "15 12 4 3 8 117 ", "15 13 4 4 8 118 ", "15 14 4 4 8 119 ", "15 15 4 4 8 120 ",
      "16 9 4 3 8 121 ",  "16 10 4 3 8 122 ", "16 11 4 3 8 123 ", "16 12 4 3 8 124 ", "16 13 4 4 8 125 ",
      "16 14 4 4 8 126 ", "16 15 4 4 8 127 ", "16 16 4 4 8 128 ", "17 10 5 3 8 129 ", "17 11 5 3 8 130 ",
      "17 12 5 3 8 131 ", "17 13 5 4 8 132 ", "17 14 5 4 8 133 ", "17 15 5 4 8 134 ", "17 16 5 4 8 135 ",
      "17 17 5 5 8 136 ", "18 11 5 3 8 137 ", "18 12 5 3 8 138 ", "18 13 5 4 8 139 ", "18 14 5 4 8 140 ",
      "18 15 5 4 8 141 ", "18 16 5 4 8 142 ", "18 17 5 5 8 143 ", "18 18 5 5 8 144 ", "19 12 5 3 8 145 ",
      "19 13 5 4 8 146 ", "19 14 5 4 8 147 ", "19 15 5 4 8 148 ", "19 16 5 4 8 149 ", "19 17 5 5 8 150 ",
      "19 18 5 5 8 151 ", "19 19 5 5 8 152 ", "20 13 5 4 8 153 ", "20 14 5 4 8 154 ", "20 15 5 4 8 155 ",
      "20 16 5 4 8 156 ", "20 17 5 5 8 157 ", "20 18 5 5 8 158 ", "20 19 5 5 8 159 ", "20 20 5 5 8 160 ",
      "21 14 6 4 8 161 ", "21 15 6 4 8 162 ", "21 16 6 4 8 163 ", "21 17 6 5 8 164 ", "21 18 6 5 8 165 ",
      "21 19 6 5 8 166 ", "21 20 6 5 8 167 ", "21 21 6 6 8 168 ", "22 15 6 4 8 169 ", "22 16 6 4 8 170 ",
      "22 17 6 5 8 171 ", "22 18 6 5 8 172 ", "22 19 6 5 8 173 ", "22 20 6 5 8 174 ", "22 21 6 6 8 175 ",
      "22 22 6 6 8 176 ", "23 16 6 4 8 177 ", "23 17 6 5 8 178 ", "23 18 6 5 8 179 ", "23 19 6 5 8 180 ",
      "23 20 6 5 8 181 ", "23 21 6 6 8 182 ", "23 22 6 6 8 183 ", "23 23 6 6 8 184 ", "24 17 6 5 8 185 ",
      "24 18 6 5 8 186 ", "24 19 6 5 8 187 ", "24 20 6 5 8 188 ", "24 21 6 6 8 189 ", "24 22 6 6 8 190 ",
      "24 23 6 6 8 191 ", "24 24 6 6 8 192 ", "25 18 7 5 8 193 ", "25 19 7 5 8 194 ", "25 20 7 5 8 195 ",
      "25 21 7 6 8 196 ", "25 22 7 6 8 197 ", "25 23 7 6 8 198 ", "25 24 7 6 8 199 ", "25 25 7 7 8 200 ",
  };

  for (int i = 1; i <= tilings.size(); ++i) {
    RunSimpleTest("int64", i, tilings[i - 1]);
  }
}

// Test case: missing bucket_list tensor
TEST_F(GenADCTiling, gen_adc_tiling_101) {
  std::string opName = "GenADC";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GenADC");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensor queryTensor;
  queryTensor.shape = {32};
  queryTensor.dtype = "float16";

  TeOpTensor codeBookTensor;
  codeBookTensor.shape = {16, 256, 2};
  codeBookTensor.dtype = "float16";

  TeOpTensor centroidsTensor;
  centroidsTensor.shape = {1000000, 32};
  centroidsTensor.dtype = "float16";

  TeOpTensor bucketListTensor;
  bucketListTensor.shape = {1024};
  bucketListTensor.dtype = "int32";

  TeOpTensor adcTablesTensor;
  adcTablesTensor.shape = {1024, 16, 256};
  adcTablesTensor.dtype = "float16";

  TeOpTensorArg tensorArgQuery;
  tensorArgQuery.tensor.push_back(queryTensor);
  tensorArgQuery.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCodeBook;
  tensorArgCodeBook.tensor.push_back(codeBookTensor);
  tensorArgCodeBook.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCentroids;
  tensorArgCentroids.tensor.push_back(centroidsTensor);
  tensorArgCentroids.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgBucketList;
  tensorArgBucketList.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgAdcTables;
  tensorArgAdcTables.tensor.push_back(adcTablesTensor);
  tensorArgAdcTables.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorArgQuery);
  opParas.inputs.push_back(tensorArgCodeBook);
  opParas.inputs.push_back(tensorArgCentroids);
  opParas.inputs.push_back(tensorArgBucketList);
  opParas.outputs.push_back(tensorArgAdcTables);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = "{\"vars\": {\"core_num\": 8}}";
  opCompileInfo.key = "gen_adc.key.101";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
}

// Test case: bucket_list shape error
TEST_F(GenADCTiling, gen_adc_tiling_102) {
  std::string opName = "GenADC";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GenADC");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensor queryTensor;
  queryTensor.shape = {32};
  queryTensor.dtype = "float16";

  TeOpTensor codeBookTensor;
  codeBookTensor.shape = {16, 256, 2};
  codeBookTensor.dtype = "float16";

  TeOpTensor centroidsTensor;
  centroidsTensor.shape = {1000000, 32};
  centroidsTensor.dtype = "float16";

  TeOpTensor bucketListTensor;
  bucketListTensor.shape = {-1};
  bucketListTensor.dtype = "int32";

  TeOpTensor adcTablesTensor;
  adcTablesTensor.shape = {1024, 16, 256};
  adcTablesTensor.dtype = "float16";

  TeOpTensorArg tensorArgQuery;
  tensorArgQuery.tensor.push_back(queryTensor);
  tensorArgQuery.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCodeBook;
  tensorArgCodeBook.tensor.push_back(codeBookTensor);
  tensorArgCodeBook.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCentroids;
  tensorArgCentroids.tensor.push_back(centroidsTensor);
  tensorArgCentroids.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgBucketList;
  tensorArgBucketList.tensor.push_back(bucketListTensor);
  tensorArgBucketList.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgAdcTables;
  tensorArgAdcTables.tensor.push_back(adcTablesTensor);
  tensorArgAdcTables.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorArgQuery);
  opParas.inputs.push_back(tensorArgCodeBook);
  opParas.inputs.push_back(tensorArgCentroids);
  opParas.inputs.push_back(tensorArgBucketList);
  opParas.outputs.push_back(tensorArgAdcTables);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = "{\"vars\": {\"core_num\": 8}}";
  opCompileInfo.key = "gen_adc.key.102";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
}

// Test case: core num error
TEST_F(GenADCTiling, gen_adc_tiling_103) {
  std::string opName = "GenADC";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GenADC");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensor queryTensor;
  queryTensor.shape = {32};
  queryTensor.dtype = "float16";

  TeOpTensor codeBookTensor;
  codeBookTensor.shape = {16, 256, 2};
  codeBookTensor.dtype = "float16";

  TeOpTensor centroidsTensor;
  centroidsTensor.shape = {1000000, 32};
  centroidsTensor.dtype = "float16";

  TeOpTensor bucketListTensor;
  bucketListTensor.shape = {1024};
  bucketListTensor.dtype = "int32";

  TeOpTensor adcTablesTensor;
  adcTablesTensor.shape = {1024, 16, 256};
  adcTablesTensor.dtype = "float16";

  TeOpTensorArg tensorArgQuery;
  tensorArgQuery.tensor.push_back(queryTensor);
  tensorArgQuery.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCodeBook;
  tensorArgCodeBook.tensor.push_back(codeBookTensor);
  tensorArgCodeBook.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCentroids;
  tensorArgCentroids.tensor.push_back(centroidsTensor);
  tensorArgCentroids.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgBucketList;
  tensorArgBucketList.tensor.push_back(bucketListTensor);
  tensorArgBucketList.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgAdcTables;
  tensorArgAdcTables.tensor.push_back(adcTablesTensor);
  tensorArgAdcTables.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorArgQuery);
  opParas.inputs.push_back(tensorArgCodeBook);
  opParas.inputs.push_back(tensorArgCentroids);
  opParas.inputs.push_back(tensorArgBucketList);
  opParas.outputs.push_back(tensorArgAdcTables);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = "{\"vars\": {\"core_num\": -1}}";
  opCompileInfo.key = "gen_adc.key.103";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, opCompileInfo, runInfo));
}
