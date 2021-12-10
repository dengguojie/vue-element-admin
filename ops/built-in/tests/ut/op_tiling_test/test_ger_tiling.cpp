#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "linalg_ops.h"
#include "array_ops.h"

using namespace std;

class GerTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GerTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GerTilingTest TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;
/*
.INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
*/

TEST_F(GerTilingTest, Ger_Tiling_Test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Ger("Ger");

  vector<vector<int64_t>> input_shapes = {{1}, {10}};

  vector<int64_t> output = {1, 10};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo =
      R"({"_fusion_index":[[0,1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"100":[32,4,21840,10920], "320":[32,4,21840,10920]}, "_vars":{"210000000": ["_block_factor_0","_ub_factor_0"], "210010000": ["_block_factor_0","_ub_factor_0"], "232000000": ["_dim_0_1","_block_factor_0","_ub_factor_0"]}, "_normal_vars": {"210000000": ["_block_factor_0","_ub_factor_0"], "210010000": ["_block_factor_0","_ub_factor_0"], "232000000": ["_dim_0_1","_block_factor_0","_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "232000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "232000000": []}, "_elewise_vars": {"210000000": [20000,30000], "210010000": [20000,30000], "232000000": [10001,20000,30000]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 10 10 ");
}

TEST_F(GerTilingTest, Ger_Tiling_Test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Ger("Ger");

  vector<vector<int64_t>> input_shapes = {{20}, {20}};

  vector<int64_t> output = {20, 20};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo =
      R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"320":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"232000000":[10001,20000,30000], "0":[10000], "1":[10000,20000,30000], "2":[10000,20000,30001], "4":[10000,20001,30001]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "20 ");
}

TEST_F(GerTilingTest, Ger_Tiling_Test_3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Ger("Ger");

  vector<vector<int64_t>> input_shapes = {{20}, {20}};

  vector<int64_t> output = {20, 20};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo =
      R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"100":[32,4,21840,10920], "320":[32,4,21840,10920], "230":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"210000000":[20000,30000], "210010000":[20000,30000], "232000000":[10001,20000,30000], "223000000":[10000,20000,30000], "0":[10000,10101], "1":[10000,10101,20000,30000], "2":[10000,10101,20000,30001], "4":[10000,10101,20001,30001]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "20 20 ");
}

TEST_F(GerTilingTest, Ger_Tiling_Test_4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Ger("Ger");

  vector<vector<int64_t>> input_shapes = {{1}, {1}};

  vector<int64_t> output = {1, 1};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo =
      R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"100":[32,4,21840,10920], "320":[32,4,21840,10920], "230":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"210000000":[20000,30000], "210010000":[20000,30000], "232000000":[10001,20000,30000], "223000000":[10000,20000,30000], "0":[10000,10101], "1":[10000,10101,20000,30000], "2":[10000,10101,20000,30001], "4":[10000,10101,20001,30001]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 ");
}

TEST_F(GerTilingTest, SmoothL1LossV2_Tiling_Test_5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Ger("Ger");

  vector<vector<int64_t>> input_shapes = {{1}, {10}};

  vector<int64_t> output = {1, 10};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo =
      R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"320":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"232000000":[10001,20000,30000], "0":[10000], "1":[10000,20000,30000], "2":[10000,20000,30001], "4":[10000,20001,30001]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 10 10 ");
}