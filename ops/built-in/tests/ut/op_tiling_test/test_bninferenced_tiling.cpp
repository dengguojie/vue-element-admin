#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_batch_norm_ops.h"
#include "array_ops.h"

using namespace std;

class BNInferenceDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNInferenceDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNInferenceDTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {

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

TEST_F(BNInferenceDTiling, BNInferenceDTiling_test_1) {

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BNInferenceD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::BNInferenceD("BNInferenceD");

  vector<vector<int64_t>> input_shapes = {
      {2, 8, 2, 2},
      {8},
      {8},
      {1},
      {8}
  };

  vector<ge::DataType> dtypes = {
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16
  };
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, mean, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, variance, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, scale, input_shapes[3], dtypes[3], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, input_shapes[4], dtypes[4], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});

  std::string compileInfo = R"({"broadcast_mean_shape": [1, 8, 1, 1], "push_status": 0, "_fusion_index": [[0], [1], [2], [4]], "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_outs_uint1": false, "_base_info": {"000": [2, 4, 15872, 7936]}, "_elewise_vars": {"0": [10000, 10200], "1": [10000, 10200, 20000, 30000], "2": [10000, 10200, 20000, 30001], "3": [10000, 10200, 20000, 30002], "4": [10000, 10200, 20000, 30003], "6": [10000, 10200, 20001, 30001], "7": [10000, 10200, 20001, 30002], "8": [10000, 10200, 20001, 30003], "11": [10000, 10200, 20002, 30002], "12": [10000, 10200, 20002, 30003], "16": [10000, 10200, 20003, 30003]}, "_vars": {"0": ["_dim_0_0", "_dim_2_0"], "1": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_3"], "6": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "7": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "8": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_3"], "11": ["_dim_0_0", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "12": ["_dim_0_0", "_dim_2_0", "_block_factor_2", "_ub_factor_3"], "16": ["_dim_0_0", "_dim_2_0", "_block_factor_3", "_ub_factor_3"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 ");
}

TEST_F(BNInferenceDTiling, BNInferenceDTiling_test_2) {

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BNInferenceD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::BNInferenceD("BNInferenceD");

  vector<vector<int64_t>> input_shapes = {
      {2, 8, 2, 2},
      {8},
      {8},
      {1},
      {8}
  };

  vector<ge::DataType> dtypes = {
    ge::DT_FLOAT,
    ge::DT_FLOAT,
    ge::DT_FLOAT,
    ge::DT_FLOAT,
    ge::DT_FLOAT
  };
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, mean, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, variance, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, scale, input_shapes[3], dtypes[3], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, b, input_shapes[4], dtypes[4], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT, ge::FORMAT_ND, {});

  std::string compileInfo = R"({"broadcast_mean_shape": [1, 8, 1, 1], "push_status": 0, "_fusion_index": [[0], [1], [2], [4]], "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_outs_uint1": false, "_base_info": {"000": [2, 4, 15872, 7936]}, "_elewise_vars": {"0": [10000, 10200], "1": [10000, 10200, 20000, 30000], "2": [10000, 10200, 20000, 30001], "3": [10000, 10200, 20000, 30002], "4": [10000, 10200, 20000, 30003], "6": [10000, 10200, 20001, 30001], "7": [10000, 10200, 20001, 30002], "8": [10000, 10200, 20001, 30003], "11": [10000, 10200, 20002, 30002], "12": [10000, 10200, 20002, 30003], "16": [10000, 10200, 20003, 30003]}, "_vars": {"0": ["_dim_0_0", "_dim_2_0"], "1": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_3"], "6": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "7": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "8": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_3"], "11": ["_dim_0_0", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "12": ["_dim_0_0", "_dim_2_0", "_block_factor_2", "_ub_factor_3"], "16": ["_dim_0_0", "_dim_2_0", "_block_factor_3", "_ub_factor_3"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 ");
}
