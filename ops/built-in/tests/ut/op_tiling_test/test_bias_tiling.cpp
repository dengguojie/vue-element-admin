#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class BiasTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BiasTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BiasTiling TearDown" << std::endl;
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

TEST_F(BiasTiling, Bias_tiling_test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Bias");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Bias("Bias");

  vector<vector<int64_t>> input_shapes = {
      {2, 3, 4},
      {3, 4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo = R"( {"boardcast_bias_shape": [1, -1, -1], "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, true, false, false, false], "_ub_factor_align":128, "_base_info": {"210": [32, 4, 21840, 10920]}, "_elewise_vars": {"221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "2 12 ");
}

TEST_F(BiasTiling, Bias_tiling_test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Bias");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Bias("Bias");

  vector<vector<int64_t>> input_shapes = {
      {2, 3, 4},
      {3, 4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT, ge::FORMAT_ND, {});
  std::string compileInfo = R"( {"boardcast_bias_shape": [1, -1, -1], "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, true, false, false, false], "_ub_factor_align":128, "_base_info": {"210": [32, 4, 21840, 10920]}, "_elewise_vars": {"221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "2 12 ");
}

