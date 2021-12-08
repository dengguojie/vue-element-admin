#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class BitwiseTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BitwiseTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BitwiseTiling TearDown" << std::endl;
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

TEST_F(BitwiseTiling, Bitwise_tiling_test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BitwiseAnd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BitwiseAnd("BitwiseAnd");

  vector<vector<int64_t>> input_shapes = {
      {3, 3, 3},
      {3, 3, 3},
  };

  vector<ge::DataType> dtypes = {ge::DT_INT16, ge::DT_INT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo = R"( {"_fusion_index": [[0], [1], [2]], "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_outs_uint1": false, "_base_info": {"000": [32, 2, 32752, 16368]}, "_elewise_vars": {"0": [10000, 10001], "1": [10000, 10001, 20000, 30000], "2": [10000, 10001, 20000, 30001], "3": [10000, 10001, 20000, 30002], "5": [10000, 10001, 20001, 30001], "6": [10000, 10001, 20001, 30002], "9": [10000, 10001, 20002, 30002]}, "_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_normal_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_attr_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}, "_custom_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "3 3 ");
}

TEST_F(BitwiseTiling, Bitwise_tiling_test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BitwiseOr");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BitwiseOr("BitwiseOr");

  vector<vector<int64_t>> input_shapes = {
      {3, 3, 3},
      {3, 3, 3},
  };

  vector<ge::DataType> dtypes = {ge::DT_INT32, ge::DT_INT32};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo = R"( {"_fusion_index": [[0], [1], [2]], "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_outs_uint1": false, "_base_info": {"000": [32, 2, 32752, 16368]}, "_elewise_vars": {"0": [10000, 10001], "1": [10000, 10001, 20000, 30000], "2": [10000, 10001, 20000, 30001], "3": [10000, 10001, 20000, 30002], "5": [10000, 10001, 20001, 30001], "6": [10000, 10001, 20001, 30002], "9": [10000, 10001, 20002, 30002]}, "_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_normal_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_attr_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}, "_custom_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "3 3 ");
}

TEST_F(BitwiseTiling, Bitwise_tiling_test_3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BitwiseXor");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BitwiseXor("BitwiseXor");

  vector<vector<int64_t>> input_shapes = {
      {3, 3, 3},
      {3, 3, 3},
  };

  vector<ge::DataType> dtypes = {ge::DT_INT32, ge::DT_INT32};
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo = R"( {"_fusion_index": [[0], [1], [2]], "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_outs_uint1": false, "_base_info": {"000": [32, 2, 32752, 16368]}, "_elewise_vars": {"0": [10000, 10001], "1": [10000, 10001, 20000, 30000], "2": [10000, 10001, 20000, 30001], "3": [10000, 10001, 20000, 30002], "5": [10000, 10001, 20001, 30001], "6": [10000, 10001, 20001, 30002], "9": [10000, 10001, 20002, 30002]}, "_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_normal_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_attr_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}, "_custom_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "3 3 ");
}
