#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace std;

class SmoothL1LossGradV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SmoothL1LossGradV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SmoothL1LossGradV2Tiling TearDown" << std::endl;
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
using namespace ut_util;

TEST_F(SmoothL1LossGradV2Tiling, SmoothL1LossGradV2_tiling_test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossGradV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossGradV2("SmoothL1LossGradV2");

  vector<vector<int64_t>> input_shapes = {
      {16},
      {16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, dout, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, gradient, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "mean", "_fusion_index": [[0]], "_pattern": "Broadcast", "_outs_uint1": false, "reduce_mean_cof_dtype": "float32", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "200": [32, 2, 42320, 21152]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "220000000": [10000, 10001, 20000, 30000]}, "push_status": 1, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "220000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "220000000": ["cof", "cof_empty"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),"16 16 16 16 1031798784 ");
}

TEST_F(SmoothL1LossGradV2Tiling, SmoothL1LossGradV2_tiling_test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossGradV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossGradV2("SmoothL1LossGradV2");

  vector<vector<int64_t>> input_shapes = {
      {16},
      {16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, dout, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, gradient, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "mean", "_fusion_index": [[0]], "_pattern": "Broadcast", "_outs_uint1": false, "reduce_mean_cof_dtype": "float16", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "200": [32, 2, 42320, 21152]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "220000000": [10000, 10001, 20000, 30000]}, "push_status": 1, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "220000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "220000000": ["cof", "cof_empty"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 16 16 11264 ");
}

TEST_F(SmoothL1LossGradV2Tiling, SmoothL1LossGradV2_tiling_test_3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossGradV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossGradV2("SmoothL1LossGradV2");

  vector<vector<int64_t>> input_shapes = {
      {16},
      {16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, dout, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});  
  TENSOR_OUTPUT_WITH_SHAPE(opParas, gradient, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "sum", "_fusion_index": [[0]], "_pattern": "Broadcast", "_outs_uint1": false, "reduce_mean_cof_dtype": "float16", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "200": [32, 2, 42320, 21152]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "220000000": [10000, 10001, 20000, 30000]}, "push_status": 1, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "220000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "220000000": ["cof", "cof_empty"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 16 16 11264 ");
}

TEST_F(SmoothL1LossGradV2Tiling, SmoothL1LossGradV2_tiling_test_4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossGradV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossGradV2("SmoothL1LossGradV2");

  vector<vector<int64_t>> input_shapes = {
      {16},
      {16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, dout, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});   
  TENSOR_OUTPUT_WITH_SHAPE(opParas, gradient, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "sum", "_fusion_index": [[0]], "_pattern": "Broadcast", "_outs_uint1": false, "reduce_mean_cof_dtype": "float32", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "200": [32, 2, 42320, 21152]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "220000000": [10000, 10001, 20000, 30000]}, "push_status": 1, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "220000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "220000000": ["cof", "cof_empty"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 16 16 1031798784 ");
}
