#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_norm_ops.h"
#include "array_ops.h"

using namespace std;

class BinaryCrossEntropyGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BinaryCrossEntropyGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BinaryCrossEntropyGradTiling TearDown" << std::endl;
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

TEST_F(BinaryCrossEntropyGradTiling, BinaryCrossEntropyGrad_tiling_test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BinaryCrossEntropyGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BinaryCrossEntropyGrad("BinaryCrossEntropyGrad");

  vector<vector<int64_t>> input_shapes = {
      {16, 8, 375},
      {16, 8, 375},
      {16, 8, 375},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, y, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad_output, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, input_shapes[0], ge::DT_FLOAT, ge::FORMAT_ND, {});

  std::string compileInfo = R"({"_fusion_index": [[0], [1, 2]],"reduce_mean_cof_dtype": "float32", "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 4, 8184, 4088], "210": [32, 4, 10920, 5456]}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10001], "221000001": [10000, 10001, 20000, 30000], "221000002": [10000, 10001, 20000, 30001], "210000004": [10000, 10001, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof"], "221000000": ["_dim_0_0", "_dim_0_1", "cof"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1", "cof"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1", "cof"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_0_1"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "221000000": [], "221000001": [], "221000002": [], "221000004": []}, "_custom_vars": {"210000000": ["cof"], "210010000": ["cof"], "221000000": ["cof"], "221000001": ["cof"], "221000002": ["cof"], "221000004": ["cof"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "48000 1 1536 934200126 ");
}

TEST_F(BinaryCrossEntropyGradTiling, BinaryCrossEntropyGrad_tiling_test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BinaryCrossEntropyGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BinaryCrossEntropyGrad("BinaryCrossEntropyGrad");

  vector<vector<int64_t>> input_shapes = {
      {16, 8, 375},
      {16, 8, 375},
      {16, 8, 375},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, y, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad_output, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});

  std::string compileInfo = R"({"_fusion_index": [[0], [1, 2]],"reduce_mean_cof_dtype": "float16", "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 21840, 10912], "210": [32, 2, 21840, 10912]}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10001], "221000001": [10000, 10001, 20000, 30000], "221000002": [10000, 10001, 20000, 30001], "210000004": [10000, 10001, 20001, 30001]}, "push_status": 1, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"], "221000000": ["_dim_0_0", "_dim_0_1", "cof", "cof_empty"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1", "cof", "cof_empty"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_0_1"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "221000000": [], "221000001": [], "221000002": [], "221000004": []}, "_custom_vars": {"210000000": ["cof", "cof_empty"], "210010000": ["cof", "cof_empty"], "221000000": ["cof", "cof_empty"], "221000001": ["cof", "cof_empty"], "221000002": ["cof", "cof_empty"], "221000004": ["cof", "cof_empty"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "48000 1 1536 350 ");
}