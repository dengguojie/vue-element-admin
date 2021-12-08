#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_norm_ops.h"
#include "array_ops.h"

using namespace std;

class ScaleTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScaleTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScaleTiling TearDown" << std::endl;
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

TEST_F(ScaleTiling, Scale_tiling_test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Scale");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Scale("Scale");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1},
      {1,1,1},
      {1,1,1}
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, scale, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo = R"({"_boardcast_scale_shape": [-1, 1, -1], "_fusion_index": [[0, 1, 2]], "push_status": 0, "_pattern": "ElemWise", "_flag_info": [false, false, true, true, true, false, false], "_outs_uint1": false, "_base_info": {"100": [2, 2, 42320, 21152], "230": [2, 2, 39584, 19792]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "223000000": [10000, 20000, 30000]}, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 1 ");
}

TEST_F(ScaleTiling, Scale_tiling_test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Scale");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Scale("Scale");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1},
      {1,1,1},
      {1,1,1}
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, scale, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo = R"({"_boardcast_scale_shape": [-1, 1, -1], "_fusion_index": [[0, 1, 2]], "push_status": 0, "_pattern": "ElemWise", "_flag_info": [false, false, true, true, true, false, false], "_outs_uint1": false, "_base_info": {"100": [2, 2, 42320, 21152], "230": [2, 2, 39584, 19792]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "223000000": [10000, 20000, 30000]}, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 1 ");
}

