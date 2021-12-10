#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
using namespace ge;
using namespace ut_util;
using namespace std;

class SplitTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitTiling TearDown" << std::endl;
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

TEST_F(SplitTiling, Split_tiling0) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Split");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Split("Split");
  vector<vector<int64_t>> input_shapes = {
      {0, 8},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {0, 8},
  };
  vector<ge::DataType> dtypes = {ge::DT_INT8, ge::DT_INT32};
  vector<int32_t> split_dim{0};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[1], dtypes[1], FORMAT_ND, split_dim);

  opParas.SetAttr("num_split", {1820});

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":253952, \"num_split\":1}}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(SplitTiling, Split_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Split");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Split("Split");
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 232},
  };
  vector<ge::DataType> dtypes = {ge::DT_INT8, ge::DT_INT32};
  vector<int32_t> split_dim{0};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[1], dtypes[1], FORMAT_ND, split_dim);

  opParas.SetAttr("num_split", {1820});

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":253952, \"num_split\":1}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 32 422240 1820 13195 13195 0 13195 13195 0 13195 13195 232 1 422240 0 0 0 0 0 0 0 0 1820 ");
}

TEST_F(SplitTiling, Split_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Split");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Split("Split");
  vector<vector<int64_t>> input_shapes = {
      {40000, 84},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {40000, 1},
      {40000, 2},
      {40000, 1},
      {40000, 80},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32};
  std::vector<int32_t> split_dim{1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_FRACTAL_NZ, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[1], dtypes[1], FORMAT_FRACTAL_NZ, split_dim);

  opParas.SetAttr("num_split", {1, 2, 1, 80});

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_FRACTAL_NZ, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}