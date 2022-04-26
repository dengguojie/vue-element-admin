#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "array_ops.h"

using namespace std;

class ExpandTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ExpandTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ExpandTiling TearDown" << std::endl;
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
.INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .INPUT(shape, TensorType({DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
*/

TEST_F(ExpandTiling, Expand_tiling_test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Expand");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Expand("Expand");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 5},
      {3},
  };
  vector<int64_t> output = {3, 1, 5};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32};
  std::vector<int32_t> shape_value{3, 1, 5};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, input_shapes[1], dtypes[1], ge::FORMAT_ND, shape_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo =
      R"( {"_pattern": "Broadcast", "_ub_factor_align": 128, "push_status": 0,"_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 2, 43680, 21840]}, "_elewise_vars": {"0": [10000, 10100], "1": [10000, 10100, 20000, 30000], "2": [10000, 10100, 20000, 30001], "3": [10000, 10100, 20000, 30002], "5": [10000, 10100, 20001, 30001], "6": [10000, 10100, 20001, 30002], "9": [10000, 10100, 20002, 30002]}, "_vars": {"0": ["_dim_0_0", "_dim_1_0"], "1": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_1_0", "_block_factor_2", "_ub_factor_2"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 1 ");
}
