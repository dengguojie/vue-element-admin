#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "all_ops.h"

using namespace std;

class SparseApplyProximalAdagradDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseApplyProximalAdagradDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseApplyProximalAdagradDTiling TearDown" << std::endl;
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

TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_0) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyProximalAdagradD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952, \"ub_tensor_num\": 2}}";

  auto opParas = op::SparseApplyProximalAdagradD("SparseApplyProximalAdagradD");
  vector<vector<int64_t>> input_shapes = {
      {10, 20, 32}, {10, 20, 32}, {1}, {1}, {1}, {10, 20, 32}, {10, 20},
  };
  vector<vector<int64_t>> output_shapes = {{10, 20, 32}, {10, 20, 32}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                 ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  TENSOR_INPUT_WITH_SHAPE(opParas, var, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, accum, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, lr, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l1, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l2, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, input_shapes[5], dtypes[5], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, input_shapes[6], dtypes[6], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output_shapes[0], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, accum, output_shapes[1], ge::DT_FLOAT, ge::FORMAT_NHWC, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 968 200 121 25 4 32 1 ");
}
TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_1) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyProximalAdagradD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952, \"ub_tensor_num\": 2}}";

  auto opParas = op::SparseApplyProximalAdagradD("SparseApplyProximalAdagradD");
  vector<vector<int64_t>> input_shapes = {
      {10, 20, 32}, {10, 20, 32}, {1}, {1}, {1}, {10, 20}, {10, 20, 32},
  };
  vector<vector<int64_t>> output_shapes = {{10, 20, 32}, {10, 20, 32}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                 ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  TENSOR_INPUT_WITH_SHAPE(opParas, var, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, accum, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, lr, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l1, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l2, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, input_shapes[5], dtypes[5], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, input_shapes[6], dtypes[6], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output_shapes[0], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, accum, output_shapes[1], ge::DT_FLOAT, ge::FORMAT_NHWC, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_2) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyProximalAdagradD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_tensor_num\": 2}}";

  auto opParas = op::SparseApplyProximalAdagradD("SparseApplyProximalAdagradD");
  vector<vector<int64_t>> input_shapes = {
      {10, 20, 32}, {10, 20, 32}, {1}, {1}, {1}, {10, 20, 32}, {10, 20},
  };
  vector<vector<int64_t>> output_shapes = {{10, 20, 32}, {10, 20, 32}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                 ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  TENSOR_INPUT_WITH_SHAPE(opParas, var, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, accum, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, lr, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l1, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l2, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, input_shapes[5], dtypes[5], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, input_shapes[6], dtypes[6], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output_shapes[0], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, accum, output_shapes[1], ge::DT_FLOAT, ge::FORMAT_NHWC, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_3) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyProximalAdagradD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952}}";

  auto opParas = op::SparseApplyProximalAdagradD("SparseApplyProximalAdagradD");
  vector<vector<int64_t>> input_shapes = {
      {10, 20, 32}, {10, 20, 32}, {1}, {1}, {1}, {10, 20, 32}, {10, 20},
  };
  vector<vector<int64_t>> output_shapes = {{10, 20, 32}, {10, 20, 32}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                 ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  TENSOR_INPUT_WITH_SHAPE(opParas, var, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, accum, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, lr, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l1, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, l2, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, input_shapes[5], dtypes[5], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, input_shapes[6], dtypes[6], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output_shapes[0], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, accum, output_shapes[1], ge::DT_FLOAT, ge::FORMAT_NHWC, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}