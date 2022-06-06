#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_profiling_reg.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"
using namespace ge;
using namespace ut_util;

class GroupNormTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GroupNormTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GroupNormTiling TearDown" << std::endl;
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

TEST_F(GroupNormTiling, GroupNorm_tiling_0) {
  std::string op_name = "GroupNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::GroupNorm("GroupNorm");
  std::string compileInfo ="{\"vars\": {\"core_num\": 8, \"num_groups\": 32}}";

  std::vector<int64_t> input{8, 64, 1, 8192};
  std::vector<int64_t> output{8, 64, 1, 8192};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_NCHW, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_NCHW, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 16384 8192 2 2 512 1 512 32 8 32 64 1024 512 ");
}

TEST_F(GroupNormTiling, GroupNorm_tiling_1) {
  std::string op_name = "GroupNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::GroupNorm("GroupNorm");
  std::string compileInfo ="{\"vars\": {\"core_num\": 8, \"num_groups\": 1}}";

  std::vector<int64_t> input{1, 1, 15, 14, 16};
  std::vector<int64_t> output{1, 1, 15, 14, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_NC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 3360 3360 1 1 210 1 210 1 1 1 16 210 210 ");
}
