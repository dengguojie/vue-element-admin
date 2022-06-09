#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
#include "common_unittest.h"
#include "drop_out_do_mask.h"
using namespace ge;
using namespace ut_util;
using namespace std;


class DropOutDoMaskTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DropOutDoMaskTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DropOutDoMaskTiling TearDown" << std::endl;
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

TEST_F(DropOutDoMaskTiling, dropout_do_mask_tiling_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DropOutDoMask");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::DropOutDoMask("DropOutDoMask");

  std::vector<int64_t> input_x_shape = {40, 10};
  std::vector<int64_t> input_mask_shape = {40};
  std::vector<int64_t> input_keep_prob_shape = {1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_x_shape, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, mask, input_mask_shape, DT_UINT8, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, keep_prob, input_keep_prob_shape, DT_FLOAT, FORMAT_ND, {});

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 128 16 ");

  optiling::DropOutDoMaskCompileInfo info;
  int32_t tiling_len = sizeof(optiling::DropOutDoMaskTilingData);
  // prepare compile info from json string to compile struct members
  TILING_PARSE_JSON_TO_COMPILEINFO("DropOutDoMask", compileInfo, info);
  ATTACH_OPERATOR_TO_HOLDER(holder, opParas, tiling_len, info);
  HOLDER_DO_TILING(holder, "DropOutDoMask", ge::GRAPH_SUCCESS);
  TILING_DATA_VERIFY_BYTYPE(holder, int64_t, "4 128 16 ");

}