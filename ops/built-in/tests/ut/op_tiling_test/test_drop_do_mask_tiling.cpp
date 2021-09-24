#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "nn_norm_ops.h"
#include "array_ops.h"

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

using namespace ge;
#include "test_common.h"
/*
.INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_UINT8}))
    .INPUT(keep_prob, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
*/

TEST_F(DropOutDoMaskTiling, dropout_do_mask_tiling_1) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("DropOutDoMask");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::DropOutDoMask("DropOutDoMask");

  std::vector<int64_t> input_x_shape = {40, 10};
  std::vector<int64_t> input_mask_shape = {40};
  std::vector<int64_t> input_keep_prob_shape = {1};

  TensorDesc tensorInputX;
  tensorInputX.SetShape(ge::Shape(input_x_shape));
  tensorInputX.SetDataType(ge::DT_FLOAT);

  TensorDesc tensorInputMask;
  tensorInputMask.SetShape(ge::Shape(input_mask_shape));
  tensorInputMask.SetDataType(ge::DT_UINT8);

  TensorDesc tensorInputKeepProb;
  tensorInputKeepProb.SetShape(ge::Shape(input_keep_prob_shape));
  tensorInputKeepProb.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensorInputX, x);
  TENSOR_INPUT(opParas, tensorInputMask, mask);
  TENSOR_INPUT(opParas, tensorInputKeepProb, keep_prob);

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 128 16 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}