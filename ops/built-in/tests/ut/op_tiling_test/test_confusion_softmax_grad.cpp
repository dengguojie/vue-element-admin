#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>
#define private public
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "nn_norm_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class ConfusionSoftmaxGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "ConfusionSoftmaxGradTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "ConfusionSoftmaxGradTiling TearDown" << std::endl; }
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

TEST_F(ConfusionSoftmaxGradTiling, ConfusionSoftmaxGradTiling1) {
  std::string op_name = "ConfusionSoftmaxGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"_pattern": "ConfusionSoftmaxGrad", 
                                  "_vars": {
                                  "10000000": ["dim_0_0", "dim_0_1", "dim_0_2"], 
                                  "10000002": ["dim_0_0", "dim_0_1", "dim_0_2", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"10000000": [], "10000002": []},
                                  "_attr_vars": {"10000000": [], "10000002": []},
                                  "_custom_vars": {"10000000": ["dim_0_0", "dim_0_1", "dim_0_2"],
                                                   "10000002": ["dim_0_0", "dim_0_1", "dim_0_2", "block_factor", "ub_factor"]}})";

  std::vector<int64_t> inputA{1, 2, 32};
  std::vector<int64_t> inputB{1, 1, 32};
  std::vector<int64_t> outputA{1, 2, 32};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputA));
  tensor_outputA.SetDataType(ge::DT_FLOAT);

  auto opParas = op::ConfusionSoftmaxGrad(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, grad);
  TENSOR_INPUT(opParas, tensor_inputB, x);
  TENSOR_OUTPUT(opParas, tensor_outputA, y);

  optiling::utils::OpRunInfo runInfo;
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000002);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 2 32 1 2 ");
}