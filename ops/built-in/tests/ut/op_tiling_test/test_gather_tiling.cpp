#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "test_common.h"

using namespace std;
using namespace ge;

class GatherTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherTiling TearDown" << std::endl;
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

TEST_F(GatherTiling, gather_tiling_0) {
  std::string op_name = "Gather";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Gather");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, "
      "\"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{
      87552,
  };
  std::vector<int64_t> inputB{174, 1};
  std::vector<int64_t> output{174, 1};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetOriginShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::Gather("Gather");
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, indices);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "13 1 87552 1 174 0 8 0 21 6 0 32512 21 65024 "
            "32512 0 65024 21 0 87552 0 0 0 0 1 1 0 1 ");
  int64_t num = 100;
  for (int64_t i = 0; i < num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(GatherTiling, gather_tiling_1) {
  std::string op_name = "Gather";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Gather");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":1}}";

  std::vector<int64_t> inputA{55, 32, 16};
  std::vector<int64_t> inputB{55, 6};
  std::vector<int64_t> output{55, 6, 16};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetOriginShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::Gather("Gather");
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, indices);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "29 1 32 16 330 0 32 0 6 138 0 6 0 "
            "2464 6 0 0 0 0 512 0 0 0 0 6 1 23 55 ");
}
