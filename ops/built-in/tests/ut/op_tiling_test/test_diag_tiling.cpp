#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "pad_ops.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class DiagTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DiagTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DiagTiling TearDown" << std::endl;
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

TEST_F(DiagTiling, Diag_tiling1) {
  std::string op_name = "Diag";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 256000}}";
  vector<vector<int64_t>> input_shapes = {
      {128, 128},
  };
  std::vector<int64_t> output{64, 1, 64, 0};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(input_shapes[0]));
  tensor_inputA.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  auto opParas = op::Diag("Diag");
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16384 32 512 512 ");
}
