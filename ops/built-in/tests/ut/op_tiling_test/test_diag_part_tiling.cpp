#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class DiagPartTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DiagPartTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DiagPartTiling TearDown" << std::endl;
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

TEST_F(DiagPartTiling, DiagPart_tiling1) {
  std::string op_name = "DiagPart";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 256000}}";
  vector<vector<int64_t>> input_shapes = {
      {128, 128},
  };
  std::vector<int64_t> output{128, 1, 128, 128};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(input_shapes[0]));
  tensor_inputA.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  auto opParas = op::DiagPart("DiagPart");
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  AscendString name(this->test_info_->name());
  AscendString key(compileInfo.c_str());
  optiling::utils::OpCompileInfo op_compile_info(name, key);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "128 1 128 128 ");
}
