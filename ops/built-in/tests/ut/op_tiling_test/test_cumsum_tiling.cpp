#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class CumsumTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CumsumTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CumsumTiling TearDown" << std::endl;
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

TEST_F(CumsumTiling, Cumsum_tiling1) {
  std::string op_name = "Cumsum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144}}";
  vector<int64_t> input_shapes = {16, 28, 52};
  vector<int64_t> axis_shapes = {1};
  vector <int32_t> axis = {1};
  

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input_shapes));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_input_axis;
  tensor_input_axis.SetShape(ge::Shape(axis_shapes));
  tensor_input_axis.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(input_shapes));
  tensor_output.SetDataType(ge::DT_FLOAT);

  auto opParas = op::Cumsum("Cumsum");
  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_INPUT_CONST(opParas, tensor_input_axis, axis, (const uint8_t *) axis.data(), axis.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 16 16 1 0 16 1456 0 16320 52 1456 16 28 1 16 52 ");
}

TEST_F(CumsumTiling, Cumsum_tiling2) {
  std::string op_name = "Cumsum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144}}";
  vector<int64_t> input_shapes = {16, 28, 52};
  vector<int64_t> axis_shapes = {1};
  vector <int32_t> axis = {2};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input_shapes));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_input_axis;
  tensor_input_axis.SetShape(ge::Shape(axis_shapes));
  tensor_input_axis.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(input_shapes));
  tensor_output.SetDataType(ge::DT_FLOAT);
  

  auto opParas = op::Cumsum("Cumsum");
  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_INPUT_CONST(opParas, tensor_input_axis, axis, (const uint8_t *) axis.data(), axis.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 448 14 14 0 52 0 16320 1 728 32 52 1 448 1 ");
}

TEST_F(CumsumTiling, Cumsum_tiling3) {
  std::string op_name = "Cumsum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 262144}}";
  vector<int64_t> input_shapes = {16, 28, 2};
  vector<int64_t> axis_shapes = {1};
  vector <int32_t> axis = {2};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input_shapes));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_input_axis;
  tensor_input_axis.SetShape(ge::Shape(axis_shapes));
  tensor_input_axis.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(input_shapes));
  tensor_output.SetDataType(ge::DT_FLOAT);
  

  auto opParas = op::Cumsum("Cumsum");
  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_INPUT_CONST(opParas, tensor_input_axis, axis, (const uint8_t *) axis.data(), axis.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 448 14 14 0 2 0 16320 1 28 32 2 1 448 1 ");
}