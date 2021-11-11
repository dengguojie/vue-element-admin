#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>
#define private public
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "transformation_ops.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class UnpackTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "UnpackTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "UnpackTiling TearDown" << std::endl; }
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

TEST_F(UnpackTiling, Unpack_tiling1) {
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_1
  std::string compileInfo =
      R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 1, "axis": 2, "is_special_tiling": false}, "vars": {"0": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{100, 1, 1, 902, 6};
  std::vector<int64_t> output{100, 1, 902, 6};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  auto opParas = op::Unpack(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100 5412 32 16913 16913 ");
}

TEST_F(UnpackTiling, Unpack_tiling2) {
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_2
  std::string compileInfo =
      R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 2, "axis": 1, "is_special_tiling": false}, "vars": {"1": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "2": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "3": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{70, 2, 758, 518};
  std::vector<int64_t> output{70, 758, 518};
  std::vector<int64_t> outputB{70, 758, 518};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputB));
  tensor_outputB.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::Unpack(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  // TENSOR_OUTPUT(opParas, tensor_outputB, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "70 392644 32 392644 130876 ");
}

TEST_F(UnpackTiling, Unpack_tiling3) {
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_3
  std::string compileInfo =
      R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 2, "axis": 1, "is_special_tiling": false}, "vars": {"1": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "2": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "3": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{1, 2, 758, 518};
  std::vector<int64_t> output{1, 758, 518};
  std::vector<int64_t> outputB{1, 758, 518};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputB));
  tensor_outputB.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::Unpack(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  // TENSOR_OUTPUT(opParas, tensor_outputB, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 392644 1 12271 1 ");
}

TEST_F(UnpackTiling, Unpack_tiling4) {
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_4
  std::string compileInfo =
      R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 2, "axis": 1, "is_special_tiling": false}, "vars": {"1": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "2": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "3": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{70, 2, 4};
  std::vector<int64_t> output{70, 4};
  std::vector<int64_t> outputB{70, 4};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputB));
  tensor_outputB.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::Unpack(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  // TENSOR_OUTPUT(opParas, tensor_outputB, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "70 4 23 4 4 ");
}

TEST_F(UnpackTiling, Unpack_tiling5) {
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_5
  std::string compileInfo =
      R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 2, "axis": 1, "is_special_tiling": false}, "vars": {"1": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "2": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "3": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{70, 2, 16};
  std::vector<int64_t> output{70, 16};
  std::vector<int64_t> outputB{70, 16};
  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputB));
  tensor_outputB.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::Unpack(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  // TENSOR_OUTPUT(opParas, tensor_outputB, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "70 16 32 16 3 ");
}

TEST_F(UnpackTiling, Unpack_tiling6) {
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_6
  std::string compileInfo =
      R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 2, "axis": 1, "is_special_tiling": false}, "vars": {"1": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "2": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "3": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{32, 2, 758, 520};
  std::vector<int64_t> output{32, 758, 520};
  std::vector<int64_t> outputB{32, 758, 520};
  std::string in_dtype = "float16";
  std::string dtype = "float16";

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputB));
  tensor_outputB.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::Unpack(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  // TENSOR_OUTPUT(opParas, tensor_outputB, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}