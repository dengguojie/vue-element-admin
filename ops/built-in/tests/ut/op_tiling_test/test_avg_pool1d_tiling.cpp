#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#define private public
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "nn_pooling_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class AvgPool1DTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "AvgPool1DTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "AvgPool1DTiling TearDown" << std::endl; }
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

TEST_F(AvgPool1DTiling, avgpool1d_tiling_0) {
  std::string op_name = "AvgPool1DD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AvgPool1DD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\":32, \"max_w_in_ub\":2730, \"ksize\":3, \"strides\":1,\"pad_l\":0,\"pad_r\":0,\"ceil_mode\":true}";

  std::vector<int64_t> inputA{1, 1, 1, 3, 16};
  std::vector<int64_t> inputB{1, 1, 1, 1, 16};
  std::vector<int64_t> output{1, 1, 1, 1, 16};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::AvgPool1DD(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, assist_matrix);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 3 1 ");
}

TEST_F(AvgPool1DTiling, avgpool1d_tiling_1) {
  std::string op_name = "AvgPool1DD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AvgPool1DD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\":32, \"max_w_in_ub\":455, \"ksize\":8, \"strides\":1,\"pad_l\":1,\"pad_r\":1,\"ceil_mode\":true}";

  std::vector<int64_t> inputA{153, 30, 1, 7, 16};
  std::vector<int64_t> inputB{153, 30, 1, 2, 16};
  std::vector<int64_t> output{153, 30, 1, 2, 16};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::AvgPool1DD(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, assist_matrix);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4590 7 2 144 ");
}

TEST_F(AvgPool1DTiling, avgpool1d_tiling_2) {
  std::string op_name = "AvgPool1DD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AvgPool1DD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\":32, \"max_w_in_ub\":455, \"ksize\":8, \"strides\":1,\"pad_l\":1,\"pad_r\":1,\"ceil_mode\":true}";

  std::vector<int64_t> inputA{153, 30, 1, 15, 16};
  std::vector<int64_t> inputB{153, 30, 1, 10, 16};
  std::vector<int64_t> output{153, 30, 1, 10, 16};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::AvgPool1DD(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, assist_matrix);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4590 15 10 144 ");
}

TEST_F(AvgPool1DTiling, avgpool1d_tiling_3) {
  std::string op_name = "AvgPool1DD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AvgPool1DD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\":32, \"max_w_in_ub\":455, \"ksize\":8, \"strides\":1,\"pad_l\":1,\"pad_r\":1,\"ceil_mode\":true}";

  std::vector<int64_t> inputA{153, 30, 1, 20, 16};
  std::vector<int64_t> inputB{153, 30, 1, 15, 16};
  std::vector<int64_t> output{153, 30, 1, 15, 16};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::AvgPool1DD(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, assist_matrix);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4590 20 15 144 ");
}

TEST_F(AvgPool1DTiling, avgpool1d_tiling_4) {
  using namespace optiling;
  std::string op_name = "AvgPool1DD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AvgPool1DD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\":32, \"max_w_in_ub\":455, \"ksize\":8, \"strides\":1,\"pad_l\":1,\"pad_r\":1,\"ceil_mode\":true}";

  std::vector<int64_t> inputA{153, 30, 1, 80, 16};
  std::vector<int64_t> inputB{153, 30, 1, 75, 16};
  std::vector<int64_t> output{153, 30, 1, 75, 16};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::AvgPool1DD(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, assist_matrix);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4590 80 75 144 ");
}

TEST_F(AvgPool1DTiling, avgpool1d_tiling_5) {
  using namespace optiling;
  std::string op_name = "AvgPool1DD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("AvgPool1DD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"core_num\":32, \"max_w_in_ub\":455, \"ksize\":8, \"strides\":1,\"pad_l\":1,\"pad_r\":1,\"ceil_mode\":true}";

  std::vector<int64_t> inputA{153, 30, 1, 300, 16};
  std::vector<int64_t> inputB{153, 30, 1, 295, 16};
  std::vector<int64_t> output{153, 30, 1, 295, 16};

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(ge::DT_FLOAT16);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(ge::DT_INT32);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT16);

  auto opParas = op::AvgPool1DD(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, assist_matrix);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4590 300 295 144 ");
}
