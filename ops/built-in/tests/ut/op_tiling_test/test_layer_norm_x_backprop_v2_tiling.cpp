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

class LayerNormXBackpropV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "LayerNormXBackpropV2Tiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "LayerNormXBackpropV2Tiling TearDown" << std::endl; }
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

TEST_F(LayerNormXBackpropV2Tiling, LayerNormXBackpropV2_tiling_test_1) {
  std::string op_name = "LayerNormXBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_pattern": "Layer_norm_x_backprop_v2",
                        "UB_SIZE":262112,
                        "CORE_NUM":32,
                        "MAX_DTYPE": 4,
                        "COEXISTING_QUANTITY": 7,
                        "_vars": {"10000": ["dim_0", "dim_1"]},
                        "_normal_vars": {"10000": []},
                        "_attr_vars": {"10000": []},
                        "_custom_vars": {"10000": ["dim_0", "dim_1"]}
                        })";

  std::vector<std::vector<int64_t>> inputs{{13, 32, 512}, {13, 32, 512}, {13, 32, 1}, {13, 32, 1}, {512}};
  std::vector<std::vector<int64_t>> outputs{{13, 32, 512}, {13, 32, 512}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT};
  ge::Format data_format = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format);
  TensorDesc tensor_inputC;
  tensor_inputC.SetShape(ge::Shape(inputs[2]));
  tensor_inputC.SetDataType(input_types[2]);
  tensor_inputC.SetFormat(data_format);
  TensorDesc tensor_inputD;
  tensor_inputD.SetShape(ge::Shape(inputs[3]));
  tensor_inputD.SetDataType(input_types[3]);
  tensor_inputD.SetFormat(data_format);
  TensorDesc tensor_inputE;
  tensor_inputE.SetShape(ge::Shape(inputs[4]));
  tensor_inputE.SetDataType(input_types[4]);
  tensor_inputE.SetFormat(data_format);
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_inputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);

  auto opParas = op::LayerNormXBackpropV2(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, dy);
  TENSOR_INPUT(opParas, tensor_inputB, x);
  TENSOR_INPUT(opParas, tensor_inputC, variance);
  TENSOR_INPUT(opParas, tensor_inputD, mean);
  TENSOR_INPUT(opParas, tensor_inputE, gamma);
  TENSOR_OUTPUT(opParas, tensor_outputA, pd_x);
  TENSOR_OUTPUT(opParas, tensor_outputB, res_for_gamma);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "13 32 ");
}
