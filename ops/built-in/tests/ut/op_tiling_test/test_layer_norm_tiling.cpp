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

class LayerNormTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "LayerNormTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "LayerNormTiling TearDown" << std::endl; }
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

TEST_F(LayerNormTiling, LayerNorm_tiling_test_1) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_ori_reduce_axis": [-1],
                        "_input_type": [0, 1, 1],
                        "_ori_broadcast_axis": [0, 1],
                        "is_support_vexp": true,
                        "reduce_mean_cof_dtype": "float32",
                        "_pattern": "Norm",
                        "_common_info": [30, 16, 128],
                        "_exist_output_after_reduce": true,
                        "_available_ub_size": {
                        "4005": [21128, 16112, 15848],
                        "5006": [16320, 13072, 13056]},
                        "_exist_workspace_after_reduce": false,
                        "_workspace_info": {
                          "1000400500": [32],
                          "1000400501": [4],
                          "1200400500": [32],
                          "1000500610": [4],
                          "1000500611": [32]
                        },
                        "_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_normal_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_attr_vars": {
                          "1000400500": [],
                          "1000400501": [],
                          "1200400500": [],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_custom_vars": {
                          "1000400500": ["mean_cof"],
                          "1000400501": ["mean_cof"],
                          "1200400500": ["mean_cof"],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_norm_vars": {
                          "1000400500": [20000, 20001, 30000, 40000],
                          "1000400501": [20000, 20001, 30000, 40000],
                          "1200400500": [20000, 20001, 30000, 40000],
                          "1000500610": [20001, 30000, 40000],
                          "1000500611": [20001, 30000, 40000]
                        }})";

  std::vector<std::vector<int64_t>> inputs{{11, 12, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{11, 12, 512}, {11, 12, 1}, {11, 12, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  ge::Format data_format = ge::FORMAT_NCHW;

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
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);
  TensorDesc tensor_outputC;
  tensor_outputC.SetShape(ge::Shape(outputs[2]));
  tensor_outputC.SetDataType(output_types[2]);
  tensor_outputC.SetFormat(data_format);

  auto opParas = op::LayerNorm(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, gamma);
  TENSOR_INPUT(opParas, tensor_inputC, beta);
  TENSOR_OUTPUT(opParas, tensor_outputA, y);
  TENSOR_OUTPUT(opParas, tensor_outputB, mean);
  TENSOR_OUTPUT(opParas, tensor_outputC, variance);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 9);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "132 512 16 16 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_2) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_ori_reduce_axis": [2],
                        "_input_type": [0, 1, 1],
                        "_ori_broadcast_axis": [0, 1],
                        "is_support_vexp": true,
                        "reduce_mean_cof_dtype": "float32",
                        "_pattern": "Norm",
                        "_common_info": [30, 16, 128],
                        "_exist_output_after_reduce": true,
                        "_available_ub_size": {
                        "4005": [21128, 16112, 15848],
                        "5006": [16320, 13072, 13056]},
                        "_exist_workspace_after_reduce": false,
                        "_workspace_info": {
                          "1000400500": [32],
                          "1000400501": [4],
                          "1200400500": [32],
                          "1000500610": [4],
                          "1000500611": [32]
                        },
                        "_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_normal_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_attr_vars": {
                          "1000400500": [],
                          "1000400501": [],
                          "1200400500": [],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_custom_vars": {
                          "1000400500": ["mean_cof"],
                          "1000400501": ["mean_cof"],
                          "1200400500": ["mean_cof"],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_norm_vars": {
                          "1000400500": [20000, 20001, 30000, 40000],
                          "1000400501": [20000, 20001, 30000, 40000],
                          "1200400500": [20000, 20001, 30000, 40000],
                          "1000500610": [20001, 30000, 40000],
                          "1000500611": [20001, 30000, 40000]
                        }})";

  std::vector<std::vector<int64_t>> inputs{{11, 12, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{11, 12, 512}, {11, 12, 1}, {11, 12, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  ge::Format data_format = ge::FORMAT_NCHW;

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
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);
  TensorDesc tensor_outputC;
  tensor_outputC.SetShape(ge::Shape(outputs[2]));
  tensor_outputC.SetDataType(output_types[2]);
  tensor_outputC.SetFormat(data_format);

  auto opParas = op::LayerNorm(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, gamma);
  TENSOR_INPUT(opParas, tensor_inputC, beta);
  TENSOR_OUTPUT(opParas, tensor_outputA, y);
  TENSOR_OUTPUT(opParas, tensor_outputB, mean);
  TENSOR_OUTPUT(opParas, tensor_outputC, variance);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 9);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "132 512 16 16 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_3) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_ori_reduce_axis": [3],
                        "_input_type": [0, 1, 1],
                        "_ori_broadcast_axis": [0, 1],
                        "is_support_vexp": true,
                        "reduce_mean_cof_dtype": "float32",
                        "_pattern": "Norm",
                        "_common_info": [30, 16, 128],
                        "_exist_output_after_reduce": true,
                        "_available_ub_size": {
                        "4005": [21128, 16112, 15848],
                        "5006": [16320, 13072, 13056]},
                        "_exist_workspace_after_reduce": false,
                        "_workspace_info": {
                          "1000400500": [32],
                          "1000400501": [4],
                          "1200400500": [32],
                          "1000500610": [4],
                          "1000500611": [32]
                        },
                        "_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_normal_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_attr_vars": {
                          "1000400500": [],
                          "1000400501": [],
                          "1200400500": [],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_custom_vars": {
                          "1000400500": ["mean_cof"],
                          "1000400501": ["mean_cof"],
                          "1200400500": ["mean_cof"],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_norm_vars": {
                          "1000400500": [20000, 20001, 30000, 40000],
                          "1000400501": [20000, 20001, 30000, 40000],
                          "1200400500": [20000, 20001, 30000, 40000],
                          "1000500610": [20001, 30000, 40000],
                          "1000500611": [20001, 30000, 40000]
                        }})";

  std::vector<std::vector<int64_t>> inputs{{11, 12, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{11, 12, 512}, {11, 12, 1}, {11, 12, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  ge::Format data_format = ge::FORMAT_NCHW;

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
  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format);
  TensorDesc tensor_outputC;
  tensor_outputC.SetShape(ge::Shape(outputs[2]));
  tensor_outputC.SetDataType(output_types[2]);
  tensor_outputC.SetFormat(data_format);

  auto opParas = op::LayerNorm(op_name);
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, gamma);
  TENSOR_INPUT(opParas, tensor_inputC, beta);
  TENSOR_OUTPUT(opParas, tensor_outputA, y);
  TENSOR_OUTPUT(opParas, tensor_outputB, mean);
  TENSOR_OUTPUT(opParas, tensor_outputC, variance);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}