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

TEST_F(LayerNormTiling, LayerNorm_Norm_tiling_test_1) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_ori_reduce_axis": [-1],
                        "_input_type": [0, 1, 1],
                        "_ori_broadcast_axis": [0, 1],
                        "is_support_vexp_pattern": true,
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
                          "1300400500": [32],
                          "1000500610": [4],
                          "1000500611": [32]
                        },
                        "_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_normal_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_attr_vars": {
                          "1000400500": [],
                          "1000400501": [],
                          "1200400500": [],
                          "1300400500": [],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_custom_vars": {
                          "1000400500": ["mean_cof"],
                          "1000400501": ["mean_cof"],
                          "1200400500": ["mean_cof"],
                          "1300400500": ["mean_cof"],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_norm_vars": {
                          "1000400500": [20000, 20001, 30000, 40000],
                          "1000400501": [20000, 20001, 30000, 40000],
                          "1200400500": [20000, 20001, 30000, 40000],
                          "1300400500": [20000, 20001, 30000, 40000],
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

TEST_F(LayerNormTiling, LayerNorm_Norm_tiling_test_2) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_ori_reduce_axis": [2],
                        "_input_type": [0, 1, 1],
                        "_ori_broadcast_axis": [0, 1],
                        "is_support_vexp_pattern": true,
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
                          "1300400500": [32],
                          "1000500610": [4],
                          "1000500611": [32]
                        },
                        "_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_normal_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_attr_vars": {
                          "1000400500": [],
                          "1000400501": [],
                          "1200400500": [],
                          "1300400500": [],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_custom_vars": {
                          "1000400500": ["mean_cof"],
                          "1000400501": ["mean_cof"],
                          "1200400500": ["mean_cof"],
                          "1300400500": ["mean_cof"],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_norm_vars": {
                          "1000400500": [20000, 20001, 30000, 40000],
                          "1000400501": [20000, 20001, 30000, 40000],
                          "1200400500": [20000, 20001, 30000, 40000],
                          "1300400500": [20000, 20001, 30000, 40000],
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

TEST_F(LayerNormTiling, LayerNorm_Norm_tiling_test_3) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_ori_reduce_axis": [3],
                        "_input_type": [0, 1, 1],
                        "_ori_broadcast_axis": [0, 1],
                        "is_support_vexp_pattern": true,
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
                          "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_normal_vars": {
                          "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "1000500610": ["_dim_1", "_block_factor", "_ub_factor"],
                          "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_attr_vars": {
                          "1000400500": [],
                          "1000400501": [],
                          "1200400500": [],
                          "1300400500": [],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_custom_vars": {
                          "1000400500": ["mean_cof"],
                          "1000400501": ["mean_cof"],
                          "1200400500": ["mean_cof"],
                          "1300400500": ["mean_cof"],
                          "1000500610": [],
                          "1000500611": []
                        },
                        "_norm_vars": {
                          "1000400500": [20000, 20001, 30000, 40000],
                          "1000400501": [20000, 20001, 30000, 40000],
                          "1200400500": [20000, 20001, 30000, 40000],
                          "1300400500": [20000, 20001, 30000, 40000],
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

// old special host tiling testcase
TEST_F(LayerNormTiling, LayerNorm_tiling_test_1) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "input_format": "NCHW",
                        "is_support_vexp_pattern": false,
                        "core_num": 32,
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                        
                        "is_tik_support":true,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952})";

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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 132 512 27 5 2 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_2) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1480005": [], "1540005": [], "2180005": [], "2240005": [], "390005": [],
                          "1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": [], "1480001": [], "1540001": [], "2180001": [], "2240001": [], "390001": [], "1480002": [], "1540002": [], "2180002": [], "2240002": [], "390002": []},
                        "_custom_vars": {
                        "1480005": ["mean_Cof"],
                        "1540005": ["mean_Cof"],
                        "2180005": ["mean_Cof"],
                        "2240005": ["mean_Cof"],
                        "390005": ["mean_Cof"],
                        "1480000": ["mean_Cof"],
                        "1540000": ["mean_Cof"],
                        "2180000": ["mean_Cof"],
                        "2240000": ["mean_Cof"],
                        "390000": ["mean_Cof"],
                        "1480001": ["mean_Cof"],
                        "1540001": ["mean_Cof"],
                        "2180001": ["mean_Cof"],
                        "2240001": ["mean_Cof"],
                        "390001": ["mean_Cof"],
                        "1480002": ["mean_Cof"],
                        "1540002": ["mean_Cof"],
                        "2180002": ["mean_Cof"],
                        "2240002": ["mean_Cof"],
                        "390002": ["mean_Cof"]},
                        "_normal_vars": {
                        "1480005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1540005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2180005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2240005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "390005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1540000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2180000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390002":["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1480005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1540005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2180005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2240005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "390005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1540000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2180000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390002":["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [39],
                        "reduce_axis": [1,2],
                        "input_format": "NCHW",
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{1024, 30, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{1024, 30, 512}, {1024, 1, 1}, {1024, 1, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 1480001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1024 30 512 32 1 20 23 948471945 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_3) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "input_format": "NCHW",
                        "is_support_vexp_pattern": false,
                        "core_num": 32,
                        "begin_norm_axis":1,
                        "begin_params_axis":1,
                        "is_tik_support":true,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",  
                        "ub_max_byte": 253952})";

  std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {309, 512}, {309, 512}};
  std::vector<std::vector<int64_t>> outputs{{34, 309, 512}, {34, 1, 1}, {34, 1, 1}};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 34 158208 17 2 2 919869235 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_4) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":0,
                        "begin_params_axis":-1,  
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1260005": [],"1320005": [],"1380005": [],"1960005": [],"2020005": [],"2080005": [],"2660005": [],"2720005": [],"2780005": [],"1260000": [],"1320000": [],"1380000": [],"1960000": [],"2020000": [],"2080000": [],"2660000": [],"2720000": [],"2780000": [], "1260001": [],"1320001": [],"1380001": [],"1960001": [],"2020001": [],"2080001": [],"2660001": [],"2720001": [],"2780001": [], "1260002": [],"1320002": [],"1380002": [],"1960002": [],"2020002": [],"2080002": [],"2660002": [],"2720002": [],"2780002": []},
                        "_custom_vars": {
                        "1260005": ["mean_cof"],
                        "1320005": ["mean_cof"],
                        "1380005": ["mean_cof"],
                        "1960005": ["mean_cof"],
                        "2020005": ["mean_cof"],
                        "2080005": ["mean_cof"],
                        "2660005": ["mean_cof"],
                        "2720005": ["mean_cof"],
                        "2780005": ["mean_cof"],
                        "1260000": ["mean_cof"],
                        "1320000": ["mean_cof"],
                        "1380000": ["mean_cof"],
                        "1960000": ["mean_cof"],
                        "2020000": ["mean_cof"],
                        "2080000": ["mean_cof"],
                        "2660000": ["mean_cof"],
                        "2720000": ["mean_cof"],
                        "2780000": ["mean_cof"],
                        "1260001": ["mean_cof"],
                        "1320001": ["mean_cof"],
                        "1380001": ["mean_cof"],
                        "1960001": ["mean_cof"],
                        "2020001": ["mean_cof"],
                        "2080001": ["mean_cof"],
                        "2660001": ["mean_cof"],
                        "2720001": ["mean_cof"],
                        "2780001": ["mean_cof"],
                        "1260002": ["mean_cof"],
                        "1320002": ["mean_cof"],
                        "1380002": ["mean_cof"],
                        "1960002": ["mean_cof"],
                        "2020002": ["mean_cof"],
                        "2080002": ["mean_cof"],
                        "2660002": ["mean_cof"],
                        "2720002": ["mean_cof"],
                        "2780002": ["mean_cof"]},
                        "_normal_vars": {
                        "1260005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1260000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1260001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1260002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1260005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1260000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1260001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1260002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1320002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1380002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1960002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2020002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2080002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2660002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2720002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "2780002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [63],
                        "reduce_axis": [0,1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{34, 309, 512}, {1, 1, 1}, {1, 1, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 2020001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "34 309 512 34 1 20 34 877108573 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_5) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1480005": [], "1540005": [], "2180005": [], "2240005": [], "390005": [],"1480000": [], "1540000": [], "2180000": [], "2240000": [], "390000": [], "1480001": [], "1540001": [], "2180001": [], "2240001": [], "390001": [], "1480002": [], "1540002": [], "2180002": [], "2240002": [], "390002": []},
                        "_custom_vars": {
                        "1480005": ["mean_Cof"],
                        "1540005": ["mean_Cof"], 
                        "2180005": ["mean_Cof"], 
                        "2240005": ["mean_Cof"],
                        "390005": ["mean_Cof"],
                        "1480000": ["mean_Cof"],
                        "1540000": ["mean_Cof"], 
                        "2180000": ["mean_Cof"],
                        "2240000": ["mean_Cof"], 
                        "390000": ["mean_Cof"],
                        "1480001": ["mean_Cof"],
                        "1540001": ["mean_Cof"],
                        "2180001": ["mean_Cof"], 
                        "2240001": ["mean_Cof"],
                        "390001": ["mean_Cof"],
                        "1480002": ["mean_Cof"],
                        "1540002": ["mean_Cof"], 
                        "2180002": ["mean_Cof"], 
                        "2240002": ["mean_Cof"],
                        "390002": ["mean_Cof"]},
                        "_normal_vars": {
                        "1480005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1480005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1480002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "1540002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2180002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "2240002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "390002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [39],
                        "reduce_axis": [1,2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{34, 309, 512}, {34, 1, 1}, {34, 1, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 5);
  EXPECT_EQ(runInfo.GetTilingKey(), 1480001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "34 309 512 8 1 20 8 919869235 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_6) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis": -1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["mean_Cof"], 
                        "270005": ["mean_Cof"], 
                        "670005": ["mean_Cof"], 
                        "671005": ["mean_Cof"],
                        "1940000": ["mean_Cof"], 
                        "270000": ["mean_Cof"], 
                        "670000": ["mean_Cof"], 
                        "671000": ["mean_Cof"],
                        "1940001": ["mean_Cof"], 
                        "270001": ["mean_Cof"], 
                        "670001": ["mean_Cof"], 
                        "671001": ["mean_Cof"],
                        "1940002": ["mean_Cof"], 
                        "270002": ["mean_Cof"], 
                        "670002": ["mean_Cof"], 
                        "671002": ["mean_Cof"]},
                        "_normal_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{20, 304, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{20, 304, 512}, {20, 304, 1}, {20, 304, 1}};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 671001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "20 304 512 2 16 19 0 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_7) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",             
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["mean_Cof"], 
                        "270005": ["mean_Cof"], 
                        "670005": ["mean_Cof"], 
                        "671005": ["mean_Cof"],
                        "1940000": ["mean_Cof"], 
                        "270000": ["mean_Cof"], 
                        "670000": ["mean_Cof"], 
                        "671000": ["mean_Cof"],
                        "1940001": ["mean_Cof"], 
                        "270001": ["mean_Cof"], 
                        "670001": ["mean_Cof"], 
                        "671001": ["mean_Cof"],
                        "1940002": ["mean_Cof"], 
                        "270002": ["mean_Cof"], 
                        "670002": ["mean_Cof"], 
                        "671002": ["mean_Cof"]},
                        "_normal_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{49, 304, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{49, 304, 512}, {49, 304, 1}, {49, 304, 1}};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 784);
  EXPECT_EQ(runInfo.GetTilingKey(), 671001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "49 304 512 49 16 19 0 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_8) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis": -1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["mean_Cof"], 
                        "270005": ["mean_Cof"], 
                        "670005": ["mean_Cof"], 
                        "671005": ["mean_Cof"],
                        "1940000": ["mean_Cof"], 
                        "270000": ["mean_Cof"], 
                        "670000": ["mean_Cof"], 
                        "671000": ["mean_Cof"],
                        "1940001": ["mean_Cof"], 
                        "270001": ["mean_Cof"], 
                        "670001": ["mean_Cof"], 
                        "671001": ["mean_Cof"],
                        "1940002": ["mean_Cof"], 
                        "270002": ["mean_Cof"], 
                        "670002": ["mean_Cof"], 
                        "671002": ["mean_Cof"]},
                        "_normal_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 8, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{34, 309, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{34, 304, 512}, {34, 304, 1}, {34, 304, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 671001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "34 309 512 2 16 20 0 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_9) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [], "1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["mean_Cof"], 
                        "270005": ["mean_Cof"], 
                        "670005": ["mean_Cof"], 
                        "671005": ["mean_Cof"],
                        "1940000": ["mean_Cof"], 
                        "270000": ["mean_Cof"], 
                        "670000": ["mean_Cof"], 
                        "671000": ["mean_Cof"],
                        "1940001": ["mean_Cof"], 
                        "270001": ["mean_Cof"], 
                        "670001": ["mean_Cof"], 
                        "671001": ["mean_Cof"],
                        "1940002": ["mean_Cof"], 
                        "270002": ["mean_Cof"], 
                        "670002": ["mean_Cof"], 
                        "671002": ["mean_Cof"]},
                        "_normal_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{352, 4, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{352, 4, 512}, {352, 4, 1}, {352, 4, 1}};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 270001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "352 4 512 352 1 5 0 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_10) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                       
                        "ub_max_byte": 253952,
                        "_attr_vars": {"1940005": [], "270005": [], "670005": [], "671005": [],"1940000": [], "270000": [], "670000": [], "671000": [], "1940001": [], "270001": [], "670001": [], "671001": [], "1940002": [], "270002": [], "670002": [], "671002": []},
                        "_custom_vars": {
                        "1940005": ["mean_cof"], 
                        "270005": ["mean_cof"], 
                        "670005": ["mean_cof"], 
                        "671005": ["mean_cof"],
                        "1940000": ["mean_cof"], 
                        "270000": ["mean_cof"], 
                        "670000": ["mean_cof"], 
                        "671000": ["mean_cof"],
                        "1940001": ["mean_cof"], 
                        "270001": ["mean_cof"], 
                        "670001": ["mean_cof"], 
                        "671001": ["mean_cof"],
                        "1940002": ["mean_cof"], 
                        "270002": ["mean_cof"], 
                        "670002": ["mean_cof"], 
                        "671002": ["mean_cof"]},
                        "_normal_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671005": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "1940002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "270002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "670002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "671002": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "NCHW",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{32, 121, 768}, {768}, {768}};
  std::vector<std::vector<int64_t>> outputs{{32, 121, 768}, {32, 121, 1}, {32, 121, 768}};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 1940001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 121 768 1 1 768 1 984263339 ");
}

// static shape case
TEST_F(LayerNormTiling, LayerNorm_tiling_test_11) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"671001": []},
                        "_custom_vars": {
                        "671001": []},
                        "_normal_vars": {"671001": ["_dim0_0", "_dim0_1", "_dim0_2"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "671001": ["_dim0_0", "_dim0_1", "_dim0_2"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "const",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "input_format": "NCHW",
                        "is_support_vexp":true,
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{20, 304, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{20, 304, 512}, {20, 304, 1}, {20, 304, 1}};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 671001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "20 304 512 0 1 1 0 2 16 19 0 1 ");
}

// NZ case
TEST_F(LayerNormTiling, LayerNorm_tiling_test_12) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "dynamic",                        
                        "ub_max_byte": 253952,
                        "_attr_vars": {"898000": [], "9040000": [], "11080000": [], "11140000": [], "5390000": [], "5791000": [], "5792000": []},
                        "_custom_vars": {
                        "898000": ["mean_cof"], 
                        "9040000": ["mean_cof"], 
                        "11080000": ["mean_cof"], 
                        "11140000": ["mean_cof"],
                        "5390000": ["mean_cof"], 
                        "5791000": ["mean_cof"], 
                        "5792000": ["mean_cof"]},
                        "_normal_vars": {
                        "898000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "9040000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "11080000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "11140000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "5390000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "5791000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "5792000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "898000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "9040000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "11080000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "11140000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"],
                        "5390000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "5791000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"], 
                        "5792000": ["_dim0_0", "_dim0_1", "_dim0_2", "_block_factor", "_block_factor_1", "_ub_factor", "_ub_fuse_factor"]},
                        "common_info": [32, 1, 16, 0],
                        "core_num": 32,
                        "max_ub_size_normal_fp16": 10240,
                        "max_ub_size_normal_fp32": 10240,
                        "mode": "original",
                        "pattern_info": [99],
                        "reduce_axis": [0, 3],
                        "reduce_mean_cof_dtype":"float32",
                        "is_support_vexp":true,
                        "input_format": "FRACTAL_NZ",
                        "ub_info":[16384]})";

  std::vector<std::vector<int64_t>> inputs{{32, 32, 16, 16}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{32, 32, 16, 16}, {512, 1}, {512, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  ge::Format data_format = ge::FORMAT_FRACTAL_NZ;

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
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 5391000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 32 16 16 1 1 1 0 989855744 ");
}

TEST_F(LayerNormTiling, LayerNorm_tiling_test_13) {
  // tik case
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "input_format": "NCHW",
                        "is_support_vexp_pattern": false,
                        "core_num": 32,
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,                       
                        "is_tik_support":true,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "const",                       
                        "ub_max_byte": 253952})";

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
  EXPECT_EQ(runInfo.GetBlockDim(), 27);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 132 512 27 5 2 989855744 ");
}

// static shape case
TEST_F(LayerNormTiling, LayerNorm_tiling_test_14) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "const",                        
                        "ub_max_byte": 245760,
                        "is_support_vexp":false,
                        "_attr_vars": {"1940001": []},
                        "_custom_vars": {
                        "1940001": []},
                        "_normal_vars": {
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2"]},
                        "_pattern": "LayerNorm", 
                        "_vars": {
                        "1940001": ["_dim0_0", "_dim0_1", "_dim0_2"]},
                        "common_info": [2, 1, 16, 0],
                        "core_num": 2,
                        "max_ub_size_normal_fp16": 8192,
                        "max_ub_size_normal_fp32": 8192,
                        "mode": "const",
                        "pattern_info": [27],
                        "reduce_axis": [2],
                        "input_format": "NCHW",
                        "ub_info":[31744]})";

  std::vector<std::vector<int64_t>> inputs{{44, 545, 780}, {780}, {780}};
  std::vector<std::vector<int64_t>> outputs{{44, 545, 780}, {44, 545, 1}, {44, 545, 1}};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 4);
  EXPECT_EQ(runInfo.GetTilingKey(), 1940001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "44 545 780 0 0 2 0 11 1 784 11 0 ");
}

// static shape case
TEST_F(LayerNormTiling, LayerNorm_tiling_test_15) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({
                        "begin_norm_axis":-1,
                        "begin_params_axis":-1,
                        "is_support_vexp_pattern": false,
                        "is_tik_support":false,
                        "atomic_clean_diff_shape":false,
                        "tik_mode": "const",
                        "ub_max_byte": 245760,
                        "is_support_vexp":false,
                        "_attr_vars": {"1000002": []},
                        "_custom_vars": {
                        "1000002": []},
                        "_normal_vars": {
                        "1000002": ["_dim0_0", "_dim0_1"]},
                        "_pattern": "LayerNorm",
                        "_vars": {
                        "1000002": ["_dim0_0", "_dim0_1"]},
                        "common_info": [2, 1, 8, 0],
                        "core_num": 2,
                        "max_ub_size_normal_fp16": 8192,
                        "max_ub_size_normal_fp32": 8192,
                        "mode": "const",
                        "pattern_info": [15],
                        "reduce_axis": [1],
                        "input_format": "NCHW",
                        "ub_info":[15872]})";

  std::vector<std::vector<int64_t>> inputs{{16400, 8200}, {8200}, {8200}};
  std::vector<std::vector<int64_t>> outputs{{16400, 8200}, {16400, 1}, {16400, 1}};
  std::vector<ge::DataType> input_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> output_types{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
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
  EXPECT_EQ(runInfo.GetBlockDim(), 4);
  EXPECT_EQ(runInfo.GetTilingKey(), 1000002);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16400 8200 0 0 1 0 4100 1 8192 4091 0 ");
}

// static shape case
TEST_F(LayerNormTiling, LayerNorm_tiling_test_unknown_rank) {
  std::string op_name = "LayerNorm";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = R"({"is_support_vexp_pattern": true, "unknown_mode": true, "_fuse_axis": true, "_input_type": [0, 1, 1], "_reduce_axis_type": 3, "_broadcast_axis_type_list": [2], "_ori_broadcast_axis": [0, 1], "is_support_vexp": true, "reduce_mean_cof_dtype": "float16", "_pattern": "Norm", "_common_info": [32, 16, 256], "_exist_output_after_reduce": true, "_available_ub_size": {"2001": [32560, 32720, 26048], "4005": [32560, 32720, 32560], "5006": [32688, 26176, 26144]}, "_exist_workspace_after_reduce": false, "_workspace_info": {"1000200190": [2], "1000200199": [32], "1300200199": [32], "1000400500": [32], "1000400501": [2], "1200400500": [32], "1300400500": [32], "1000500610": [2], "1000500611": [32]}, "_vars": {"1000200190": ["_dim_0", "_ub_factor", "mean_cof"], "1000200199": ["_dim_0", "mean_cof"], "1300200199": ["_dim_0", "mean_cof"], "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"], "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"], "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"], "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof"], "1000500610": ["_dim_1", "_block_factor", "_ub_factor"], "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]}, "_normal_vars": {"1000200190": ["_dim_0", "_ub_factor"], "1000200199": ["_dim_0"], "1300200199": ["_dim_0"], "1000400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400501": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1200400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1300400500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000500610": ["_dim_1", "_block_factor", "_ub_factor"], "1000500611": ["_dim_1", "_block_factor", "_ub_factor"]}, "_attr_vars": {"1000200190": [], "1000200199": [], "1300200199": [], "1000400500": [], "1000400501": [], "1200400500": [], "1300400500": [], "1000500610": [], "1000500611": []}, "_custom_vars": {"1000200190": ["mean_cof"], "1000200199": ["mean_cof"], "1300200199": ["mean_cof"], "1000400500": ["mean_cof"], "1000400501": ["mean_cof"], "1200400500": ["mean_cof"], "1300400500": ["mean_cof"], "1000500610": [], "1000500611": []}, "_norm_vars": {"1000200190": [20000, 40000], "1000200199": [20000], "1300200199": [20000], "1000400500": [20000, 20001, 30000, 40000], "1000400501": [20000, 20001, 30000, 40000], "1200400500": [20000, 20001, 30000, 40000], "1300400500": [20000, 20001, 30000, 40000], "1000500610": [20001, 30000, 40000], "1000500611": [20001, 30000, 40000]}})";

  std::vector<std::vector<int64_t>> inputs{{20, 304, 512}, {512}, {512}};
  std::vector<std::vector<int64_t>> outputs{{20, 304, 512}, {20, 304, 1}, {20, 304, 1}};
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
}

