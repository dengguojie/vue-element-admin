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

/*
.INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
.INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
.INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16}))
.INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16}))
.INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16}))
.OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16}))
.OUTPUT(res_for_gamma, TensorType({DT_FLOAT}))
.OP_END_FACTORY_REG(LayerNormXBackpropV2)
*/

TEST_F(LayerNormXBackpropV2Tiling, LayerNormXBackpropV2_tiling_test_1) {
  std::string op_name = "LayerNormXBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::LayerNormXBackpropV2(op_name.c_str());

  vector<vector<int64_t>> input_shapes = {{30, 496, 456}, {30, 496, 456}, {30, 496, 1}, {30, 496, 1}, {456}};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  // input_dy, input_x, input_variance, input_mean,
  // input_gamma, output_pd_x, output_res_gamma

  TENSOR_INPUT_WITH_SHAPE(opParas, dy, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, variance, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, mean, input_shapes[3], dtypes[3], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, gamma, input_shapes[4], dtypes[4], ge::FORMAT_ND, {});

  vector<vector<int64_t>> output_shapes = {{30, 496, 456}, {30, 496, 456}};
  TENSOR_OUTPUT_WITH_SHAPE(opParas, pd_x, output_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, res_for_gamma, output_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_input_type": [0, 0, 1, 1, 2],
                        "_ori_reduce_axis": [2],
                        "reduce_mean_cof": true,
                        "_pattern": "Norm",
                        "_common_info": [32, 8, 128],
                        "_exist_output_after_reduce": false,
                        "_available_ub_size": {"4012": [9040, 10888, 9008, 8168]},
                        "_block_size": {"4012": 8},
                        "_exist_workspace_after_reduce": false,
                        "_workspace_info": {
                          "401200": [32, 32],
                          "401201": [4, 4],
                          "200401200": [32, 32],
                          "300401200": [32, 32],
                          "400401200": [32, 32]
                        },
                        "_vars": {
                          "401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "401201": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "200401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "300401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "400401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"]
                        },
                        "_normal_vars": {
                          "401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "401201": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "200401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "300401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "400401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_custom_vars": {
                          "401200": ["mean_cof", "mean_cof_double"],
                          "401201": ["mean_cof", "mean_cof_double"],
                          "200401200": ["mean_cof", "mean_cof_double"],
                          "300401200": ["mean_cof", "mean_cof_double"],
                          "400401200": ["mean_cof", "mean_cof_double"]
                        },
                        "_norm_vars": {
                          "401200": [20000, 20001, 30000, 40000],
                          "401201": [20000, 20001, 30000, 40000],
                          "200401200": [20000, 20001, 30000, 40000],
                          "300401200": [20000, 20001, 30000, 40000],
                          "400401200": [20000, 20001, 30000, 40000]
                        }})";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 300401200);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "14880 456 465 19 990885924 999274532 ");
}

TEST_F(LayerNormXBackpropV2Tiling, LayerNormXBackpropV2_tiling_test_2) {
  std::string op_name = "LayerNormXBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  vector<vector<int64_t>> input_shapes = {{64, 114, 16, 16}, {64, 114, 16, 16}, {1824, 1}, {1824, 1}, {1024}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_input_type": [0, 0, 1, 1, 2],
                        "_ori_reduce_axis": [0, 3],
                        "reduce_mean_cof": true,
                        "_pattern": "Norm",
                        "_common_info": [32, 16, 128],
                        "_exist_output_after_reduce": false,
                        "_available_ub_size": {"12012": [10544, 8168, 9000, 8160]},
                        "_block_size": {"12012": 16},
                        "_exist_workspace_after_reduce": false,
                        "_workspace_info": {
                          "1201210": [4, 4, 32, 32],
                          "1201211": [32, 32, 32, 32],
                          "1201212": [4, 4, 32, 32],
                          "101201210": [4, 4, -4, -4],
                          "101201212": [4, 4, -4, -4],
                          "301201211": [32, 32, 32, 32]
                        },
                        "_vars": {
                          "1201210": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "1201211": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "1201212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "101201210": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "101201212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "301201211": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"]
                        },
                        "_normal_vars": {
                          "1201210": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"],
                          "1201211": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"],
                          "1201212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"],
                          "101201210": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"],
                          "101201212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"],
                          "301201211": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]
                        },
                        "_custom_vars": {
                          "1201210": ["mean_cof", "mean_cof_double"],
                          "1201211": ["mean_cof", "mean_cof_double"],
                          "1201212": ["mean_cof", "mean_cof_double"],
                          "101201210": ["mean_cof", "mean_cof_double"],
                          "101201212": ["mean_cof", "mean_cof_double"],
                          "301201211": ["mean_cof", "mean_cof_double"]
                        },
                        "_norm_vars": {
                          "1201210": [20000, 20001, 20002, 30000, 40000],
                          "1201211": [20000, 20001, 20002, 30000, 40000],
                          "1201212": [20000, 20001, 20002, 30000, 40000],
                          "101201210": [20000, 20001, 20002, 30000, 40000],
                          "101201212": [20000, 20001, 20002, 30000, 40000],
                          "301201211": [20000, 20001, 20002, 30000, 40000]
                        }})";

  vector<vector<int64_t>> inputs = {{64, 114, 16, 16}, {64, 114, 16, 16}, {1824, 1}, {1824, 1}, {1024}};
  vector<ge::DataType> input_types = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  vector<vector<int64_t>> outputs = {{64, 1824, 16}, {64, 1824, 16}};
  vector<ge::DataType> output_types = {ge::DT_FLOAT16, ge::DT_FLOAT};
  
  ge::Format data_format1 = ge::FORMAT_FRACTAL_NZ;
  ge::Format data_format2 = ge::FORMAT_ND;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputs[0]));
  tensor_inputA.SetDataType(input_types[0]);
  tensor_inputA.SetFormat(data_format1);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputs[1]));
  tensor_inputB.SetDataType(input_types[1]);
  tensor_inputB.SetFormat(data_format1);
  TensorDesc tensor_inputC;
  tensor_inputC.SetShape(ge::Shape(inputs[2]));
  tensor_inputC.SetDataType(input_types[2]);
  tensor_inputC.SetFormat(data_format2);
  TensorDesc tensor_inputD;
  tensor_inputC.SetShape(ge::Shape(inputs[2]));
  tensor_inputC.SetDataType(input_types[2]);
  tensor_inputC.SetFormat(data_format2);
  TensorDesc tensor_inputE;
  tensor_inputC.SetShape(ge::Shape(inputs[2]));
  tensor_inputC.SetDataType(input_types[2]);
  tensor_inputC.SetFormat(data_format2);

  TensorDesc tensor_outputA;
  tensor_outputA.SetShape(ge::Shape(outputs[0]));
  tensor_outputA.SetDataType(output_types[0]);
  tensor_outputA.SetFormat(data_format2);
  TensorDesc tensor_outputB;
  tensor_outputB.SetShape(ge::Shape(outputs[1]));
  tensor_outputB.SetDataType(output_types[1]);
  tensor_outputB.SetFormat(data_format2);

  auto opParas = op::LayerNormXBackpropV2(op_name.c_str());
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
  EXPECT_EQ(runInfo.GetTilingKey(), 301201211);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "64 1824 16 57 10 981467136 989855744 ");
}

TEST_F(LayerNormXBackpropV2Tiling, LayerNormXBackpropV2_tiling_test_3) {
  std::string op_name = "LayerNormXBackpropV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::LayerNormXBackpropV2(op_name.c_str());

  vector<vector<int64_t>> input_shapes = {{30, 496, 456}, {30, 496, 456}, {30, 496, 1}, {30, 496, 1}, {456}};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  // input_dy, input_x, input_variance, input_mean,
  // input_gamma, output_pd_x, output_res_gamma

  TENSOR_INPUT_WITH_SHAPE(opParas, dy, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, variance, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, mean, input_shapes[3], dtypes[3], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, gamma, input_shapes[4], dtypes[4], ge::FORMAT_ND, {});

  vector<vector<int64_t>> output_shapes = {{30, 496, 456}, {30, 496, 456}};
  TENSOR_OUTPUT_WITH_SHAPE(opParas, pd_x, output_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, res_for_gamma, output_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  
  std::string compileInfo = R"({
                        "_fuse_axis": true,
                        "_input_type": [0, 0, 1, 1, 2],
                        "_ori_reduce_axis": [2],
                        "reduce_mean_cof": true,
                        "unknown_mode": true,
                        "_pattern": "Norm",
                        "_common_info": [32, 8, 128],
                        "_exist_output_after_reduce": false,
                        "_available_ub_size": {"4012": [9040, 10888, 9008, 8168]},
                        "_block_size": {"4012": 8},
                        "_exist_workspace_after_reduce": false,
                        "_workspace_info": {
                          "401200": [32, 32],
                          "401201": [4, 4],
                          "200401200": [32, 32],
                          "300401200": [32, 32],
                          "400401200": [32, 32]
                        },
                        "_vars": {
                          "401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "401201": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "200401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "300401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"],
                          "400401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "mean_cof", "mean_cof_double"]
                        },
                        "_normal_vars": {
                          "401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "401201": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "200401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "300401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"],
                          "400401200": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]
                        },
                        "_custom_vars": {
                          "401200": ["mean_cof", "mean_cof_double"],
                          "401201": ["mean_cof", "mean_cof_double"],
                          "200401200": ["mean_cof", "mean_cof_double"],
                          "300401200": ["mean_cof", "mean_cof_double"],
                          "400401200": ["mean_cof", "mean_cof_double"]
                        },
                        "_norm_vars": {
                          "401200": [20000, 20001, 30000, 40000],
                          "401201": [20000, 20001, 30000, 40000],
                          "200401200": [20000, 20001, 30000, 40000],
                          "300401200": [20000, 20001, 30000, 40000],
                          "400401200": [20000, 20001, 30000, 40000]
                        }})";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
