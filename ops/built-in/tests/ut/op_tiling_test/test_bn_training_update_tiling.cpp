#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "reduce_ops.h"
#define private public
#include "register/op_tiling_registry.h"
#include "array_ops.h"

using namespace std;

class BNTrainingUpdateTiling : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "BNTrainingUpdateTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "BNTrainingUpdateTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data)
{
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

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(BNTrainingUpdateTiling, BNTrainingUpdateTiling1)
{
  using namespace optiling;
  std::string op_name = "BNTrainingUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::BNTrainingUpdate(op_name.c_str());

  vector<vector<int64_t>> input_shapes = {{32, 16, 26, 26, 16}, {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16},
                                          {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                 ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, sum, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, square_sum, input_shapes[2], dtypes[2], ge::FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, scale, input_shapes[3], dtypes[3], ge::FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, offset, input_shapes[4], dtypes[4], ge::FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, mean, input_shapes[5], dtypes[5], ge::FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, variance, input_shapes[6], dtypes[6], ge::FORMAT_NC1HWC0, {});

  vector<vector<int64_t>> output_shapes = {
      {32, 16, 26, 26, 16}, {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}};
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, mean_out, output_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, variance_out, output_shapes[2], dtypes[2], ge::FORMAT_NC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, batch_mean, output_shapes[3], dtypes[3], ge::FORMAT_NC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, batch_variance, output_shapes[4], dtypes[4], ge::FORMAT_NC1HWC0, {});

  std::string compileInfo = R"({
                        "_fusion_index": [[0], [1], [2], [3], [4]],
                        "_pattern": "Broadcast", 
                        "_outs_uint1": false, 
                        "_soc_version": "Ascend910",
                        "_flag_info": [false, false, true, false, false, false, false],
                        "_base_info": {"000": [32, 4, 10920, 5456]},
                        "_vars": {
                        "0": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "num_rec", "batch_var_scaler"], 
                        "1": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_0", "num_rec", "batch_var_scaler"], 
                        "2": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_1", "num_rec", "batch_var_scaler"], 
                        "3": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_2", "num_rec", "batch_var_scaler"], 
                        "4": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_3", "num_rec", "batch_var_scaler"],
                        "5": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_4", "num_rec", "batch_var_scaler"],
                        "7": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_1", "num_rec", "batch_var_scaler"],
                        "8": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_2", "num_rec", "batch_var_scaler"],
                        "9": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_3", "num_rec", "batch_var_scaler"],
                        "10": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_4", "num_rec", "batch_var_scaler"],
                        "13": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_2", "_ub_factor_2", "num_rec", "batch_var_scaler"],
                        "14": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_2", "_ub_factor_3", "num_rec", "batch_var_scaler"],
                        "15": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_2", "_ub_factor_4", "num_rec", "batch_var_scaler"],
                        "19": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_3", "_ub_factor_3", "num_rec", "batch_var_scaler"],
                        "20": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_3", "_ub_factor_4", "num_rec", "batch_var_scaler"],
                        "25": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_4", "_ub_factor_4", "num_rec", "batch_var_scaler"]}, 
                        "_normal_vars": {
                        "0": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0"],
                        "1": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_0"], 
                        "2": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_1"], 
                        "3": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_2"], 
                        "4": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_3"], 
                        "5": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_0", "_ub_factor_4"], 
                        "7": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_1"],
                        "8": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_2"],
                        "9": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_3"],
                        "10": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_1", "_ub_factor_4"],
                        "13": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_2", "_ub_factor_2"],
                        "14": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_2", "_ub_factor_3"],
                        "15": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_2", "_ub_factor_4"],
                        "19": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_3", "_ub_factor_3"],
                        "20": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_3", "_ub_factor_4"],
                        "25": ["_dim_0_0", "_dim_1_0", "_dim_2_0", "_dim_3_0", "_dim_4_0", "_block_factor_4", "_ub_factor_4"]}, 
                        "_attr_vars": {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], 
                                       "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": []}, 
                        "_custom_vars": {
                        "0": ["num_rec", "batch_var_scaler"], 
                        "1": ["num_rec", "batch_var_scaler"], 
                        "2": ["num_rec", "batch_var_scaler"], 
                        "3": ["num_rec", "batch_var_scaler"], 
                        "4": ["num_rec", "batch_var_scaler"],
                        "5": ["num_rec", "batch_var_scaler"],
                        "7": ["num_rec", "batch_var_scaler"],
                        "8": ["num_rec", "batch_var_scaler"],
                        "9": ["num_rec", "batch_var_scaler"],
                        "10": ["num_rec", "batch_var_scaler"],
                        "13": ["num_rec", "batch_var_scaler"],
                        "14": ["num_rec", "batch_var_scaler"],
                        "15": ["num_rec", "batch_var_scaler"],
                        "19": ["num_rec", "batch_var_scaler"],
                        "20": ["num_rec", "batch_var_scaler"],
                        "25": ["num_rec", "batch_var_scaler"]},
                        "_elewise_vars": {
                        "0": [10000, 10100, 10200, 10300, 10400], 
                        "1": [10000, 10100, 10200, 10300, 10400, 20000, 30000], 
                        "2": [10000, 10100, 10200, 10300, 10400, 20000, 30001], 
                        "3": [10000, 10100, 10200, 10300, 10400, 20000, 30002], 
                        "4": [10000, 10100, 10200, 10300, 10400, 20000, 30003],
                        "5": [10000, 10100, 10200, 10300, 10400, 20000, 30004],
                        "7": [10000, 10100, 10200, 10300, 10400, 20001, 30001],
                        "8": [10000, 10100, 10200, 10300, 10400, 20001, 30002],
                        "9": [10000, 10100, 10200, 10300, 10400, 20001, 30003],
                        "10": [10000, 10100, 10200, 10300, 10400, 20001, 30004],
                        "13": [10000, 10100, 10200, 10300, 10400, 20002, 30002],
                        "14": [10000, 10100, 10200, 10300, 10400, 20002, 30003],
                        "15": [10000, 10100, 10200, 10300, 10400, 20002, 30004],
                        "19": [10000, 10100, 10200, 10300, 10400, 20003, 30003],
                        "20": [10000, 10100, 10200, 10300, 10400, 20003, 30004],
                        "25": [10000, 10100, 10200, 10300, 10400, 20004, 30004]
                        }})";

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 3);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 16 26 26 16 512 26 943842492 1065353604 ");
}
