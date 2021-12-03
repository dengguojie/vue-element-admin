#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace std;

class SmoothL1LossV2TilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SmoothL1LossV2TilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SmoothL1LossV2TilingTest TearDown" << std::endl;
  }
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

using namespace ge;
using namespace ut_util;

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossV2("SmoothL1LossV2");

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, loss, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "mean","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float32","_pattern": "CommReduce","_common_info": [32, 1, 16, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [10752, 9088],"_ub_info": [10752, 10624],"_idx_before_reduce": 0, "_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor", "cof"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor", "cof"], "2147483647": ["_dim_1", "cof"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof"]}, "_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}, "_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []}, "_custom_vars":{"-1000500":["cof"],"-1100500":["cof"],"2147483647":["cof"],"-400":["cof"],"-100400":["cof"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 4000 1 1 964891246 ");
}

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossV2("SmoothL1LossV2");

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, loss, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "sum","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float32","_pattern": "CommReduce","zero_ub_factor": 10624,"_common_info": [32, 1, 16, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [10752, 9088],"_ub_info": [10752, 10624],"_idx_before_reduce": 0,"_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []},"_custom_vars":{"-1000500":[],"-1100500":[],"2147483647":[],"-400":[],"-100400":[]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 4000 1 1 964891246 ");
}

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossV2("SmoothL1LossV2");

  vector<vector<int64_t>> input_shapes = {
      {100, 50, 40},
      {100, 50, 40}
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, loss, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "none","reduce_mean_cof_dtype": "float32","_pattern": "ElemWise","_outs_uint1": false,"_flag_info": [false, false, false, true, false, false, false],"_base_info": {"100": [261760, 4, 4, 32]},"_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000]},"_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]},"_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]},"_attr_vars": {"210000000": [], "210010000": []},"_custom_vars": {"210000000": [], "210010000": []}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "200000 200000 200000 ");
}


TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossV2("SmoothL1LossV2");

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, loss, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "sum","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float16","_pattern": "CommReduce","zero_ub_factor": 10624,"_common_info": [2, 1, 256, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [17792, 15360],"_ub_info": [17792, 17536],"_idx_before_reduce": 0,"_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []},"_custom_vars":{"-1000500":[],"-1100500":[],"2147483647":[],"-400":[],"-100400":[]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 4000 1 1 3097 ");
}

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SmoothL1LossV2("SmoothL1LossV2");

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, predict, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, label, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, loss, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo = R"({"reduction": "mean","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float16","_pattern": "CommReduce","_common_info": [2, 1, 256, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [17792, 15360],"_ub_info": [17792, 17536],"_idx_before_reduce": 0, "_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"], "2147483647": ["_dim_1", "cof", "cof_empty"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"]}, "_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}, "_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []}, "_custom_vars":{"-1000500":["cof", "cof_empty"],"-1100500":["cof", "cof_empty"],"2147483647":["cof", "cof_empty"],"-400":["cof", "cof_empty"],"-100400":["cof", "cof_empty"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 4000 1 1 3097 ");
}