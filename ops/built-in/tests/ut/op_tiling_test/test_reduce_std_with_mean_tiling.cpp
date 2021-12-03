#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace std;

class ReduceStdWithMeanTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReduceStdWithMeanTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReduceStdWithMeanTiling TearDown" << std::endl;
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

TEST_F(ReduceStdWithMeanTiling, ReduceStdWithMeanTiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ReduceStdWithMean");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ReduceStdWithMean("ReduceStdWithMean");

  vector<vector<int64_t>> input_shapes = {
      {7, 2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, {7}, ge::DT_FLOAT, ge::FORMAT_ND, {}); 
  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32", "attr_unbiased":"false"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 2 8 8 1056964608 ");
}

TEST_F(ReduceStdWithMeanTiling, ReduceStdWithMeanTiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ReduceStdWithMean");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ReduceStdWithMean("ReduceStdWithMean");

  vector<vector<int64_t>> input_shapes = {
      {7, 2},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, {7}, ge::DT_FLOAT16, ge::FORMAT_ND, {}); 
  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [2, 1, 16, 0, 1], "_pattern_info": [5, 4, 9], "_ub_info": [31488, 31104, 31488], "_ub_info_rf": [31488, 31104, 31488], "reduce_mean_cof_dtype": "float16", "attr_unbiased":"false"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 2 16 16 14336 ");
}

TEST_F(ReduceStdWithMeanTiling, ReduceStdWithMeanTiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ReduceStdWithMean");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ReduceStdWithMean("ReduceStdWithMean");

  vector<vector<int64_t>> input_shapes = {
      {3, 4, 5},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, {3}, ge::DT_FLOAT, ge::FORMAT_ND, {}); 
  std::string compileInfo = R"({ "_ori_axis": [1, 2], "_pattern": "CommReduce", "push_status": 0, "_common_info": [2, 1, 16, 0, 1], "_pattern_info": [5, 4, 9], "_ub_info": [31488, 31104, 31488], "_ub_info_rf": [31488, 31104, 31488], "reduce_mean_cof_dtype": "float16", "attr_unbiased":"false"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 20 16 16 10854 ");
}

TEST_F(ReduceStdWithMeanTiling, ReduceStdWithMeanTiling4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ReduceStdWithMean");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ReduceStdWithMean("ReduceStdWithMean");

  vector<vector<int64_t>> input_shapes = {
      {3, 4, 5},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, {3}, ge::DT_FLOAT16, ge::FORMAT_ND, {}); 
  std::string compileInfo = R"({ "_ori_axis": [1, 2], "_pattern": "CommReduce", "push_status": 0, "_common_info": [2, 1, 16, 0, 1], "_pattern_info": [5, 4, 9], "_ub_info": [31488, 31104, 31488], "_ub_info_rf": [31488, 31104, 31488], "reduce_mean_cof_dtype": "float16", "attr_unbiased":"false"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 20 16 16 10854 ");
}