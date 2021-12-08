#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "reduce_ops.h"
#include "array_ops.h"

using namespace std;

class BNReduceGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "BnReduceGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "BnReduceGradTiling TearDown" << std::endl;
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

static std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4]], "push_status": 0, "_pattern": "Broadcast",
                          "_flag_info": [false, false, true, true, false, false, false],
                          "_outs_uint1": false,
                          "reduce_mean_cof_dtype": "float32",
                          "_base_info": {
                          "100": [32, 4, 10920, 5456],
                          "121": [32, 4, 10920, 5456],
                          "210": [32, 4, 10920, 5456],
                          "000": [32, 4, 10920, 5456]},
                          "_elewise_vars": {
                          "210000000": [10000, 30000, 30000, 20000, 300000],
                          "210010000": [10000, 30000, 30000, 20000, 300000],
                          "212100000": [10000, 10100, 10101, 10200, 30000, 30000, 20000, 300000],
                          "212100001": [10000, 10100, 10101, 10200, 30000, 30000, 20000, 300000],
                          "212100002": [10000, 10100, 10101, 10200, 30000, 30000, 20000, 300000],
                          "212100003": [10000, 10100, 10101, 10200, 30000, 30000, 20000, 300000],
                          "212100005": [10000, 10100, 10101, 10200, 30000, 30000, 20000, 300000],
                          "212100006": [10000, 10100, 10101, 10200, 30000, 30000, 20000, 300000],
                          "212100009": [10000, 10100, 10101, 10200, 30000, 30000, 20000, 300000],
                          "221000000": [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10100, 30000, 30000],
                          "221000001": [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10100, 30000, 30000, 20000, 30000],
                          "221000002": [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10100, 30000, 30000, 20000, 30000],
                          "221000004": [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10100, 30000, 30000, 20000, 30000],
                          "0": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000],
                          "1": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20000, 30000],
                          "2": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20000, 30001],
                          "3": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20000, 30002],
                          "4": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20000, 30003],
                          "5": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20000, 30004],
                          "7": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20001, 30001],
                          "8": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20001, 30002],
                          "9": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20001, 30003],
                          "10": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20001, 30004],
                          "13": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20002, 30002],
                          "14": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20002, 30003],
                          "15": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20002, 30004],
                          "19": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20003, 30003],
                          "20": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20003, 30004],
                          "25": [10000, 10001, 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10200, 10201, 10300, 10301, 30000, 30000, 20004, 30004]}
                          })";



using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(BNReduceGradTiling, BnReduceGradTiling1) {
  std::string op_name = "BNTrainingReduceGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::vector<int64_t> input_x{128, 16, 20, 20, 16};
  std::vector<int64_t> input_scaler{1, 16, 1, 1, 16};
  std::vector<int64_t> output{128, 16, 20, 20, 16};
  ge::DataType in_dtype = ge::DT_FLOAT16;
  
  auto opParas = op::BNTrainingReduceGrad("BNTrainingReduceGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grads, input_x, in_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_x, in_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, diff_scale, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, diff_offset, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, scale, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, batch_mean, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, batch_variance, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "128 128 16 16 16 16 16 16 16 20 20 20 20 1 1 4 1 933484298 -1213999350 ");
}

TEST_F(BNReduceGradTiling, BnReduceGradTiling2) {
  std::string op_name = "BNTrainingReduceGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::vector<int64_t> input_x{128, 16, 20, 20, 16};
  std::vector<int64_t> input_scaler{1, 16, 1, 1, 16};
  std::vector<int64_t> output{128, 16, 20, 20, 16};
  ge::DataType in_dtype = ge::DT_FLOAT;
  
  auto opParas = op::BNTrainingReduceGrad("BNTrainingReduceGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, grads, input_x, in_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_x, in_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, diff_scale, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, diff_offset, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, scale, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, batch_mean, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, batch_variance, input_scaler, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "128 128 16 16 16 16 16 16 16 20 20 20 20 1 1 4 1 933484298 -1213999350 ");
}