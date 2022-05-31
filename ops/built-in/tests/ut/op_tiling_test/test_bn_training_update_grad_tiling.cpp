#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "array_ops.h"
#include "reduce_ops.h"

using namespace std;

class BNTrainingUpdateGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNTrainingUpdateGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNTrainingUpdateGradTiling TearDown" << std::endl;
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
#include "common/utils/ut_op_util.h"
using namespace ut_util;

/*
.INPUT(grads, TensorType({DT_FLOAT16,DT_FLOAT}))
.INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
.INPUT(batch_mean, TensorType({DT_FLOAT}))
.INPUT(batch_variance, TensorType({DT_FLOAT}))
.ATTR(epsilon, Float, 0.0001)
.OUTPUT(diff_scale, TensorType({DT_FLOAT}))
.OUTPUT(diff_offset, TensorType({DT_FLOAT}))
.OP_END_FACTORY_REG(BNTrainingUpdateGrad)
*/

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling1) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    auto opParas = op::BNTrainingUpdateGrad(op_name.c_str());

    vector<vector<int64_t>> input_shapes = {
        {32, 16, 13, 13, 16}, {32, 16, 13, 13, 16}, {1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}};

    vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_mean, input_shapes[2], dtypes[2], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_variance, input_shapes[3], dtypes[3], ge::FORMAT_NC1HWC0, {});

    vector<vector<int64_t>> output_shapes = {{1, 16, 1, 1, 16}, {1, 16, 1, 1, 16}};
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_scale, output_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_offset, output_shapes[1], dtypes[0], ge::FORMAT_NC1HWC0, {});

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "has_epsilon": true,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(runInfo.GetBlockDim(), 32);
    EXPECT_EQ(runInfo.GetTilingKey(), 13400);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "953267991 32 16 13 13 1 1 ");
}

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling2) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    auto opParas = op::BNTrainingUpdateGrad(op_name.c_str());

    vector<vector<int64_t>> input_shapes = {
        {256, 32, 28, 28, 16}, {256, 32, 28, 28, 16}, {1, 32, 1, 1, 16}, {1, 32, 1, 1, 16}};

    vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_mean, input_shapes[2], dtypes[2], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_variance, input_shapes[3], dtypes[3], ge::FORMAT_NC1HWC0, {});

    vector<vector<int64_t>> output_shapes = {{1, 32, 1, 1, 16}, {1, 32, 1, 1, 16}};
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_scale, output_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_offset, output_shapes[1], dtypes[0], ge::FORMAT_NC1HWC0, {});

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(runInfo.GetBlockDim(), 32);
    EXPECT_EQ(runInfo.GetTilingKey(), 1213400);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "256 32 28 28 1 14 ");
}

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling3) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    auto opParas = op::BNTrainingUpdateGrad(op_name.c_str());

    vector<vector<int64_t>> input_shapes = {
        {2, 2, 256, 256, 16}, {2, 2, 256, 256, 16}, {1, 2, 1, 1, 16}, {1, 2, 1, 1, 16}};

    vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_mean, input_shapes[2], dtypes[2], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_variance, input_shapes[3], dtypes[3], ge::FORMAT_NC1HWC0, {});

    vector<vector<int64_t>> output_shapes = {{1, 2, 1, 1, 16}, {1, 2, 1, 1, 16}};
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_scale, output_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_offset, output_shapes[1], dtypes[0], ge::FORMAT_NC1HWC0, {});

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(runInfo.GetBlockDim(), 32);
    EXPECT_EQ(runInfo.GetTilingKey(), 2213400);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 256 256 8 1 ");
}

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling4) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
    auto opParas = op::BNTrainingUpdateGrad(op_name.c_str());

    vector<vector<int64_t>> input_shapes = {
        {2, 2, 16, 16, 16}, {2, 2, 16, 16, 16}, {1, 2, 1, 1, 16}, {1, 2, 1, 1, 16}};

    vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_mean, input_shapes[2], dtypes[2], ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, batch_variance, input_shapes[3], dtypes[3], ge::FORMAT_NC1HWC0, {});

    vector<vector<int64_t>> output_shapes = {{1, 2, 1, 1, 16}, {1, 2, 1, 1, 16}};
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_scale, output_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, diff_offset, output_shapes[1], dtypes[0], ge::FORMAT_NC1HWC0, {});

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(runInfo.GetBlockDim(), 32);
    EXPECT_EQ(runInfo.GetTilingKey(), 5213400);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 16 16 2 1 ");
}
