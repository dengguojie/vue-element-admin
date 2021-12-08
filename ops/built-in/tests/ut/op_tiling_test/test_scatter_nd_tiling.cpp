#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class ScatterNdTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterNdTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterNdTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
  return result;
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_0) {
  // using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,7,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shapeValue{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7,11};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shapeValue);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "14 112 32 4235 0 210 296450 0 4235 15118950 0 4235 59290 51878 3570 474320 3 35 7 0 0 0 0 0 19 21968 17 10294 0 0 0 0 0 0 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_1) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{31037,1};
  std::vector<int64_t> inputB{31037,256};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{300000,256};
  std::vector<int64_t> output{300000,256};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shape);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "17 9375 32 256 0 0 0 0 0 0 0 0 0 0 300000 0 1 0 0 0 0 0 0 0 0 0 0 0 970 967 0 970 0 967 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_2) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{279424,1};
  std::vector<int64_t> inputB{279424,1};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{279424,1};
  std::vector<int64_t> output{279424,1};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shape);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 0 32 1 0 0 0 0 0 0 0 0 0 0 279424 0 1 0 0 0 0 0 0 0 0 0 0 0 8732 8732 0 8732 0 8732 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_3) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{21340,1,2};
  std::vector<int64_t> inputB{21340,1};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{640000,1};
  std::vector<int64_t> output{640000,1};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shape);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 0 1 1 1 10936 21340 0 21340 0 0 0 0 0 640000 0 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_4) {
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,7,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shape{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shape);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_5) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 0, \"core_num\": 0, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,7,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shape{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7,11};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shape);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_INT32, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_7) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{21340,1};
  std::vector<int64_t> inputB{21340,1};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{640000,1};
  std::vector<int64_t> output{640000,1};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shape);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 0 32 1 0 0 0 0 0 0 0 0 0 0 640000 0 1 0 0 0 0 0 0 0 0 0 0 0 667 663 0 667 0 663 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_6) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,6,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shape{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7,11};

  auto opParas = op::ScatterNd("ScatterNd");

  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, inputC, ge::DT_INT32, ge::FORMAT_ND, shape);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
