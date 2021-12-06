#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "matrix_calculation_ops.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
using namespace std;
using namespace ut_util;
using namespace ge;


class ScatterUpdateTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterUpdateTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterUpdateTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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

TEST_F(ScatterUpdateTiling, scatter_update_tiling_0) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 17 704 0 2112 1486848 0 704 ");
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_1) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{1039, 34, 23, 31};
  std::vector<int64_t> inputB{1039};
  std::vector<int64_t> inputC{1039, 34, 23, 31};
  std::vector<int64_t> output{1039, 34, 23, 31};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 33 32 24242 0 1039 25187438 0 24242 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_4) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 22};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_5) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> inputC{1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_6) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_7) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_8) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_9) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_10) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_13) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4}}";

  std::vector<int64_t> inputA{};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_14) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, scatter_update_tiling_15) {
  std::string op_name = "ScatterUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterUpdate");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 0, \"core_num\": 0, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterUpdate("ScatterUpdate");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterUpdateTiling, inplace_update_tiling_1) {
  std::string op_name = "InplaceUpdate";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::InplaceUpdate(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, DT_FLOAT, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, DT_INT32, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, v, inputC, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 17 704 0 2112 1486848 0 704 1408 0 0 1408 704 0 0 704 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}
