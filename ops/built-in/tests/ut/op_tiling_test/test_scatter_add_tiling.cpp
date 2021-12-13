#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class ScatterAddTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterAddTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterAddTiling TearDown" << std::endl;
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

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(ScatterAddTiling, scatter_add_tiling_0) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 2 17 704 0 2112 1486848 0 704 23232 0 23232 176 88 33 1408 0 0 0 0 0 0 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_1) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{3,2210};
  std::vector<int64_t> inputB{258,1};
  std::vector<int64_t> inputC{258,1,2210};
  std::vector<int64_t> output{3,2210};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_INT32, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "14 1 3 2210 0 258 570180 0 2210 6630 0 2210 0 0 3 0 0 0 0 0 0 0 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_2) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{21340,1};
  std::vector<int64_t> inputB{21340,};
  std::vector<int64_t> inputC{21340,1};
  std::vector<int64_t> output{21340,1};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 0 32 1 0 0 0 0 0 0 0 0 0 0 21340 0 667 663 0 667 0 663 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_3) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 17 704 0 2112 1486848 0 704 0 0 0 0 0 33 0 0 0 0 0 0 0 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_4) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 22};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_5) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> inputC{1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_6) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_7) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_8) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_9) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_10) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_11) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_13) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4, \"support_atomic\": 0}}";

  std::vector<int64_t> inputA{};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterAddTiling, scatter_add_tiling_14) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4, \"support_atomic\": 0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(ScatterAddTiling, scatter_add_tiling_15) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":0, \"core_num\":0, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{21340,1};
  std::vector<int64_t> inputB{21340,};
  std::vector<int64_t> inputC{21340,1};
  std::vector<int64_t> output{21340,1};

  auto opParas = op::ScatterMax("ScatterAdd");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}